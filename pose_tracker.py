"""
pose_tracker.py

Live skeletal pose tracking + real-time form COACHING for both pathways --
a browser-side MediaPipe Pose Landmarker component (~30fps) that:

  - draws a skeleton overlay and measures the joint angle relevant to the
    selected exercise
  - counts reps AND judges each one on range of motion, left/right symmetry,
    tempo, and (for squats) knee tracking -- with per-rep feedback
  - speaks every rep count and every fault out loud (blind pathway) while
    also showing it as big text (deaf pathway) -- one feature, both users
  - supports multiple exercises (squat, bicep curl, overhead press, lateral
    raise), switchable from an in-frame dropdown WITHOUT restarting the camera
  - manages sets hands-free: a new set begins automatically on your first rep
    after stopping (no gesture -- the old "wave overhead to start" collided
    with overhead exercises); cross your arms in an X to END a set
  - measures joint angles from MediaPipe's 3D WORLD landmarks (depth-aware),
    not flat 2D image coordinates, so perspective doesn't distort the angles

Why browser-side JS instead of Python mediapipe: the Python package risks a
protobuf conflict with Streamlit; the JS build runs client-side via CDN,
needs zero new Python dependencies, keeps every frame on the user's machine
(most private path in the app), and is fast enough for 30fps + audio.

IMPORTANT -- every form/gesture threshold lives in the EX (per-exercise) and
CFG objects near the top of the JS. They are STARTING POINTS: "how deep is
deep enough," "how much knee-cave is a fault," "what counts as a wave" can
only be calibrated against a real body on camera. Expect to tune them during
testing, and to tune each exercise separately.

Same secure-origin rule as the camera/mic: the webcam needs localhost/HTTPS.
"""

POSE_TRACKER_HEIGHT = 800

# Kept as a plain string (not an f-string) -- the JS is full of braces.
POSE_TRACKER_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<style>
  :root {
    --accent: #007AFF; --success: #34C759; --warn: #FF9F0A; --error: #FF3B30;
    --text: rgba(255,255,255,0.92); --muted: rgba(255,255,255,0.6); --surface: #131316;
    --font: "Barlow", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  html, body { margin: 0; padding: 0; background: transparent; font-family: var(--font); color: var(--text); }
  .wrap { display: flex; flex-direction: column; gap: 12px; }
  .toprow { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .toprow label { font-size: 14px; color: var(--muted); }
  select { font-family: var(--font); font-size: 15px; font-weight: 600; color: var(--text); background: var(--surface); border: 1.5px solid rgba(255,255,255,0.18); border-radius: 10px; padding: 8px 12px; }
  .stage { position: relative; width: 100%; background: var(--surface); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; overflow: hidden; }
  video { display: block; width: 100%; transform: scaleX(-1); }
  canvas { position: absolute; inset: 0; width: 100%; height: 100%; transform: scaleX(-1); }
  .hud { position: absolute; top: 10px; left: 10px; right: 10px; display: flex; justify-content: space-between; align-items: flex-start; pointer-events: none; }
  .chips { display: flex; gap: 8px; }
  .chip { background: rgba(0,0,0,0.68); border-radius: 10px; padding: 8px 14px; }
  .chip .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  .chip .value { font-size: 30px; font-weight: 700; line-height: 1.1; }
  .depthwrap { position: absolute; right: 10px; bottom: 64px; top: 84px; width: 14px; background: rgba(0,0,0,0.55); border-radius: 7px; overflow: hidden; }
  .depthfill { position: absolute; bottom: 0; left: 0; right: 0; height: 0%; background: var(--accent); transition: height 80ms linear, background 200ms; }
  .cue { position: absolute; left: 50%; bottom: 52px; transform: translateX(-50%); font-size: 26px; font-weight: 800; letter-spacing: 0.03em; background: rgba(0,0,0,0.68); padding: 6px 18px; border-radius: 999px; text-align: center; }
  .form { position: absolute; left: 10px; right: 10px; bottom: 10px; font-size: 18px; font-weight: 600; text-align: center; background: rgba(0,0,0,0.62); border-radius: 10px; padding: 8px 12px; min-height: 22px; }
  .statusbar { font-size: 15px; color: var(--muted); min-height: 22px; }
  .hints { font-size: 14px; color: var(--muted); line-height: 1.5; }
  .hints b { color: var(--text); }
  .sound { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; color: var(--muted); cursor: pointer; user-select: none; }
</style>
</head>
<body>
<div class="wrap">
  <div class="toprow">
    <label for="exsel">Exercise</label>
    <select id="exsel">
      <option value="squat">Squat</option>
      <option value="curl">Bicep curl</option>
      <option value="press">Overhead press</option>
      <option value="raise">Lateral raise</option>
    </select>
    <label class="sound"><input type="checkbox" id="soundToggle" checked> Speak reps &amp; feedback out loud</label>
  </div>
  <div class="stage">
    <video id="video" autoplay playsinline muted></video>
    <canvas id="overlay"></canvas>
    <div class="hud">
      <div class="chips">
        <div class="chip"><div class="label">Set</div><div class="value" id="setn">1</div></div>
        <div class="chip"><div class="label">Reps</div><div class="value" id="reps">0</div></div>
      </div>
      <div class="chip"><div class="label" id="anglabel">Angle</div><div class="value" id="angle">--</div></div>
    </div>
    <div class="depthwrap"><div class="depthfill" id="depth"></div></div>
    <div class="cue" id="cue" style="display:none"></div>
    <div class="form" id="form">Pick an exercise, then stand back so the joints being tracked are all in frame.</div>
  </div>
  <div class="statusbar" id="status" role="status">Loading pose model (first load downloads ~6MB)...</div>
  <div class="hints">
    <b>Sets are automatic:</b> a new set begins on your <b>first rep</b>, and ends when you <b>stop for a few seconds</b> (or cross your arms in an X) &mdash; it then reads out your rep count.
  </div>
</div>

<script type="module">
import { PoseLandmarker, FilesetResolver, DrawingUtils } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ---- Per-exercise config (CALIBRATE against a real body) ------------------
// joints: [a,b,c] landmark indices for the tracked angle, left and right.
// dir: 'below' = the active/contracted position is a SMALLER angle (squat
//   bottom, curl top); 'above' = a LARGER angle (press lockout, raise top).
// enter/exit: angle to enter the active phase / return to rest (completes rep).
// romGood/romShallow: full range vs. too-short range, for depth feedback.
// rest/full: angles mapping the range bar 0%->100%.
// valgus: squat-only knee-tracking check.
const EX = {
  squat: { name:"Squat", angLabel:"Knee", L:[23,25,27], R:[24,26,28], dir:"below",
    enter:140, exit:155, romGood:95, romShallow:115, rest:170, full:90, valgus:true,
    romMsg:"Go deeper next rep", romSay:"Go deeper" },
  curl:  { name:"Bicep curl", angLabel:"Elbow", L:[11,13,15], R:[12,14,16], dir:"below",
    enter:90, exit:150, romGood:55, romShallow:85, rest:160, full:45, valgus:false,
    romMsg:"Curl all the way up", romSay:"Curl higher" },
  press: { name:"Overhead press", angLabel:"Elbow", L:[11,13,15], R:[12,14,16], dir:"above",
    enter:125, exit:105, romGood:150, romShallow:130, rest:90, full:160, valgus:false,
    romMsg:"Press all the way up", romSay:"Press higher" },
  raise: { name:"Lateral raise", angLabel:"Shoulder", L:[23,11,13], R:[24,12,14], dir:"above",
    enter:70, exit:35, romGood:80, romShallow:60, rest:20, full:90, valgus:false,
    romMsg:"Raise up to shoulder height", romSay:"Raise higher" },
};
const CFG = {
  VIS:0.5, SYM_TOL:25, TEMPO_MIN_SEC:0.5, VALGUS_RATIO:0.80,
  GESTURE_HOLD_MS:600, GESTURE_COOLDOWN_MS:3000, SET_IDLE_MS:10000,
};

const $=(id)=>document.getElementById(id);
const statusEl=$("status"),video=$("video"),canvas=$("overlay"),ctx=canvas.getContext("2d");
const repsEl=$("reps"),setnEl=$("setn"),angleEl=$("angle"),angLabelEl=$("anglabel"),depthEl=$("depth"),cueEl=$("cue"),formEl=$("form"),soundToggle=$("soundToggle"),exsel=$("exsel");

let landmarker=null,drawingUtils=null,lastVideoTime=-1;
let cur=EX.squat;

// ---- Audio ----------------------------------------------------------------
let audioCtx=null;
function ensureAudio(){ try{ audioCtx=audioCtx||new (window.AudioContext||window.webkitAudioContext)(); if(audioCtx.state==="suspended") audioCtx.resume(); }catch(e){} }
document.addEventListener("pointerdown",ensureAudio); document.addEventListener("keydown",ensureAudio);
function beep(freq,dur){ if(!soundToggle.checked) return; ensureAudio(); if(!audioCtx) return;
  try{ const t=audioCtx.currentTime,d=dur||0.12,o=audioCtx.createOscillator(),g=audioCtx.createGain();
    o.type="sine"; o.frequency.value=freq||880;
    g.gain.setValueAtTime(0.0001,t); g.gain.exponentialRampToValueAtTime(0.25,t+0.01); g.gain.exponentialRampToValueAtTime(0.0001,t+d);
    o.connect(g); g.connect(audioCtx.destination); o.start(t); o.stop(t+d+0.02);
  }catch(e){} }
function say(text){ if(!soundToggle.checked) return; try{ const u=new SpeechSynthesisUtterance(text); u.rate=1.1; speechSynthesis.cancel(); speechSynthesis.speak(u); }catch(e){} }

// ---- Geometry -------------------------------------------------------------
function vis(p){ return p && (p.visibility ?? 1) >= CFG.VIS; }
// Angle from 3D WORLD landmarks (wl, in real-world metres with an estimated
// depth/z) rather than flat 2D image coordinates -- so a knee-over-toe or a
// forward lean is measured in true 3D instead of distorted by the camera's
// perspective. Visibility is still gated on the image landmarks (il), which
// carry the confidence score.
function angleOf(wl, il, j){
  const a=wl[j[0]],b=wl[j[1]],c=wl[j[2]];
  if(!a||!b||!c||!vis(il[j[0]])||!vis(il[j[1]])||!vis(il[j[2]])) return null;
  const v1={x:a.x-b.x,y:a.y-b.y,z:a.z-b.z},v2={x:c.x-b.x,y:c.y-b.y,z:c.z-b.z};
  const dot=v1.x*v2.x+v1.y*v2.y+v1.z*v2.z, m1=Math.hypot(v1.x,v1.y,v1.z), m2=Math.hypot(v2.x,v2.y,v2.z);
  if(m1===0||m2===0) return null;
  return Math.acos(Math.min(1,Math.max(-1,dot/(m1*m2))))*180/Math.PI;
}

// ---- Rep state machine + per-rep form metrics -----------------------------
// A set begins automatically on the first rep after you've stopped (no
// "start" gesture -- that used to collide with overhead exercises). setStarted
// flips true on rep 1; ending a set (cross arms) flips it back to false.
let setNum=1,reps=0,everStarted=false,setStarted=false,lastRepTime=0;
let active=false,repExtreme=0,repSym=0,repValgus=9,descentStart=0,bottomTime=0;
const isActive=(a)=> cur.dir==="below" ? a<cur.enter : a>cur.enter;
const isRest  =(a)=> cur.dir==="below" ? a>cur.exit  : a<cur.exit;
const better  =(a,e)=> cur.dir==="below" ? a<e : a>e;
const romOK   =(e)=> cur.dir==="below" ? e<=cur.romGood : e>=cur.romGood;
const romBad  =(e)=> cur.dir==="below" ? e>cur.romShallow : e<cur.romShallow;

function evaluateRep(){
  const descentSec=(bottomTime-descentStart)/1000;
  if(romBad(repExtreme)) return {msg:cur.romMsg, say:cur.romSay, color:"var(--warn)", good:false};
  if(cur.valgus && repValgus<CFG.VALGUS_RATIO) return {msg:"Push your knees out", say:"Knees out", color:"var(--warn)", good:false};
  if(repSym>CFG.SYM_TOL) return {msg:"Keep both sides even", say:"Even it out", color:"var(--warn)", good:false};
  if(descentSec>0 && descentSec<CFG.TEMPO_MIN_SEC) return {msg:"Control it -- slower", say:"Slower", color:"var(--warn)", good:false};
  return {msg:"Good rep", say:"", color:"var(--success)", good:true};
}
function showCue(t,c){ cueEl.textContent=t; cueEl.style.color=c; cueEl.style.display="block"; }
function showForm(t,c){ formEl.textContent=t; formEl.style.color=c||"var(--text)"; }
function onRepComplete(){
  // Auto-start a set on the first rep after a stop -- the set doesn't exist
  // until rep 1 is counted (your suggested design). No gesture needed to begin.
  var setAnnounce="";
  if(!setStarted){
    if(everStarted) setNum+=1;
    everStarted=true; setStarted=true; reps=0; setnEl.textContent=setNum;
    setAnnounce="Set "+setNum+". ";
  }
  reps+=1; repsEl.textContent=reps; lastRepTime=performance.now();
  const f=evaluateRep();
  beep(f.good?880:520,0.12);
  say(setAnnounce + (f.say ? (reps+". "+f.say) : String(reps)));
  showCue((setAnnounce?("SET "+setNum+" — "):"")+"REP "+reps,"var(--text)");
  showForm(f.msg,f.color);
}

// ---- Gesture: cross arms in an X to END a set -----------------------------
// (The "wave to start a set" gesture was removed: it false-triggered whenever
// the arms went overhead, e.g. during an overhead press. New sets now begin
// automatically on the first rep -- see onRepComplete.)
let gestureCooldownUntil=0,crossHold=0;
function detectGestures(lm,now){
  if(now<gestureCooldownUntil) return;
  const Ls=lm[11],Rs=lm[12],Lh=lm[23],Rh=lm[24],Lw=lm[15],Rw=lm[16];
  // Only the shoulders need to be clearly visible -- crossing the arms
  // occludes the wrists, so requiring high wrist visibility (the old bug)
  // meant the gesture could never fire. MediaPipe still returns estimated
  // wrist positions when occluded; we use those.
  if(!vis(Ls)||!vis(Rs)||!Lw||!Rw){ crossHold=0; return; }
  const shoulderY=(Ls.y+Rs.y)/2, hipY=(vis(Lh)&&vis(Rh))?(Lh.y+Rh.y)/2:shoulderY+0.3;
  // Arms crossed = the wrists' left-right order is FLIPPED relative to the
  // shoulders' order (orientation-agnostic, no midline threshold to miss),
  // with both hands up around the chest/shoulders.
  const shouldersOrder=Math.sign(Ls.x-Rs.x), wristsOrder=Math.sign(Lw.x-Rw.x);
  const upperBody = Lw.y<hipY+0.05 && Rw.y<hipY+0.05 && Lw.y>shoulderY-0.25 && Rw.y>shoulderY-0.25;
  const crossed = upperBody && wristsOrder!==0 && wristsOrder!==shouldersOrder;
  if(crossed){ if(!crossHold) crossHold=now; if(now-crossHold>=CFG.GESTURE_HOLD_MS){ endSet(); gestureCooldownUntil=now+CFG.GESTURE_COOLDOWN_MS; crossHold=0; } } else crossHold=0;
}
function endSet(){
  if(!setStarted) return;  // no set in progress to end
  beep(660,0.1); setTimeout(()=>beep(440,0.14),120);
  say("Set "+setNum+" complete. "+reps+(reps===1?" rep.":" reps."));
  showCue("SET "+setNum+" — "+reps+(reps===1?" REP":" REPS"),"var(--accent)");
  showForm("Set ended. Your next set starts automatically on your first rep.","var(--muted)");
  setStarted=false;
}

// ---- Exercise switch (no camera restart) ----------------------------------
exsel.addEventListener("change",()=>{
  cur=EX[exsel.value]; active=false; setStarted=false; reps=0; repsEl.textContent=0; angLabelEl.textContent=cur.angLabel;
  say("Now tracking "+cur.name); showForm("Now tracking the "+cur.name+". Your next rep starts a new set.","var(--text)");
});

// ---- Main loop ------------------------------------------------------------
async function init(){
  try{
    const fileset=await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    landmarker=await PoseLandmarker.createFromOptions(fileset,{
      baseOptions:{ modelAssetPath:"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", delegate:"GPU" },
      runningMode:"VIDEO", numPoses:1 });
  }catch(e){ statusEl.textContent="Couldn't load the pose model -- check your internet connection and refresh."; return; }
  statusEl.textContent="Starting camera...";
  try{ video.srcObject=await navigator.mediaDevices.getUserMedia({video:{width:960,height:540}}); }
  catch(e){ statusEl.textContent="Camera blocked. Use http://localhost:8501 (not a network address) and allow camera access."; return; }
  video.addEventListener("loadeddata",()=>{ canvas.width=video.videoWidth; canvas.height=video.videoHeight;
    statusEl.textContent="Tracking. Switch exercises above; sets start on your first rep, cross your arms to end one.";
    angLabelEl.textContent=cur.angLabel; requestAnimationFrame(loop); });
}
function loop(){
  if(video.currentTime!==lastVideoTime){
    lastVideoTime=video.currentTime;
    const result=landmarker.detectForVideo(video,performance.now());
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const now=performance.now();
    if(result.landmarks&&result.landmarks.length>0){
      const lm=result.landmarks[0];
      // 3D world landmarks for depth-aware angles (fall back to image
      // landmarks if the model didn't return them this frame).
      const wl=(result.worldLandmarks&&result.worldLandmarks[0])?result.worldLandmarks[0]:lm;
      if(!drawingUtils) drawingUtils=new DrawingUtils(ctx);
      drawingUtils.drawConnectors(lm,PoseLandmarker.POSE_CONNECTIONS,{color:"#007AFF",lineWidth:4});
      drawingUtils.drawLandmarks(lm,{color:"#FFFFFF",radius:4});

      const la=angleOf(wl,lm,cur.L), ra=angleOf(wl,lm,cur.R);
      const a=(la!==null&&ra!==null)?(cur.dir==="below"?Math.min(la,ra):Math.max(la,ra)):(la ?? ra);
      const sym=(la!==null&&ra!==null)?Math.abs(la-ra):0;

      detectGestures(lm,now);
      // Reliable, gesture-free set end: if you've stopped repping for a few
      // seconds, the set ends and reads out its count. (Cross-arms is the
      // faster manual option.)
      if(setStarted && lastRepTime && (now-lastRepTime>CFG.SET_IDLE_MS)) endSet();

      if(a!==null){
        angleEl.textContent=Math.round(a)+"°";
        const frac=(a-cur.rest)/(cur.full-cur.rest);
        depthEl.style.height=Math.min(100,Math.max(0,frac*100))+"%";
        depthEl.style.background = romOK(a) ? "var(--success)" : "var(--accent)";

        let valgus=9;
        if(cur.valgus && vis(lm[25])&&vis(lm[26])&&vis(lm[27])&&vis(lm[28])){
          const kw=Math.abs(lm[25].x-lm[26].x), aw=Math.abs(lm[27].x-lm[28].x); if(aw>0.01) valgus=kw/aw;
        }
        if(!active && isActive(a)){ active=true; repExtreme=a; repSym=sym; repValgus=valgus; descentStart=now; bottomTime=now; }
        else if(active){
          if(better(a,repExtreme)){ repExtreme=a; bottomTime=now; }
          if(sym>repSym) repSym=sym; if(valgus<repValgus) repValgus=valgus;
          if(isRest(a)){ active=false; onRepComplete(); }
        }
      } else { angleEl.textContent="--"; if(now>=gestureCooldownUntil) showForm("Step back so the tracked joints are fully in frame.","var(--warn)"); }
    } else { angleEl.textContent="--"; if(now>=gestureCooldownUntil) showForm("No one in frame -- step in front of the camera.","var(--error)"); crossHold=0; }
  }
  requestAnimationFrame(loop);
}
init();
</script>
</body>
</html>
"""
