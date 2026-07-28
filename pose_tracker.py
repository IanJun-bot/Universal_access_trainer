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

POSE_TRACKER_HEIGHT = 980

# High-Contrast override for the tracker iframe. Iframes see none of the
# parent page's CSS, so the app injects this in place of /*THEME_OVERRIDE*/
# (see render call in app.py) when High-Contrast mode is on. It redeclares
# the same :root variables the styles below already consume -- including the
# skeleton color, which the JS reads from --accent at startup.
TRACKER_HC_CSS = """
  :root {
    --accent: #FFFF00; --success: #00FF66; --warn: #FF9F0A; --error: #FF5C4D;
    --text: #FFFFFF; --muted: #FFFF00; --surface: #000000;
    --chipbg: #000000;
  }
  select { border-color: #FFFF00 !important; }
  .stage { border: 2px solid #FFFF00 !important; }
  .chip, .cue, .form { border: 1px solid #FFFF00; }
  .kin { border: 2px solid #FFFF00 !important; }
  .kin-phase { border-color: #FFFF00 !important; }
  .kin-bar { background: #1a1a1a !important; }
"""

# Kept as a plain string (not an f-string) -- the JS is full of braces.
POSE_TRACKER_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<style>
  /* Iframes do NOT inherit the parent page's fonts -- without this import the
     tracker silently falls back to the system stack while the rest of the app
     renders Barlow. */
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@400;500;600;700&display=swap');
  :root {
    --accent: #007AFF; --success: #34C759; --warn: #FF9F0A; --error: #FF3B30;
    --text: rgba(255,255,255,0.92); --muted: rgba(255,255,255,0.6); --surface: #131316;
    --chipbg: rgba(0,0,0,0.68);
    --font: "Barlow", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --font-display: "Barlow Condensed", "Barlow", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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
  .chip { background: var(--chipbg); border-radius: 10px; padding: 8px 14px; }
  .chip .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  .chip .value { font-family: var(--font-display); font-size: 32px; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
  .depthwrap { position: absolute; right: 10px; bottom: 64px; top: 84px; width: 14px; background: rgba(0,0,0,0.55); border-radius: 7px; overflow: hidden; }
  .depthfill { position: absolute; bottom: 0; left: 0; right: 0; height: 0%; background: var(--accent); transition: height 80ms linear, background 200ms; }
  .cue { position: absolute; left: 50%; bottom: 52px; transform: translateX(-50%); font-family: var(--font-display); font-size: 28px; font-weight: 700; letter-spacing: 0.03em; background: var(--chipbg); padding: 6px 18px; border-radius: 999px; text-align: center; }
  .form { position: absolute; left: 10px; right: 10px; bottom: 10px; font-size: 18px; font-weight: 600; text-align: center; background: var(--chipbg); border-radius: 10px; padding: 8px 12px; min-height: 22px; }
  .statusbar { font-size: 15px; color: var(--muted); min-height: 22px; }
  .hints { font-size: 14px; color: var(--muted); line-height: 1.5; }
  .hints b { color: var(--text); }
  .sound { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; color: var(--muted); cursor: pointer; user-select: none; }
  /* Live joint kinematics readout (golf-launch-monitor style): one row per
     tracked quantity -- label, range bar, numeric degrees. */
  .kin { background: var(--surface); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }
  .kin-head { display: flex; align-items: center; gap: 10px; }
  .kin-title { font-family: var(--font-display); font-weight: 700; font-size: 15px; letter-spacing: 0.08em; text-transform: uppercase; }
  .kin-phase { font-family: var(--font-display); font-weight: 700; font-size: 14px; letter-spacing: 0.08em; background: var(--chipbg); border: 1px solid rgba(255,255,255,0.18); border-radius: 6px; padding: 2px 10px; min-width: 64px; text-align: center; }
  .kin-clock { margin-left: auto; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .kin-row { display: grid; grid-template-columns: 128px 1fr 72px; gap: 10px; align-items: center; }
  .kin-label { font-size: 13px; color: var(--muted); }
  .kin-bar { height: 8px; background: rgba(255,255,255,0.10); border-radius: 4px; overflow: hidden; }
  .kin-fill { height: 100%; width: 0%; background: var(--accent); transition: width 60ms linear, background 200ms; }
  .kin-val { font-family: var(--font-display); font-size: 18px; font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }
  .kin-lastrep { font-size: 13px; color: var(--muted); border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px; }
  .kin-lastrep b { color: var(--text); font-variant-numeric: tabular-nums; }
  /*THEME_OVERRIDE*/
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
  <!-- aria-hidden: this panel repaints ~30x/second, which would flood a
       screen reader -- the spoken per-rep feedback is the accessible channel
       for the same information. -->
  <div class="kin" id="kin" aria-hidden="true">
    <div class="kin-head">
      <span class="kin-title">Live joint data</span>
      <span class="kin-phase" id="kinPhase">REST</span>
      <span class="kin-clock" id="kinClock">t=0.00s&nbsp;&nbsp;f0</span>
    </div>
    <div id="kinRows"></div>
    <div class="kin-lastrep" id="kinLast">Last rep: — complete a rep to see its peak angle, left/right gap, and descent time (use these to calibrate the thresholds).</div>
  </div>
  <div class="statusbar" id="status" role="status">Loading pose model (first load downloads ~6MB)...</div>
  <div class="hints">
    <b>Sets are automatic:</b> a new set begins on your <b>first rep</b>, and ends when you <b>stop for a few seconds</b> (or cross your arms in an X) &mdash; it then reads out your rep count.
  </div>
</div>

<script type="module">
import { PoseLandmarker, FilesetResolver, DrawingUtils } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// Skeleton color follows the CSS theme (--accent), so High-Contrast mode
// re-colors the drawn overlay too, not just the DOM around it.
const SKELETON_COLOR = getComputedStyle(document.documentElement)
  .getPropertyValue("--accent").trim() || "#007AFF";

// ---- Per-exercise config (CALIBRATE against a real body) ------------------
// joints: [a,b,c] landmark indices for the tracked angle, left and right.
// dir: 'below' = the active/contracted position is a SMALLER angle (squat
//   bottom, curl top); 'above' = a LARGER angle (press lockout, raise top).
// enter/exit: angle to enter the active phase / return to rest (completes rep).
// romGood/romShallow: full range vs. too-short range, for depth feedback.
// rest/full: angles mapping the range bar 0%->100%.
// valgus: squat-only knee-tracking check.
// tiltWarn: spine-tilt (degrees from vertical) beyond which the live data
//   panel flags the torso row -- a forward lean is normal in a squat but is
//   a cheat/arch signal in the arm exercises.
// Squat/curl thresholds re-baselined July 2026 for 3D world-landmark angles,
// applying the compression the press taught us (extremes read ~10-15 deg
// less extreme in 3D than 2D): deep-flexion targets loosened accordingly.
// VERIFY against the live data panel's "Last rep" peak readout on a real
// body, then pin the numbers.
const EX = {
  squat: { name:"Squat", angLabel:"Knee", L:[23,25,27], R:[24,26,28], dir:"below",
    // Calibrated 2026-07-22 against a real body: honest full-depth bottom
    // read peak 122 deg on 3D world landmarks -- romGood sits ~4 deg above.
    enter:140, exit:155, romGood:126, romShallow:135, rest:170, full:118, valgus:true,
    tiltWarn:50, romMsg:"Go deeper next rep", romSay:"Go deeper" },
  curl:  { name:"Bicep curl", angLabel:"Elbow", L:[11,13,15], R:[12,14,16], dir:"below",
    // Calibrated 2026-07-22 against a real body: honest full-squeeze top
    // read peak 68 deg on 3D world landmarks -- romGood sits ~4 deg above.
    enter:100, exit:150, romGood:72, romShallow:95, rest:155, full:64, valgus:false,
    tiltWarn:15, romMsg:"Curl all the way up", romSay:"Curl higher" },
  press: { name:"Overhead press", angLabel:"Elbow", L:[11,13,15], R:[12,14,16], dir:"above",
    enter:125, exit:105, romGood:150, romShallow:130, rest:90, full:160, valgus:false,
    tiltWarn:15, romMsg:"Press all the way up", romSay:"Press higher" },
  raise: { name:"Lateral raise", angLabel:"Shoulder", L:[23,11,13], R:[24,12,14], dir:"above",
    enter:70, exit:35, romGood:80, romShallow:60, rest:20, full:90, valgus:false,
    tiltWarn:15, romMsg:"Raise up to shoulder height", romSay:"Raise higher" },
};
const CFG = {
  VIS:0.5, SYM_TOL:25, TEMPO_MIN_SEC:0.5, VALGUS_RATIO:0.80,
  GESTURE_HOLD_MS:600, GESTURE_COOLDOWN_MS:3000, SET_IDLE_MS:10000,
};

const $=(id)=>document.getElementById(id);
const statusEl=$("status"),video=$("video"),canvas=$("overlay"),ctx=canvas.getContext("2d");
const repsEl=$("reps"),setnEl=$("setn"),angleEl=$("angle"),angLabelEl=$("anglabel"),depthEl=$("depth"),cueEl=$("cue"),formEl=$("form"),soundToggle=$("soundToggle"),exsel=$("exsel");
const kinPhaseEl=$("kinPhase"),kinClockEl=$("kinClock"),kinRowsEl=$("kinRows"),kinLastEl=$("kinLast");

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

// ---- Live joint kinematics readout ---------------------------------------
// Golf-launch-monitor-style data rows: per-joint angles, spine tilt, and the
// left/right gap, live at camera rate. The "Last rep" line captures each
// rep's peak angle / symmetry / descent time -- the numbers used to
// calibrate the EX thresholds against a real body.
function midpt(p,q){ return {x:(p.x+q.x)/2,y:(p.y+q.y)/2,z:(p.z+q.z)/2}; }
// Torso tilt from vertical, in degrees, from 3D world landmarks. World y is
// down-positive, so an upright hip->shoulder vector points along -y (tilt 0).
function spineTiltDeg(wl,il){
  if(!vis(il[11])||!vis(il[12])||!vis(il[23])||!vis(il[24])) return null;
  const s=midpt(wl[11],wl[12]), h=midpt(wl[23],wl[24]);
  const dx=s.x-h.x, dy=s.y-h.y, dz=s.z-h.z;
  const horiz=Math.hypot(dx,dz);
  if(horiz===0&&dy===0) return null;
  return Math.atan2(horiz,-dy)*180/Math.PI;
}
// Row definitions. frac maps the value onto the 0..1 bar; color may return a
// CSS value to recolor the fill (default accent).
function kinRowDefs(){
  const jointFrac=(v)=>Math.min(1,Math.max(0,(v-cur.rest)/(cur.full-cur.rest)));
  const jointColor=(v)=>romOK(v)?"var(--success)":"var(--accent)";
  const rows=[
    { id:"L", label:cur.angLabel+" (L)", frac:jointFrac, color:jointColor },
    { id:"R", label:cur.angLabel+" (R)", frac:jointFrac, color:jointColor },
  ];
  if(cur.valgus) rows.push({ id:"hip", label:"Hip angle", frac:jointFrac, color:()=> "var(--accent)" });
  rows.push({ id:"tilt", label:"Spine tilt", frac:(v)=>Math.min(1,v/60),
    color:(v)=> v>cur.tiltWarn?"var(--warn)":"var(--accent)" });
  rows.push({ id:"sym", label:"L/R gap", frac:(v)=>Math.min(1,v/40),
    color:(v)=> v>CFG.SYM_TOL?"var(--warn)":"var(--accent)" });
  return rows;
}
let kinDefs=[],kinEls={};
function rebuildKin(){
  kinDefs=kinRowDefs(); kinEls={}; kinRowsEl.innerHTML="";
  for(const d of kinDefs){
    const row=document.createElement("div"); row.className="kin-row";
    row.innerHTML='<span class="kin-label"></span><div class="kin-bar"><div class="kin-fill"></div></div><span class="kin-val">--</span>';
    row.querySelector(".kin-label").textContent=d.label;
    kinRowsEl.appendChild(row);
    kinEls[d.id]={ fill:row.querySelector(".kin-fill"), val:row.querySelector(".kin-val") };
  }
}
let kinFrame=0,angTrend=0,lastAngle=null;
function updateKin(values,now){
  kinFrame+=1;
  kinClockEl.textContent="t="+video.currentTime.toFixed(2)+"s  f"+kinFrame;
  for(const d of kinDefs){
    const el=kinEls[d.id], v=values[d.id];
    if(v===null||v===undefined||Number.isNaN(v)){ el.val.textContent="--"; el.fill.style.width="0%"; continue; }
    el.val.textContent=Math.round(v)+"°";
    el.fill.style.width=(d.frac(v)*100).toFixed(0)+"%";
    el.fill.style.background=d.color(v);
  }
  // Phase from the smoothed angle trend: moving toward the contracted
  // extreme = DOWN (squat/curl) or UP (press/raise); away = the reverse.
  const a=values.primary;
  if(a!==null&&a!==undefined){
    if(lastAngle!==null) angTrend=0.7*angTrend+0.3*(a-lastAngle);
    lastAngle=a;
    let phase="REST";
    if(active){
      const towardExtreme = cur.dir==="below" ? angTrend<-0.5 : angTrend>0.5;
      const awayFromExtreme = cur.dir==="below" ? angTrend>0.5 : angTrend<-0.5;
      phase = towardExtreme ? (cur.dir==="below"?"DOWN":"UP")
            : awayFromExtreme ? (cur.dir==="below"?"UP":"DOWN") : "HOLD";
    }
    kinPhaseEl.textContent=phase;
  }
}
let sessionBest=null;
function updateKinLastRep(descentSec){
  if(sessionBest===null||better(repExtreme,sessionBest)) sessionBest=repExtreme;
  kinLastEl.innerHTML="Last rep: peak <b>"+Math.round(repExtreme)+"°</b> · L/R gap <b>"
    +Math.round(repSym)+"°</b> · <b>"+descentSec.toFixed(1)+"s</b> down · session best peak <b>"
    +Math.round(sessionBest)+"°</b> (rep counts past "+cur.enter+"°, full range past "+cur.romGood+"°)";
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
  updateKinLastRep(Math.max(0,(bottomTime-descentStart)/1000));
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
  rebuildKin(); lastAngle=null; angTrend=0; sessionBest=null;
  kinLastEl.textContent="Last rep: — complete a rep to see its peak angle, left/right gap, and descent time.";
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
  catch(e){ statusEl.textContent="Camera blocked. Open the app at a localhost address (not a network IP) and allow camera access."; return; }
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
      drawingUtils.drawConnectors(lm,PoseLandmarker.POSE_CONNECTIONS,{color:SKELETON_COLOR,lineWidth:4});
      drawingUtils.drawLandmarks(lm,{color:"#FFFFFF",radius:4});

      const la=angleOf(wl,lm,cur.L), ra=angleOf(wl,lm,cur.R);
      const a=(la!==null&&ra!==null)?(cur.dir==="below"?Math.min(la,ra):Math.max(la,ra)):(la ?? ra);
      const sym=(la!==null&&ra!==null)?Math.abs(la-ra):0;

      // Feed the live kinematics panel every processed frame.
      const tilt=spineTiltDeg(wl,lm);
      let hipA=null;
      if(cur.valgus){
        const lh=angleOf(wl,lm,[11,23,25]), rh=angleOf(wl,lm,[12,24,26]);
        hipA=(lh!==null&&rh!==null)?(lh+rh)/2:(lh ?? rh);
      }
      updateKin({L:la,R:ra,hip:hipA,tilt:tilt,sym:(la!==null&&ra!==null)?sym:null,primary:a},now);

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
rebuildKin();
init();
</script>
</body>
</html>
"""
