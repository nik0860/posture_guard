"""
PostureGuard - Real-Time Posture Correction (single-file web version)
--------------------------------------------------------------------
Everything lives in this file: the Flask server, the HTML page, the
styling, and the JavaScript that runs MediaPipe Pose in the user's
browser. No templates/ or static/ folders required.

Tracks TWO things only, both around the neck/shoulder region:
  1. Neck ratio    -> vertical gap between ear midpoint and shoulder
                      midpoint, normalized by shoulder width.
  2. Shoulder tilt -> angle of the line between the two shoulders.

No eye, nose, or mouth landmarks are read or drawn - ears are used
only as a geometric anchor because MediaPipe Pose has no dedicated
"neck" point.

The posture model runs in the browser via MediaPipe Tasks Vision, so
the Flask server does almost nothing: it serves this one page. That
makes free tiers (Render, Koyeb, etc.) more than enough.

Local run:
    pip install -r requirements.txt
    python posture_guard.py
    open http://127.0.0.1:5000

Deploy:
    Push to GitHub, point Render at the repo with:
        build:  pip install -r requirements.txt
        start:  gunicorn posture_guard:app
    Set PYTHON_VERSION=3.11.11 in the environment.
"""

import os
from flask import Flask, Response

app = Flask(__name__)

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PostureGuard</title>
<style>
  :root {
    --bg: #1b2430;
    --panel: #121a24;
    --text: #edeff2;
    --muted: #8b98a8;
    --accent: #7fb69e;
    --accent-dim: #4d6e63;
    --bad: #e2685f;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    min-height: 100vh;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 24px 64px;
  }

  header {
    max-width: 480px;
    text-align: center;
    margin-bottom: 32px;
  }

  h1 {
    font-size: 2rem;
    font-weight: 650;
    letter-spacing: -0.01em;
    margin: 0 0 10px;
  }

  header p {
    color: var(--muted);
    font-size: 0.98rem;
    line-height: 1.5;
    margin: 0;
  }

  .frame {
    width: min(640px, 92vw);
    aspect-ratio: 4 / 3;
    background: var(--panel);
    border: 1px solid #26313f;
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }

  .frame canvas {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: none;
  }

  .placeholder {
    color: var(--muted);
    font-size: 0.95rem;
    text-align: center;
    padding: 0 32px;
  }

  .badge {
    margin-top: 14px;
    display: inline-block;
    font-size: 0.9rem;
    font-weight: 650;
    letter-spacing: 0.01em;
    padding: 7px 16px;
    border-radius: 999px;
    border: 1px solid var(--muted);
    color: var(--muted);
    transition: color 0.2s ease, border-color 0.2s ease;
  }

  .controls {
    display: flex;
    gap: 12px;
    margin-top: 28px;
    flex-wrap: wrap;
    justify-content: center;
  }

  button {
    font: inherit;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 12px 22px;
    border-radius: 7px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: opacity 0.15s ease;
  }

  button:disabled {
    opacity: 0.35;
    cursor: default;
  }

  #startBtn {
    background: var(--accent);
    color: #0d1712;
  }

  #calibrateBtn {
    background: transparent;
    color: var(--text);
    border-color: #2e3a48;
  }

  #stopBtn {
    background: transparent;
    color: var(--bad);
    border-color: #4a2e2c;
  }

  button:not(:disabled):hover {
    opacity: 0.85;
  }

  .hint {
    margin-top: 18px;
    color: var(--muted);
    font-size: 0.85rem;
    text-align: center;
    min-height: 1.2em;
    max-width: 560px;
    line-height: 1.5;
  }

  .hint.error {
    color: var(--bad);
  }
</style>
</head>
<body>

  <header>
    <h1>PostureGuard</h1>
    <p>Watches your neck and shoulders through your webcam and tells you when you start to slouch or lean. Nothing else on your face is tracked.</p>
  </header>

  <div class="frame">
    <canvas id="canvas" width="640" height="480"></canvas>
    <div class="placeholder" id="placeholder">Camera is off. Press Start to begin.</div>
  </div>

  <video id="video" playsinline muted autoplay style="display:none"></video>

  <div class="badge" id="badge">Camera is off</div>

  <div class="controls">
    <button id="startBtn">Start</button>
    <button id="calibrateBtn" disabled>Calibrate</button>
    <button id="stopBtn" disabled>Stop</button>
  </div>

  <p class="hint" id="hint">Sit up straight, then press Calibrate to set your baseline.</p>

  <script type="module">
    import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

    // -------------------------------------------------------------
    // CONFIG
    // -------------------------------------------------------------
    const NECK_RATIO_TOLERANCE_PCT = 15;
    const SHOULDER_TILT_TOLERANCE_DEG = 8;
    const ALERT_AFTER_FRAMES = 20;
    const VISIBILITY_THRESHOLD = 0.5;

    const LEFT_EAR = 7;
    const RIGHT_EAR = 8;
    const LEFT_SHOULDER = 11;
    const RIGHT_SHOULDER = 12;

    const MODEL_URL =
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
    const WASM_URL =
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

    const C_OK = "#00ff00";
    const C_BAD = "#ff0000";
    const C_WARN = "#ffff00";
    const C_NONE = "#c8c8c8";
    const C_OFF = "#8b98a8";

    // -------------------------------------------------------------
    // DOM
    // -------------------------------------------------------------
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const placeholder = document.getElementById("placeholder");
    const badge = document.getElementById("badge");
    const hint = document.getElementById("hint");
    const startBtn = document.getElementById("startBtn");
    const calibrateBtn = document.getElementById("calibrateBtn");
    const stopBtn = document.getElementById("stopBtn");

    // -------------------------------------------------------------
    // STATE
    // -------------------------------------------------------------
    let landmarker = null;
    let running = false;
    let rafId = null;
    let lastInferenceTime = -1;
    let baselineNeck = null;
    let baselineTilt = null;
    let badStreak = 0;
    let calibrateRequested = false;

    // -------------------------------------------------------------
    // HELPERS
    // -------------------------------------------------------------
    function setHint(text, isError = false) {
      hint.textContent = text;
      hint.classList.toggle("error", isError);
    }

    function setBadge(text, color) {
      badge.textContent = text;
      badge.style.color = color;
      badge.style.borderColor = color;
    }

    function dimColor(hex) {
      const r = parseInt(hex.slice(1, 3), 16) / 2;
      const g = parseInt(hex.slice(3, 5), 16) / 2;
      const b = parseInt(hex.slice(5, 7), 16) / 2;
      return `rgb(${r | 0},${g | 0},${b | 0})`;
    }

    function angleDiff(a, b) {
      const d = Math.abs(a - b) % 360;
      return d > 180 ? 360 - d : d;
    }

    // -------------------------------------------------------------
    // POSE METRICS
    // -------------------------------------------------------------
    function computeMetrics(lms, w, h) {
      const lEar = lms[LEFT_EAR];
      const rEar = lms[RIGHT_EAR];
      const lSh = lms[LEFT_SHOULDER];
      const rSh = lms[RIGHT_SHOULDER];
      if (!lEar || !rEar || !lSh || !rSh) return null;

      const vis = [lEar.visibility, rEar.visibility, lSh.visibility, rSh.visibility]
        .map(v => (v == null ? 0 : v));
      if (Math.min(...vis) < VISIBILITY_THRESHOLD) return null;

      const lEarP = { x: lEar.x * w, y: lEar.y * h };
      const rEarP = { x: rEar.x * w, y: rEar.y * h };
      const lShP = { x: lSh.x * w, y: lSh.y * h };
      const rShP = { x: rSh.x * w, y: rSh.y * h };

      const shWidth = Math.hypot(lShP.x - rShP.x, lShP.y - rShP.y);
      if (shWidth < 1e-3) return null;

      const earMid = { x: (lEarP.x + rEarP.x) / 2, y: (lEarP.y + rEarP.y) / 2 };
      const shMid = { x: (lShP.x + rShP.x) / 2, y: (lShP.y + rShP.y) / 2 };
      const verticalGap = shMid.y - earMid.y;
      const neckRatio = verticalGap / shWidth;
      if (neckRatio < 1e-6) return null;

      const tilt = Math.atan2(rShP.y - lShP.y, rShP.x - lShP.x) * 180 / Math.PI;

      return { neckRatio, tilt, points: { lEarP, rEarP, lShP, rShP, earMid, shMid } };
    }

    function drawOverlay(pts, color, w) {
      const M = p => ({ x: w - p.x, y: p.y });
      const lEar = M(pts.lEarP);
      const rEar = M(pts.rEarP);
      const lSh = M(pts.lShP);
      const rSh = M(pts.rShP);
      const earMid = M(pts.earMid);
      const shMid = M(pts.shMid);
      const dim = dimColor(color);

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(lSh.x, lSh.y); ctx.lineTo(rSh.x, rSh.y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(earMid.x, earMid.y); ctx.lineTo(shMid.x, shMid.y); ctx.stroke();

      ctx.strokeStyle = dim;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(lEar.x, lEar.y); ctx.lineTo(lSh.x, lSh.y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(rEar.x, rEar.y); ctx.lineTo(rSh.x, rSh.y); ctx.stroke();

      ctx.fillStyle = color;
      for (const p of [lSh, rSh, earMid]) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // -------------------------------------------------------------
    // MODEL LOADING
    // -------------------------------------------------------------
    async function ensureLandmarker() {
      if (landmarker) return landmarker;
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
      landmarker = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
      });
      return landmarker;
    }

    // -------------------------------------------------------------
    // MAIN LOOP
    // -------------------------------------------------------------
    function loop() {
      if (!running) return;
      rafId = requestAnimationFrame(loop);

      if (video.readyState < 2 || !landmarker) return;

      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w || !h) return;

      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }

      ctx.save();
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, w, h);
      ctx.restore();

      const now = performance.now();
      if (now === lastInferenceTime) return;
      lastInferenceTime = now;

      let result;
      try {
        result = landmarker.detectForVideo(video, now);
      } catch (e) {
        return;
      }

      let statusText = "No pose detected";
      let statusColor = C_NONE;

      if (result.landmarks && result.landmarks.length > 0) {
        const m = computeMetrics(result.landmarks[0], w, h);

        if (m) {
          if (calibrateRequested) {
            baselineNeck = m.neckRatio;
            baselineTilt = m.tilt;
            badStreak = 0;
            calibrateRequested = false;
            calibrateBtn.disabled = false;
          }

          if (baselineNeck == null) {
            statusText = "Press Calibrate to set your baseline";
            statusColor = C_WARN;
          } else {
            const dropPct = ((baselineNeck - m.neckRatio) / baselineNeck) * 100;
            const tiltDiff = angleDiff(m.tilt, baselineTilt);
            const isBad =
              dropPct > NECK_RATIO_TOLERANCE_PCT ||
              tiltDiff > SHOULDER_TILT_TOLERANCE_DEG;

            badStreak = isBad ? badStreak + 1 : 0;

            if (badStreak >= ALERT_AFTER_FRAMES) {
              statusText = "Bad posture - sit up straight";
              statusColor = C_BAD;
            } else {
              statusText = "Posture OK";
              statusColor = C_OK;
            }
          }

          drawOverlay(m.points, statusColor, w);
        }
      }

      ctx.strokeStyle = statusColor;
      ctx.lineWidth = 8;
      ctx.strokeRect(4, 4, w - 8, h - 8);
      ctx.font = "bold 26px -apple-system, Segoe UI, Roboto, sans-serif";
      ctx.fillStyle = statusColor;
      ctx.textBaseline = "top";
      ctx.fillText(statusText, 20, 18);

      setBadge(statusText, statusColor);
    }

    // -------------------------------------------------------------
    // BUTTONS
    // -------------------------------------------------------------
    startBtn.addEventListener("click", async () => {
      startBtn.disabled = true;
      setHint("Loading pose model...");

      try {
        await ensureLandmarker();
      } catch (e) {
        setHint("Could not load the pose model: " + e.message, true);
        startBtn.disabled = false;
        return;
      }

      setHint("Starting the camera...");
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
      } catch (e) {
        setHint(
          "Could not open the camera: " + e.message +
          ". Check permissions and that no other app is using it.",
          true
        );
        startBtn.disabled = false;
        return;
      }

      video.srcObject = stream;
      try {
        await video.play();
      } catch (e) {
        setHint("Could not start video playback: " + e.message, true);
        stream.getTracks().forEach(t => t.stop());
        video.srcObject = null;
        startBtn.disabled = false;
        return;
      }

      running = true;
      baselineNeck = null;
      baselineTilt = null;
      badStreak = 0;
      calibrateRequested = false;
      lastInferenceTime = -1;

      canvas.style.display = "block";
      placeholder.style.display = "none";
      calibrateBtn.disabled = false;
      stopBtn.disabled = false;
      setHint("Sit up straight, then press Calibrate to set your baseline.");
      setBadge("Looking for you...", C_WARN);

      loop();
    });

    calibrateBtn.addEventListener("click", () => {
      if (!running) return;
      calibrateRequested = true;
      calibrateBtn.disabled = true;
      setHint("Holding still - capturing baseline on the next clear pose...");
    });

    stopBtn.addEventListener("click", () => {
      running = false;
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }

      const stream = video.srcObject;
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
        video.srcObject = null;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      canvas.style.display = "none";
      placeholder.style.display = "block";

      startBtn.disabled = false;
      calibrateBtn.disabled = true;
      stopBtn.disabled = true;
      calibrateRequested = false;
      baselineNeck = null;
      baselineTilt = null;
      badStreak = 0;

      setBadge("Camera is off", C_OFF);
      setHint("Sit up straight, then press Calibrate to set your baseline.");
    });
  </script>

</body>
</html>
"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html; charset=utf-8")


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        threaded=True,
    )
