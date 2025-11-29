# web_co_painter.py
#
# pip install fastapi uvicorn opencv-python numpy
# python -m uvicorn web_co_painter:app \
#   --reload \
#   --host 0.0.0.0 \
#   --port 16868
#
# Then open: http://localhost:16868

import base64
import cv2, os
import numpy as np
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from config import CANVAS_SIZE, CONTROL_PROMPT, ADVERSARIAL_PROMPT, SUBSEQUENT_PROMPT, CONDITIONS
# from image_to_svg import image_to_svg
from paint_with_gemini import (
    get_gemini_drawing,
    init_gemini_api,
    get_model,
    API_KEY,
)
from daehwa_api_key import d_api_key
from dataclasses import dataclass
from typing import Optional, Dict
import random


app = FastAPI()

# --- App state / model init -------------------------------------------------


@dataclass
class SessionState:
    mode: str
    prompt: str
    turn_idx: int
    last_drawing: Optional[np.ndarray]
    api_key: Optional[str] = None
    model: Optional[any] = None
    condition: Optional[Dict] = None
    mode_seq: Optional[list] = None

@app.on_event("startup")
async def startup_event():
    """Initialize Gemini model once when server starts."""
    # init_gemini_api(API_KEY)
    # app.state.model = get_model()
    app.state.sessions: Dict[str, SessionState] = {}
    app.state.save_folder = "saved_images"
    app.state.api_key = d_api_key  # default API key
    init_gemini_api(d_api_key)
    app.state.model = get_model()



# --- Request models ---------------------------------------------------------

class ImagePayload(BaseModel):
    image: str  # data URL: "data:image/png;base64,...."
    participant: str | None


class ModePayload(BaseModel):
    mode: str   # "collaborative" or "adversarial"
    participant: str

class ApiKeyPayload(BaseModel):
    api_key: str
    participant: str

class ParticipantPayload(BaseModel):
    participant: str

class ResetPayload(BaseModel):
    participant: str

# --- Helpers ----------------------------------------------------------------

def get_or_create_session(participant: str, default_mode: str = "adversarial") -> SessionState:
    sessions = app.state.sessions
    if participant not in sessions:
        sessions[participant] = SessionState(
            mode=default_mode,
            prompt=ADVERSARIAL_PROMPT,
            turn_idx=0,
            last_drawing=None,
            api_key=d_api_key,
            condition=CONDITIONS[default_mode],
        )
    return sessions[participant]


def data_url_to_cv2_image(data_url: str) -> np.ndarray:
    """Convert data URL (PNG) from canvas to BGR numpy array shaped like CANVAS_SIZE."""
    if "," in data_url:
        _, b64data = data_url.split(",", 1)
    else:
        b64data = data_url
    raw = base64.b64decode(b64data)
    nparr = np.frombuffer(raw, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize to match your CANVAS_SIZE (H, W, 3)
    if len(CANVAS_SIZE) == 3:
        H, W, _ = CANVAS_SIZE
    else:
        H, W = CANVAS_SIZE[:2]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    return img


def cv2_image_to_data_url(img: np.ndarray) -> str:
    """Encode BGR numpy image to PNG data URL for the browser canvas."""
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# --- HTML UI ----------------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" name="viewport" content="width=device-width, initial-scale=1"/>
  <title>HRI Robot Co-Painter</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: #111;
      color: #eee;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 1.5rem;
    }
    #canvas-wrapper {
      border: 1px solid #444;
      background: #fff;
      position: relative;
      display: inline-block;
      margin: 0 auto;
    }
    #main-ui {
      display: flex;
      flex-direction: column;
      align-items: center;
      max-width: 350px;
    }
    canvas {
      display: block;
      cursor: crosshair;
      width: 100%;              /* fill available width */
      max-width: {{WIDTH}}px;   /* but never bigger than original */
      height: auto;             /* keep aspect ratio */
    }
    #login-ui {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      margin-top: 32px;
      width: 100%;
      max-width: 400px;
    }
    #login-ui p {
      margin: 0;
      text-align: center;
      font-size: 0.95rem;
      color: #ccc;
    }
    #participant-name {
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #444;
      background: #222;
      color: #eee;
      font-size: 1rem;
      box-sizing: border-box;
    }
    #start-session {
      width: 100%;
      padding: 10px 16px;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      background: #2b7cff;
      color: white;
      font-weight: 600;
      font-size: 1rem;
    }
    #controls {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: center;
      align-items: center;
      max-width: {{WIDTH}}px;
      width: 100%;
    }
    button {
      padding: 8px 16px;
      border-radius: 6px;
      border: none;
      cursor: pointer;
      background: #2b7cff;
      color: white;
      font-weight: 500;
    }
    button.secondary {
      background: #333;
    }
    button:disabled {
      opacity: 0.6;
      cursor: wait;
    }

    /* Mobile tweaks */
    @media (max-width: 600px) {
      body {
        padding: 12px;
        gap: 12px;
      }

      #login-ui {
        margin-top: 24px;
        max-width: 100%;
      }

      h1 {
        font-size: 1.25rem;
        text-align: center;
      }

      #participant-name,
      #start-session {
        font-size: 1rem;
      }

      #controls {
        flex-direction: column;
        align-items: stretch;
      }

      #controls button,
      #api-key-controls button,
      #api-key-controls input {
        width: 100%;
      }

      #api-key-controls {
        max-width: {{WIDTH}}px;
        width: 100%;
      }

      h1 {
        font-size: 1.2rem;
        text-align: center;
      }
    }

    select {
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid #444;
      background: #222;
      color: #eee;
    }
    #status {
      max-width: {{WIDTH}}px;
      font-size: 0.9rem;
      color: #ccc;
      min-height: 1.2em;
      text-align: center;
    }
    #gemini-output-wrapper {
      max-width: {{WIDTH}}px;
      width: 100%;
      margin: 0 auto;
    }
    #gemini-output {
      white-space: pre-wrap;
      font-size: 0.9rem;
      background: #181818;
      border-radius: 8px;
      padding: 10px 12px;
      border: 1px solid #333;
      min-height: 2.4em;

      max-height: 150px;     /* <<< hard cap on height */
      overflow-y: auto;      /* scroll when text is long */

      overflow-wrap: break-word; /* avoid long tokens expanding width */
      word-break: break-word;
    }
  </style>
</head>
<body>

  <div id="login-ui">
    <h1>HRI Robot Co-Painter</h1>
    <p>Your task is to draw a landscape together with an agent. You will start the drawing and the agent will add to the drawing. When it's turn is done, it will pass back control of the canvas to you. You have three turns to complete the drawing with the agent.</p>
    <p>Please enter your participant number to begin.</p>
    <input id="participant-name" type="text" placeholder="Participant number" />
    <button id="start-session">Start</button>
  </div>

  <div id="main-ui" style="display:none;">
    <h1>HRI Robot Co-Painter</h1>
    <p>Draw a landscape together with an agent. Note that the agent does not know the task, it can only see what you draw.</p>
    <div id="canvas-wrapper">
      <canvas id="drawingCanvas" width="{{WIDTH}}" height="{{HEIGHT}}"></canvas>
    </div>

    <div id="controls">
      <button id="ask-ai">Pass to Agent</button>
      <button id="clear" class="secondary">Clear</button>
      <!--<button id="save-svg" class="secondary">Save SVG on Server</button>-->

      <label for="mode-select" style="margin-left:8px;">Mode:</label>
      <!--<select id="mode-select">
        <option value="collaborative">Collaborative</option>
        <option value="adversarial" selected>Adversarial</option>
        <option value="antagonistic">Antagonistic</option>
      </select>-->
      <select id="mode-select">
        <option value="custom_visual-similar_semantic-similar" selected>Mode1</option>
        <option value="custom_visual-similar_semantic-neutral">Mode2</option>
        <option value="custom_visual-similar_semantic-different">Mode3</option>
        <option value="custom_visual-neutral_semantic-similar">Mode4</option>
        <option value="custom_visual-neutral_semantic-neutral">Mode5</option>
        <option value="custom_visual-neutral_semantic-different">Mode6</option>
        <option value="custom_visual-different_semantic-similar">Mode7</option>
        <option value="custom_visual-different_semantic-neutral">Mode8</option>
        <option value="custom_visual-different_semantic-different">Mode9</option>
      </select>
    </div>

    <!--<div id="api-key-controls" style="margin-top:12px;">
      <label for="api-key-input" style="margin-left:12px;">API key:</label>
      <input id="api-key-input" type="password" placeholder="Enter Agent API key" size="28" />
      <button id="set-api-key" class="secondary">Set</button>
    </div>-->


    <div id="status"></div>

    <div id="gemini-output-wrapper" hidden>
      <div id="gemini-output-label">Agent explanation / thoughts:</div>
      <div id="gemini-output">(no output yet)</div>
    </div>
  </div>

<script>
  const canvas = document.getElementById('drawingCanvas');
  const ctx = canvas.getContext('2d');

  const CANVAS_W = canvas.width;
  const CANVAS_H = canvas.height;

  // init white background
  function resetCanvas() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }
  resetCanvas();

  let drawing = false;
  let lastX = 0;
  let lastY = 0;

  function getPos(evt) {
    const rect = canvas.getBoundingClientRect();
    if (evt.touches && evt.touches.length > 0) {
      const t = evt.touches[0];
      return {
        x: t.clientX - rect.left,
        y: t.clientY - rect.top,
      };
    } else {
      return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top,
      };
    }
  }

  function startDrawing(evt) {
    // If a session end is pending, handle it instead of starting a new stroke
    if (pendingSessionEnd) {
      const info = pendingSessionEnd;
      pendingSessionEnd = null;  // consume it
      geminiOutputEl.textContent = '(no text output from Gemini)';

      if (info.studyDone) {
        // Finished the last mode -> whole study done
        alert('Study is finished. Thank you for participating!');

        // Return to login screen
        mainUi.style.display = 'none';
        loginUi.style.display = 'block';

        // Reset participant & UI state
        participantName = null;
        if (participantNameInput) participantNameInput.value = '';
        if (modeSelect) modeSelect.value = 'adversarial';  // default
        setBusy(false, 'Study finished.');
      } else{
        alert(`Drawing session ended. Switching to next session.`);

        // Clear canvas for the new session
        resetCanvas();

        // Update dropdown to the next mode (so UI matches backend state)
        if (modeSelect && info.nextMode) {
          modeSelect.value = info.nextMode;
        }
        if (askAiBtn) {
          askAiBtn.disabled = false;
        }

        setBusy(false, `New drawing session started.`);
      }

      // Don't start drawing on this attempt; user can draw after the popup
      return;
    }

    // Normal drawing path
    drawing = true;
    const pos = getPos(evt);
    lastX = pos.x;
    lastY = pos.y;
  }


  function draw(evt) {
    if (!drawing) return;
    evt.preventDefault();
    const pos = getPos(evt);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    lastX = pos.x;
    lastY = pos.y;
  }

  function stopDrawing() {
    drawing = false;
  }

  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseleave', stopDrawing);

  canvas.addEventListener('touchstart', startDrawing, { passive: false });
  canvas.addEventListener('touchmove', draw, { passive: false });
  canvas.addEventListener('touchend', stopDrawing);
  canvas.addEventListener('touchcancel', stopDrawing);

  const askAiBtn = document.getElementById('ask-ai');
  const clearBtn = document.getElementById('clear');
  const saveSvgBtn = document.getElementById('save-svg');
  const modeSelect = document.getElementById('mode-select');
  const statusEl = document.getElementById('status');
  const geminiOutputEl = document.getElementById('gemini-output');
  const loginUi = document.getElementById('login-ui');
  const mainUi = document.getElementById('main-ui');
  const participantNameInput = document.getElementById('participant-name');
  const startSessionBtn = document.getElementById('start-session');

  let participantName = null;
  let pendingSessionEnd = null;  // { studyDone: boolean, nextMode: string } or null


  if (modeSelect) {
    modeSelect.addEventListener('change', async () => {
      const mode = modeSelect.value;  // "collaborative" / "adversarial" / "antagonistic"
      console.log('Mode changed in UI to:', mode);  // debug

      try {

        const payload = {
          mode: mode,
          participant: participantName,
        };

        setBusy(true, `Switching mode to "${mode}" ...`);
        const res = await fetch('/api/set_mode', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const errJson = await res.json().catch(() => ({}));
          console.error('Error response from /api/set_mode:', errJson);
          throw new Error(errJson.error || 'Server error');
        }

        const json = await res.json();
        console.log('Server replied from /api/set_mode:', json);
        setBusy(false, json.message || `Mode set to ${mode}.`);
      } catch (err) {
        console.error('Exception when calling /api/set_mode:', err);
        setBusy(false, 'Error switching mode. See console.');
      }
    });
  } else {
    console.error('modeSelect is null – check id="mode-select" in HTML');
  }


  function setBusy(isBusy, message) {
    askAiBtn.disabled = isBusy;
    clearBtn.disabled = isBusy;
    modeSelect.disabled = isBusy;
    if (saveSvgBtn) {
      saveSvgBtn.disabled = isBusy;
    }
    statusEl.textContent = message || '';
  }

  startSessionBtn.addEventListener('click', async () => {
    const name = participantNameInput.value.trim();
    if (!name) {
      alert('Please enter your participant number.');
      return;
    }

    participantName = name;
    console.log('Participant number:', participantName);

    // Hide login screen, show main UI
    loginUi.style.display = 'none';
    mainUi.style.display = 'block';  // or 'flex' if you use flexbox

    try {
      setBusy(true, 'Starting session...');
      const res = await fetch('/api/start_session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ participant: participantName }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        console.error('Error response from /api/start_session:', errJson);
        alert(errJson.error || 'Failed to start session.');
        setBusy(false);
        return;
      }

      const json = await res.json();

      if (json.study_done) {
        // This participant finished all modes already
        alert('This study is already finished for you. Thank you!');
        // Stay on / return to login page, reset fields
        participantName = null;
        participantNameInput.value = '';
        loginUi.style.display = 'block';
        mainUi.style.display = 'none';
        return;
      }

      const initialMode = json.mode || 'adversarial';

      // Show main UI, set dropdown to chosen mode, clear canvas
      loginUi.style.display = 'none';
      mainUi.style.display = 'block';

      if (modeSelect) {
        modeSelect.value = initialMode;
      }

      resetCanvas();
      setBusy(false, `Drawing session started`);
    } catch (err) {
      console.error('Exception in startSession:', err);
      setBusy(false);
      alert('Error starting session. Probably no API key typed.');
    }
  });

  askAiBtn.addEventListener('click', async () => {
    try {
      setBusy(true, 'Asking Gemini to add to your drawing...');
      const dataUrl = canvas.toDataURL('image/png');

      const payload = {
        image: dataUrl,
        participant: participantName,
      };
      const res = await fetch('/api/ai_draw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        throw new Error('Server error');
      }
      const json = await res.json();
      const newImg = new Image();
      newImg.onload = () => {
        ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
        ctx.drawImage(newImg, 0, 0, CANVAS_W, CANVAS_H);

        // Show Gemini text if you have it
        if (json.text && json.text.trim().length > 0) {
          geminiOutputEl.textContent = json.text;
        } else {
          geminiOutputEl.textContent = '(no text output from Gemini)';
        }

        if (json.session_ended) {
          // Don't popup yet; store what we should do later
          pendingSessionEnd = {
            studyDone: !!json.study_done,
            nextMode: json.next_mode || 'adversarial',
          };

          setBusy(false, 'Final drawing. Press anywhere on canvas to continue.');
          if (askAiBtn) {
            askAiBtn.disabled = true;
          }

        } else {
          setBusy(false, 'AI updated the drawing.');
        }
      };

      newImg.src = json.image;

      // Show Gemini's text output (if any)
      if (json.text && json.text.trim().length > 0) {
        geminiOutputEl.textContent = json.text;
      } else {
        geminiOutputEl.textContent = '(no text output from Gemini)';
      }
    } catch (err) {
      console.error(err);
      setBusy(false, 'Error talking to AI. Probably no API key typed.');
    }
  });

  clearBtn.addEventListener('click', async () => {
    try {
      setBusy(true, 'Reverting to previous drawing...');
      const res = await fetch('/api/reset_to_old', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ participant: participantName }),
      });

      const json = await res.json();

      if (!json.has_old || !json.image) {
        // Nothing to undo: just clear to blank
        resetCanvas();
        setBusy(false, 'No previous drawing. Canvas cleared.');
        return;
      }

      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        setBusy(false, 'Reverted to previous drawing.');
      };
      img.src = json.image;
    } catch (err) {
      console.error('Error resetting to old drawing:', err);
      resetCanvas();
      setBusy(false, 'Error undoing. Canvas cleared instead.');
    }
  });
</script>
</body>
</html>
"""


# --- Routes -----------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the drawing UI."""
    if len(CANVAS_SIZE) == 3:
        H, W, _ = CANVAS_SIZE
    else:
        H, W = CANVAS_SIZE[:2]

    html = (
        HTML_TEMPLATE
        .replace("{{WIDTH}}", str(W))
        .replace("{{HEIGHT}}", str(H))
    )
    return HTMLResponse(content=html)

@app.post("/api/start_session")
async def api_start_session(payload: ParticipantPayload):
    """Start a session for a participant and choose mode based on folders."""
    participant = payload.participant.strip()
    if not participant:
        return JSONResponse({"error": "Participant number is empty."}, status_code=400)

    base_dir = f"saved_images/{participant}"
    os.makedirs(base_dir, exist_ok=True)

    # choose mode based on existing folders
    chosen_mode = None
    _mode_seq = list(CONDITIONS.keys())
    random.shuffle(_mode_seq)
    mode_seq = []
    for mode in _mode_seq:
        folder = os.path.join(base_dir, mode)  # saved_images/<participant>/<mode>
        howmany = 0
        if os.path.exists(folder):
            howmany = len([f for f in os.listdir(folder) if f.endswith(".png")]) if os.path.exists(folder) else 0

        if (not os.path.exists(folder)) or howmany < 6:
            mode_seq.append(mode)
    if len(mode_seq) != 0:
        chosen_mode = mode_seq[0]

    print(mode_seq)
    print(chosen_mode)

    if chosen_mode is None:
        # All modes already have folders: study is done for this participant
        print(f"Participant '{participant}' has completed all modes. Study done.")
        # Reset state to a neutral default
        app.state.sessions.pop(participant, None)

        return JSONResponse({"mode": None, "study_done": True})

    session = SessionState(
        mode=chosen_mode,
        condition=CONDITIONS[chosen_mode],
        prompt=CONTROL_PROMPT, # ADVERSARIAL_PROMPT
        turn_idx=0,
        last_drawing=None,
        mode_seq=mode_seq,
    )
    app.state.sessions[participant] = session


    print(f"Participant '{participant}' starting mode '{chosen_mode}'")

    return JSONResponse({"mode": chosen_mode})


@app.post("/api/ai_draw")
async def api_ai_draw(payload: ImagePayload):
    """Take current canvas image, call Gemini co-painter, return updated canvas + text."""
    try:
        if not payload.participant:
            return JSONResponse({"error": "Missing participant"}, status_code=400)
        participant = payload.participant.strip()
        if not participant:
            return JSONResponse({"error": "Empty participant"}, status_code=400)

        session = get_or_create_session(participant)

        img = data_url_to_cv2_image(payload.image)

        # old_drawing, new_drawing, combined_drawing, text_output = img.copy(), img.copy(), img.copy(), "hello" # for debugging
        # text_output = "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello "
        
        # Support both 3-return and 4-return versions of get_gemini_drawing.
        try:
            old_drawing, new_drawing, combined_drawing, text_output = get_gemini_drawing(
                img.copy(),
                session.prompt,
                app.state.model,
                session.condition,
            )
        except ValueError:
            old_drawing, new_drawing, combined_drawing = get_gemini_drawing(
                img.copy(),
                session.prompt,
                app.state.model,
                session.condition,
            )
            text_output = ""

        if old_drawing is not None:
            session.last_drawing = combined_drawing.copy()
        else:
            session.last_drawing = img.copy()

        # After first call, switch to subsequent prompt (shared for both modes for now)
        session.prompt = SUBSEQUENT_PROMPT

        # In your own code you can also choose to keep condition or change it here.

        if combined_drawing is None:
            combined_drawing = img

        # Make sure output matches canvas size
        if len(CANVAS_SIZE) == 3:
            H, W, _ = CANVAS_SIZE
        else:
            H, W = CANVAS_SIZE[:2]
        combined_drawing = cv2.resize(
            combined_drawing,
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        )

        pname = payload.participant or "unknown"
        mode = session.mode
        foldername = f"{app.state.save_folder}/{pname}/{mode}"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(foldername, exist_ok=True)
        save_path_user = f"{foldername}/turn_{session.turn_idx}_human.png"
        cv2.imwrite(save_path_user, old_drawing)
        save_path_robot = f"{foldername}/turn_{session.turn_idx}_robot.png"
        cv2.imwrite(save_path_robot, combined_drawing)
        print(f"[{participant}] Saved AI turn {session.turn_idx} in mode '{session.mode}'")

        session.turn_idx += 1

        session_ended = False
        study_done = False
        next_mode = session.mode

        if session.turn_idx >= 3:
            session_ended = True

            mode_seq = session.mode_seq
            mode_seq.pop(0)

            # If we are at the LAST mode and would wrap to first -> study done
            if len(mode_seq) == 0:
                study_done = True
                text_output = ""

                # Reset backend state to neutral default for future participants
                participant = payload.participant.strip() or "anonymous"
                print(
                    f"Study complete for '{participant}'. "
                    "All modes finished; returning to login."
                )
                session.participant_name = None
                session.condition = CONDITIONS.get(session.mode, None)
                session.prompt = CONTROL_PROMPT
                session.turn_idx = 0
                session.last_drawing = None
            else:
                # Normal: advance to next mode and start a new session
                next_mode = mode_seq[0]
                session.mode = next_mode
                session.condition = CONDITIONS.get(next_mode, None)
                session.prompt = CONTROL_PROMPT  # new session starts with initial prompt
                session.turn_idx = 0
                session.last_drawing = None

                participant = getattr(app.state, "participant_name", "anonymous")
                print(
                    f"Session ended for '{participant}'. "
                    f"Switching to mode='{next_mode}', resetting prompt & turn_idx."
                )
        # --------------------------------------------------------------------

        data_url = cv2_image_to_data_url(combined_drawing)
        return JSONResponse({
            "image": data_url,
            "text": text_output,
            "session_ended": session_ended,
            "next_mode": next_mode,
            "study_done": study_done,
        })
    except Exception as e:
        print("Error in /api/ai_draw:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/reset_to_old")
async def api_reset_to_old(payload: ResetPayload):
    """Return the last old_drawing so frontend can 'undo', or signal none."""
    try:
        participant = payload.participant.strip()
        if not participant:
            return JSONResponse({"has_old": False})
        session = app.state.sessions.get(participant)

        if session.last_drawing is None:
            # No previous drawing: tell frontend there's nothing to undo
            return JSONResponse({"has_old": False})

        # Ensure it matches canvas size
        if len(CANVAS_SIZE) == 3:
            H, W, _ = CANVAS_SIZE
        else:
            H, W = CANVAS_SIZE[:2]
        restored = cv2.resize(session.last_drawing, (W, H), interpolation=cv2.INTER_NEAREST)

        data_url = cv2_image_to_data_url(restored)
        return JSONResponse({
            "has_old": True,
            "image": data_url,
        })
    except Exception as e:
        print("Error in /api/reset_to_old:", e)
        # Even on error, avoid 400 to keep frontend simple
        return JSONResponse({"has_old": False, "error": str(e)})


# @app.post("/api/set_api_key")
# async def api_set_api_key(payload: ApiKeyPayload):
#     """Set / override Gemini API key at runtime."""
#     participant = payload.participant.strip()
#     if not participant:
#         return JSONResponse({"error": "Empty participant"}, status_code=400)

#     session = get_or_create_session(participant)
#     key = payload.api_key.strip()
#     if key == "daehwa's":
#       key = d_api_key
#     if not key:
#         return JSONResponse({"error": "API key is empty."}, status_code=400)
#     session.api_key = key
#     init_gemini_api(session.api_key)
#     session.model = get_model()

#     print(f"API key set for participant '{participant}'")
#     return JSONResponse({"message": "API key stored for this participant."})


@app.post("/api/save_svg")
async def api_save_svg(payload: ImagePayload):
    """Convert current canvas image into SVG path on server (robot_path.svg)."""
    try:
        img = data_url_to_cv2_image(payload.image)

        if img is None:
            raise RuntimeError("Decoded image is None")

        # Handle different channel counts safely
        if len(img.shape) == 2:
            # already grayscale
            gray = img
        elif len(img.shape) == 3:
            h, w, c = img.shape
            if c == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif c == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                raise RuntimeError(f"Unsupported channel count: {c}")
        else:
            raise RuntimeError(f"Unsupported image shape: {img.shape}")

        # create filename for current time
        mode = app.state.mode
        filename = mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S.svg")
        image_to_svg(gray, filename=filename)
        return JSONResponse({"message": f"Saved as {filename}"})
    except Exception as e:
        print("Error in /api/save_svg:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# 43cm x 35cm

@app.post("/api/set_mode")
async def api_set_mode(payload: ModePayload):
    """Switch between 'collaborative' and 'adversarial' modes."""
    mode = payload.mode.lower()
    participant = payload.participant.strip()
    if mode not in CONDITIONS.keys():
        return JSONResponse(
            {"error": f"Unknown mode '{payload.mode}'"},
            status_code=400,
        )
    session = get_or_create_session(participant)
    try:
        session.mode = mode
        session.condition = CONDITIONS[mode]
        # Reset prompt when switching modes so Gemini knows fresh context.
        session.prompt = CONTROL_PROMPT
        msg = f"Mode set to '{mode}'. Prompt reset to initial CONTROL_PROMPT."
        print(msg)
        return JSONResponse(
                    {
                        "ok": True,
                        "message": msg,
                        "mode": mode,
                        "participant": participant,
                    },
                    status_code=200,
                )
    except KeyError:
        # If CONDITIONS doesn't have this key, surface a helpful error.
        err = f"Mode '{mode}' not found in config.CONDITIONS."
        print(err)
        return JSONResponse({"error": err}, status_code=500)

if __name__ in {'__main__', '__mp_main__'}:
    uvicorn.run(app,
                host='0.0.0.0',
                port=16868,
                reload=False,
                )

# <!--setApiKeyBtn.addEventListener('click', async () => {
#     const key = apiKeyInput.value.trim();
#     if (!key) {
#       alert('Please paste your Gemini API key first.');
#       return;
#     }
#     if (!participantName) {
#       alert('Please enter your participant number and click Start first.');
#       return;
#     }

#     try {
#       setBusy(true, 'Saving API key...');
#       const res = await fetch('/api/set_api_key', {
#         method: 'POST',
#         headers: { 'Content-Type': 'application/json' },
#         body: JSON.stringify({
#           participant: participantName,
#           api_key: key,
#         }),
#       });

#       const json = await res.json();
#       if (!res.ok) {
#         console.error('Error response from /api/set_api_key:', json);
#         alert(json.error || 'Failed to set API key.');
#         setBusy(false);
#         return;
#       }

#       setBusy(false, 'API key saved for this participant.');
#     } catch (err) {
#       console.error('Exception while setting API key:', err);
#       setBusy(false, 'Error setting API key. See console.');
#     }
#   });

# const apiKeyInput = document.getElementById('api-key-input');
# const setApiKeyBtn = document.getElementById('set-api-key');