# web_co_painter.py
#
#   pip install fastapi uvicorn opencv-python numpy
#   uvicorn web_co_painter:app --reload --port 16868
#
# Then open: http://localhost:16868

import base64
import cv2, os
import numpy as np
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from config import CANVAS_SIZE, GEMINI_PROMPT, SUBSEQUENT_PROMPT, CONDITIONS
from image_to_svg import image_to_svg
from paint_with_gemini import (
    get_gemini_drawing,
    init_gemini_api,
    get_model,
    API_KEY,
)
from daehwa_api_key import d_api_key

app = FastAPI()

# --- App state / model init -------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize Gemini model once when server starts."""
    init_gemini_api(API_KEY)
    app.state.model = get_model()
    app.state.prompt = GEMINI_PROMPT
    app.state.mode = "adversarial"          # default mode
    app.state.condition = CONDITIONS["adversarial"]
    app.state.turn_idx = 0                  # if you added turn saving
    app.state.api_key = API_KEY             # track current key
    print("Gemini model initialized (mode=adversarial)")



# --- Request models ---------------------------------------------------------

class ImagePayload(BaseModel):
    image: str  # data URL: "data:image/png;base64,...."


class ModePayload(BaseModel):
    mode: str   # "collaborative" or "adversarial"


class ApiKeyPayload(BaseModel):
    api_key: str

# --- Helpers ----------------------------------------------------------------

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
  <meta charset="utf-8" />
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
    }
    canvas {
      display: block;
      cursor: crosshair;
    }
    #controls {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: center;
      align-items: center;
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
    select {
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid #444;
      background: #222;
      color: #eee;
    }
    #status {
      font-size: 0.9rem;
      color: #ccc;
      min-height: 1.2em;
      text-align: center;
    }
    #gemini-output-wrapper {
      max-width: 800px;
      width: 100%;
    }
    #gemini-output-label {
      font-size: 0.9rem;
      color: #aaa;
      margin-bottom: 4px;
    }
    #gemini-output {
      white-space: pre-wrap;
      font-size: 0.9rem;
      background: #181818;
      border-radius: 8px;
      padding: 10px 12px;
      border: 1px solid #333;
      min-height: 2.4em;
    }
  </style>
</head>
<body>
  <h1>HRI Robot Co-Painter</h1>
  <div id="canvas-wrapper">
    <canvas id="drawingCanvas" width="{{WIDTH}}" height="{{HEIGHT}}"></canvas>
  </div>

  <div id="controls">
    <button id="ask-ai">Ask AI to Add</button>
    <button id="clear" class="secondary">Clear</button>
    <!--<button id="save-svg" class="secondary">Save SVG on Server</button>-->

    <label for="mode-select" style="margin-left:8px;">Mode:</label>
    <select id="mode-select">
      <option value="collaborative">Collaborative</option>
      <option value="adversarial" selected>Adversarial</option>
    </select>
  </div>

  <div id="api-key-controls" style="margin-top:12px;">
    <label for="api-key-input" style="margin-left:12px;">API key:</label>
    <input id="api-key-input" type="password" placeholder="Enter Gemini API key" size="28" />
    <button id="set-api-key" class="secondary">Set</button>
  </div>


  <div id="status"></div>

  <div id="gemini-output-wrapper">
    <div id="gemini-output-label">Gemini explanation / thoughts:</div>
    <div id="gemini-output">(no output yet)</div>
  </div>

<script>
  const canvas = document.getElementById('drawingCanvas');
  const ctx = canvas.getContext('2d');

  // init white background
  function resetCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
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

  const apiKeyInput = document.getElementById('api-key-input');
  const setApiKeyBtn = document.getElementById('set-api-key');

  function setBusy(isBusy, message) {
    askAiBtn.disabled = isBusy;
    clearBtn.disabled = isBusy;
    modeSelect.disabled = isBusy;
    if (saveSvgBtn) {
      saveSvgBtn.disabled = isBusy;
    }
    if (setApiKeyBtn) {
      setApiKeyBtn.disabled = isBusy;
    }
    statusEl.textContent = message || '';
  }

  setApiKeyBtn.addEventListener('click', async () => {
    const key = apiKeyInput.value.trim();
    if (!key) {
      statusEl.textContent = 'Please enter an API key.';
      return;
    }

    try {
      setBusy(true, 'Setting API key...');
      const res = await fetch('/api/set_api_key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || 'Server error');
      }

      const json = await res.json();
      statusEl.textContent = json.message || 'API key set.';
    } catch (err) {
      console.error(err);
      statusEl.textContent = 'Error setting API key. See console.';
    } finally {
      setBusy(false);
    }
  });


  askAiBtn.addEventListener('click', async () => {
    try {
      setBusy(true, 'Asking Gemini to add to your drawing...');
      const dataUrl = canvas.toDataURL('image/png');
      const res = await fetch('/api/ai_draw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl }),
      });
      if (!res.ok) {
        throw new Error('Server error');
      }
      const json = await res.json();
      const newImg = new Image();
      newImg.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(newImg, 0, 0, canvas.width, canvas.height);
        setBusy(false, 'AI updated the drawing.');
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
      setBusy(false, 'Error talking to AI. See console.');
    }
  });

  clearBtn.addEventListener('click', () => {
    resetCanvas();
    statusEl.textContent = 'Canvas cleared.';
  });

  saveSvgBtn.addEventListener('click', async () => {
    try {
      setBusy(true, 'Saving SVG on server as robot_path.svg ...');
      const dataUrl = canvas.toDataURL('image/png');
      const res = await fetch('/api/save_svg', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl }),
      });
      if (!res.ok) {
        throw new Error('Server error');
      }
      const json = await res.json();
      setBusy(false, json.message || 'Saved robot_path.svg on server.');
    } catch (err) {
      console.error(err);
      setBusy(false, 'Error saving SVG. See console.');
    }
  });

  modeSelect.addEventListener('change', async () => {
    const mode = modeSelect.value;  // "collaborative" or "adversarial"
    try {
      setBusy(true, `Switching mode to "${mode}" ...`);
      const res = await fetch('/api/set_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      if (!res.ok) {
        throw new Error('Server error');
      }
      const json = await res.json();
      setBusy(false, json.message || `Mode set to ${mode}.`);
    } catch (err) {
      console.error(err);
      setBusy(false, 'Error switching mode. See console.');
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


@app.post("/api/ai_draw")
async def api_ai_draw(payload: ImagePayload):
    """Take current canvas image, call Gemini co-painter, return updated canvas + text."""
    try:
        img = data_url_to_cv2_image(payload.image)

        # Support both 3-return and 4-return versions of get_gemini_drawing.
        try:
            old_drawing, new_drawing, combined_drawing, text_output = get_gemini_drawing(
                img.copy(),
                app.state.prompt,
                app.state.model,
                app.state.condition,
            )
        except ValueError:
            old_drawing, new_drawing, combined_drawing = get_gemini_drawing(
                img.copy(),
                app.state.prompt,
                app.state.model,
                app.state.condition,
            )
            text_output = ""

        # After first call, switch to subsequent prompt (shared for both modes for now)
        app.state.prompt = SUBSEQUENT_PROMPT

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

        os.makedirs("saved_images", exist_ok=True)
        app.state.turn_idx += 1
        postfix = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("saved_images", f"turn_{app.state.turn_idx:04d}_{postfix}.png")
        cv2.imwrite(save_path, combined_drawing)
        print(f"Saved AI turn image to {save_path}")

        data_url = cv2_image_to_data_url(combined_drawing)
        return JSONResponse({"image": data_url, "text": text_output})
    except Exception as e:
        print("Error in /api/ai_draw:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/set_api_key")
async def api_set_api_key(payload: ApiKeyPayload):
    """Set / override Gemini API key at runtime."""
    try:
        key = payload.api_key.strip()
        if key == "daehwa's":
          key = d_api_key
        if not key:
            return JSONResponse({"error": "API key is empty."}, status_code=400)

        # Re-init Gemini client with the new key
        init_gemini_api(key)
        model = get_model()

        # Store in app state
        app.state.api_key = key
        app.state.model = model

        # Optional: reset prompt & condition when key changes
        app.state.prompt = GEMINI_PROMPT
        # keep current mode's condition
        app.state.condition = CONDITIONS.get(app.state.mode, None)

        msg = "API key updated and Gemini model re-initialized."
        print(msg)
        return JSONResponse({"message": msg})
    except Exception as e:
        print("Error in /api/set_api_key:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


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



@app.post("/api/set_mode")
async def api_set_mode(payload: ModePayload):
    """Switch between 'collaborative' and 'adversarial' modes."""
    mode = payload.mode.lower()
    if mode not in ("collaborative", "adversarial"):
        return JSONResponse(
            {"error": f"Unknown mode '{payload.mode}'"},
            status_code=400,
        )

    try:
        app.state.mode = mode
        app.state.condition = CONDITIONS[mode]
        # Reset prompt when switching modes so Gemini knows fresh context.
        app.state.prompt = GEMINI_PROMPT
        msg = f"Mode set to '{mode}'. Prompt reset to initial GEMINI_PROMPT."
        print(msg)
        return JSONResponse({"message": msg})
    except KeyError:
        # If CONDITIONS doesn't have this key, surface a helpful error.
        err = f"Mode '{mode}' not found in config.CONDITIONS."
        print(err)
        return JSONResponse({"error": err}, status_code=500)
