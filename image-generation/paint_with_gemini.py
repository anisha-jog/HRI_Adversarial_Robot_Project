import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import io
import os
import sys
from config import API_KEY, CANVAS_SIZE, GEMINI_PROMPT, SUBSEQUENT_PROMPT, CONDITIONS

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tests'))
sys.path.insert(0, module_dir)

# Initialize Gemini API
genai.configure(api_key=API_KEY)

# --- Global Variables & Drawing Setup ---
drawing = False         # True if mouse is pressed
last_point = (0, 0)     # Last known mouse position
canvas = np.full(CANVAS_SIZE, 255, dtype=np.uint8) # White canvas
previous_canvas = np.full(CANVAS_SIZE, 255, dtype=np.uint8) # White canvas
robot_turn = np.full(CANVAS_SIZE, 255, dtype=np.uint8) # White canvas

def separate_colors(img):
    data = np.array(img)
    red_output_data = np.full_like(data, 255, dtype=np.uint8)
    black_output_data = np.full_like(data, 255, dtype=np.uint8)

    r, g, b = data[:,:,2].astype(int), data[:,:,1].astype(int), data[:,:,0].astype(int)

    RED_MIN_R_VALUE = 90      # Min R value to be considered "reddish"
    RED_DOMINANCE_DIFF = 30   # R must be this much greater than G and B
    
    red_mask = (r > RED_MIN_R_VALUE) & \
               (r > g + RED_DOMINANCE_DIFF) & \
               (r > b + RED_DOMINANCE_DIFF)
    
    BLACK_MAX_VALUE = 120
    
    initial_black_mask = (r < BLACK_MAX_VALUE) & \
                         (g < BLACK_MAX_VALUE) & \
                         (b < BLACK_MAX_VALUE)
    
    black_mask = initial_black_mask & ~red_mask

    red_output_data[red_mask] = data[red_mask]
    
    black_output_data[black_mask] = data[black_mask]

    red_img = Image.fromarray(red_output_data)
    black_img = Image.fromarray(black_output_data)

    return black_img, red_img

# --- 3. Mouse Callback Function ---
def draw_callback(event, x, y, flags, param):
    """Handles mouse events for drawing on the canvas."""
    global canvas, drawing, last_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw a black line from the last point to the current point
            cv2.line(canvas, last_point, (x, y), (0, 0, 0), 1.5)
            last_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def combine_images(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    _, img1 = cv2.threshold(img1, 200, 255, cv2.THRESH_BINARY)
    _, img2 = cv2.threshold(img2, 200, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(img1, img2)
    combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    return combined

def blackize(image_data):
    img_ = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    _, img_bw = cv2.threshold(img_, 200, 255, cv2.THRESH_BINARY)
    img_rgb = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB)
    return img_rgb

def init_gemini_api(API_KEY):
    """Initializes the Gemini API with the provided API key."""
    genai.configure(api_key=API_KEY)

def get_model():
    """Updates the Gemini model to ensure the latest version is used."""
    model = genai.GenerativeModel('gemini-2.5-flash-image')
    return model

def get_gemini_drawing(image_data, prompt, model, condition = CONDITIONS["custom"]):
    """Sends the current canvas to Gemini and returns the modified image."""
    NEW_PROMPT = prompt
    if condition is not None:
        NEW_PROMPT = NEW_PROMPT + f"""
            (1) {condition['visual']}
            (2) {condition['semantic']}
            """

    prompt = NEW_PROMPT.format(canvas_size=CANVAS_SIZE)
        
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    # convert image into only black and white. not even gray
    img_rgb = blackize(image_data)
    previous_canvas[:] = np.array(img_rgb)
    img_rgb = cv2.resize(img_rgb, (1024, 1024))
    pil_img = Image.fromarray(img_rgb)
    
    print("Sending to Gemini... this may take a moment.")
    try:
        # Send the prompt and the image to the model
        response = model.generate_content([prompt, pil_img])

        all_text = []
        # Iterate through all parts of the response
        for part in response.candidates[0].content.parts:
            # Check if the part has the 'text' attribute
            if part.text:
                all_text.append(part.text)

        # Join all the text pieces together
        final_text = "".join(all_text)
        print(final_text)

        # Check the response for an image part
        for part in response.parts:
            # The image is nested inside 'inline_data'
            if part.inline_data:
                # Check if the mime_type is an image
                if part.inline_data.mime_type.startswith("image/"):
                    
                    # Found an image! Get its binary data.
                    image_bytes = part.inline_data.data
                    
                    # Convert bytes to a PIL Image
                    pil_output_img = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert the PIL Image (RGB) back to an OpenCV image (BGR)
                    cv_output_img = cv2.cvtColor(np.array(pil_output_img), cv2.COLOR_RGB2BGR)
                    cv_output_img = cv2.resize(cv_output_img, (CANVAS_SIZE[1], CANVAS_SIZE[0]))

                    old, new = separate_colors(cv_output_img)
                    robot_turn[:] = np.array(new)
                    cv_output_img = (previous_canvas, robot_turn, combine_images(previous_canvas, robot_turn), final_text)
                    print("Got new drawing from Gemini!")
                    return cv_output_img
        print("------------------------------------------------")
        # --- MODIFIED SECTION END ---

        # If no image is found in the response
        print(f"Error: Gemini responded but did not return an image.")
        # Try to print text for debugging
        try:
            print(f"   Response text: {response.text}")
        except Exception:
            pass # No text part
        return None

    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return None