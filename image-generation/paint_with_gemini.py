import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import io
import os
from HRI_Adversarial_Robot_Project.jack_genai_test.config_files.gemini_api import GOOGLE_API_KEY 

# --- 1. Configuration ---
API_KEY = GOOGLE_API_KEY

genai.configure(api_key=API_KEY)

canvas_size = (300, 300, 3)
# --- 2. Global Variables & Drawing Setup ---
drawing = False         # True if mouse is pressed
last_point = (0, 0)     # Last known mouse position
canvas = np.full(canvas_size, 255, dtype=np.uint8) # White canvas
old_drawing = np.full(canvas_size, 255, dtype=np.uint8) # White canvas
new_drawing = np.full(canvas_size, 255, dtype=np.uint8) # White canvas
window_name = "AI Co-Painter (Draw with mouse, 'a' = AI, 'c' = Clear, 'q' = Quit)"

# This is the prompt you requested
similar = "draw the next few strokes that is align with"
opposite = "draw a subject that is diametrically opposite to"
# GEMINI_PROMPT = f"""
# You are an agent who helps the user to complete the drawing.
# Given this partial drawing from the user, infer the general context
# that the user is trying to draw, and {opposite} the current drawing's theme, context, vibe, etc.
# Return the image of your additional drawing.

# You must keep this format for the new drawing:
# 1. Keep the original drawing intact.
# 2. Overlay new drawing with red strokes on the original drawing.
# 3. The thickness of new strokes should uniform, which is same as the thickness of the user's strokes.
# 4. The return canvas resolution should be {canvas_size}.
# 5. The new drawing should be feasible to draw with lines and it is simple enough so that we can be drawn within 5 strokes.
# """

GEMINI_PROMPT = f"""
You are a creative painting agent collaborating with a user.
Given this partial drawing from the user, infer the general context
that the user is trying to draw, your role is not to imitate, but to challenge the human drawing in a constructive, surprising, and meaningful way.
You can oppose the human's intent in three ways:
(1) Visually (unrelated shape, color, texture),
(2) Semantically (unrelated concept or meaning),
(3) Compositionally (unrelated balance or layout).
Always explain your reasoning before generating the next stroke suggestion.

You must keep this format for the new drawing:
1. Keep the original drawing intact.
2. Overlay new drawing with red strokes on the original drawing.
3. The thickness of new strokes should uniform, which is same as the thickness of the user's strokes.
4. The return canvas resolution should be {canvas_size}.
5. The new drawing should be feasible to draw with lines and it is simple enough so that we can be drawn within 5 strokes.
"""

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
            cv2.line(canvas, last_point, (x, y), (0, 0, 0), 5)
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

# --- 4. Gemini API Function ---
def get_gemini_drawing(image_data, prompt):
    """Sends the current canvas to Gemini and returns the modified image."""
    
    print("Sending to Gemini... this may take a moment.")
    
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    # convert image into only black and white. not even gray
    img_rgb = blackize(image_data)
    old_drawing[:] = np.array(img_rgb)
    img_rgb = cv2.resize(img_rgb, (1024, 1024))
    pil_img = Image.fromarray(img_rgb)
    
    # --- Make sure you are using the image generation model! ---
    model = genai.GenerativeModel('gemini-2.5-flash-image')
    
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
                    cv_output_img = cv2.resize(cv_output_img, (canvas_size[1], canvas_size[0]))

                    old, new = separate_colors(cv_output_img)
                    new_drawing[:] = np.array(new)
                    cv_output_img = combine_images(old_drawing, new_drawing)


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

# --- 5. Main Application Loop ---
def main():
    global canvas, old_drawing, new_drawing
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_callback)
    
    print("Starting AI Co-Painter...")
    print("Controls:")
    print("  - Draw with the mouse.")
    print("  - Press 'a' to ask the AI to add to your drawing.")
    print("  - Press 'c' to clear the canvas.")
    print("  - Press 'q' or 'ESC' to quit.")

    while True:
        # Display the current canvas
        cv2.imshow(window_name, canvas)
        space = np.zeros((old_drawing.shape[0], 3, 3)).astype(type(old_drawing[0,0,0]))
        cv2.imshow("Old drawing / New drawing", np.hstack((old_drawing, space, new_drawing)))
        
        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # --- Handle Key Presses ---

        # 'q' or 'ESC' to quit
        if key == ord('q') or key == 27:
            print("Exiting...")
            break
        
        # 'a' to send to AI
        elif key == ord('a'):
            # Send a copy of the canvas to Gemini
            new_canvas = get_gemini_drawing(canvas.copy(), GEMINI_PROMPT)
            
            if new_canvas is not None:
                # Update our canvas with the AI's drawing
                canvas = new_canvas
        
        # 'c' to clear
        elif key == ord('c'):
            print("Canvas cleared.")
            canvas = np.full(canvas_size, 255, dtype=np.uint8)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()