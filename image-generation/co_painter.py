import cv2, numpy as np
from config import CANVAS_SIZE, ADVERSARIAL_PROMPT, CONTROL_PROMPT
from image_to_svg import image_to_svg
from paint_with_gemini import (
    canvas,
    draw_callback, get_gemini_drawing, init_gemini_api, get_model, API_KEY
)

def run_application():
    window_name = "AI Co-Painter"
    old_drawing = np.full(CANVAS_SIZE, 255, dtype=np.uint8) # White canvas
    new_drawing = np.full(CANVAS_SIZE, 255, dtype=np.uint8) # White canvas
    """Main application loop for the AI Co-Painter."""
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_callback)
    
    print("Starting AI Co-Painter...")
    print("Controls:")
    print("  - Draw with the mouse.")
    print("  - Press 'a' to ask the AI to add to your drawing.")
    print("  - Press 'c' to clear the canvas.")
    print("  - Press 'q' or 'ESC' to quit.")
    print("  - Press 's' to save the current drawing as PNG and SVG.")
    init_gemini_api(API_KEY)
    model = get_model()
    prompt = ADVERSARIAL_PROMPT # CONTROL_PROMPT

    while True:
        cv2.imshow(window_name, canvas)
        space = np.zeros((old_drawing.shape[0], 3, 3)).astype(type(old_drawing[0,0,0]))
        cv2.imshow("Old drawing / New drawing", np.hstack((old_drawing, space, new_drawing)))
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            print("Exiting...")
            break
        elif key == ord('a'):
            model = get_model()
            (old_drawing, new_drawing, combined_drawing, text) = get_gemini_drawing(canvas.copy(), prompt, model, None)
            if combined_drawing is not None:
                canvas[:] = combined_drawing
        elif key == ord('c'):
            print("Canvas cleared.")
            canvas[:] = 255
            old_drawing[:] = 255
            new_drawing[:] = 255
        elif key == ord('s'):
            print("Saving new strokes to 'robot_path.svg'")
            image_to_svg(new_drawing, filename="robot_path.svg")
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_application()