import cv2, numpy as np
from config import CANVAS_SIZE, MODELS
from paint_with_gemini import (
    canvas,
    draw_callback, get_gemini_drawing, image_to_svg
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

    while True:
        cv2.imshow(window_name, canvas)
        space = np.zeros((old_drawing.shape[0], 3, 3)).astype(type(old_drawing[0,0,0]))
        cv2.imshow("Old drawing / New drawing", np.hstack((old_drawing, space, new_drawing)))
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            print("Exiting...")
            break
        elif key == ord('a'):
            (old_drawing, new_drawing, combined_drawing) = get_gemini_drawing(canvas.copy(), MODELS["custom"])
            if combined_drawing is not None:
                canvas[:] = combined_drawing
        elif key == ord('c'):
            print("Canvas cleared.")
            canvas[:] = 255
            old_drawing[:] = 255
            new_drawing[:] = 255
        elif key == ord('s'):
            print("Saving new strokes to 'robot_path.svg'")
            #lines = image_to_svg.image_to_svg(new_drawing, filename="robot_path.svg")
            print("Saving full canvas to 'full_canvas.png'")
            cv2.imwrite("full_canvas.png", canvas)
            # print("Lines:", lines) # This is x,y robot trajectory points
            # print("Saving full_canvas to 'full.svg'")
            # image_to_svg.image_to_svg(canvas, filename="full.svg")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_application()