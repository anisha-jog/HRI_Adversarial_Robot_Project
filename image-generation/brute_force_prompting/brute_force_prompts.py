import cv2, numpy as np
import sys, os
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, module_dir)
from config import CANVAS_SIZE, MODELS, VISUAL, SEMANTIC, COMPOSITIONAL
from paint_with_gemini import (
    canvas,
    draw_callback, get_gemini_drawing, image_to_svg
)

def run_application():
    new_drawing = np.full(CANVAS_SIZE, 255, dtype=np.uint8) # White canvas
    image_path = "input/happy_face_doodle.jpg"
    test_drawing = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # add test drawing to canvas
    # resize test_drawing to fit in canvas
    test_drawing = cv2.resize(test_drawing, (CANVAS_SIZE[1], CANVAS_SIZE[0]))    
    canvas = cv2.cvtColor(test_drawing, cv2.COLOR_GRAY2BGR)
    for vis in VISUAL:
        for sem in SEMANTIC:
            for comp in COMPOSITIONAL:
                new_drawing[:] = 255
                model = {
                    "visual": VISUAL[vis],
                    "semantic": SEMANTIC[sem],
                    "compositional": COMPOSITIONAL[comp]
                      }
                model_name = f"visually-{vis}_semantically-{sem}_compositionally-{comp}"
                
                #add image to old drawing in canvas, image is located in testdata/happy_face_doodle.jpg
                print(f"Running model: {model_name}")
                cv2.namedWindow("AI Co-Painter")
                (_, _, combined_drawing) = get_gemini_drawing(canvas.copy(), model)
                # save combined drawing as png 
                cv2.imwrite(f"output/{model_name}.png", combined_drawing)

if __name__ == "__main__":
    run_application()