import cv2, numpy as np
from config import CANVAS_SIZE, GEMINI_PROMPT, SUBSEQUENT_PROMPT, CONDITIONS
from PIL import Image
import image_to_svg as im
from paint_with_gemini import (
    canvas,
    get_gemini_drawing, init_gemini_api, get_model, API_KEY
)

# Study prompt and condition
prompt = GEMINI_PROMPT
condition = CONDITIONS["adversarial"]

# ArUco marker IDs for the corners
# Please ensure these IDs match the markers you are using
aruco_corner_order = {"top-left" : 20,
                      "top-right": 40,
                      "bottom-right": 10,
                      "bottom-left": 30}

def init_camera(camera_index=0, resolution=(1920, 1080)):
    ''' Initializes the camera with the given index and resolution. '''
    # 0 for device native, 1 for USB webcam
    cam = cv2.VideoCapture(camera_index)   
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    return cam

def transform_image(img, corners, width=500, height=500):
    ''' Transforms the image based on the provided corners using perspective transformation. '''
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sorted_corners = np.array(corners, dtype="float32")
    width, height = int(width*1.294), height
    destination_corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    # Get the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(
        sorted_corners, destination_corners)

    # Apply the perspective transformation
    transformed_image = cv2.warpPerspective(img_bw, matrix, (width, height))
    return transformed_image

def update_camera_feed(frame):
    ''' Updates and displays the camera feed with detected ArUco markers. '''
    # Display the camera feed as smaller so you don't go insane.
    resized_frame = cv2.resize(frame, (640, 480))
    # Show detected ArUco markers on the camera feed:
    img_bw = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)  
    corners, ids, rejected = detector.detectMarkers(img_bw)

    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(resized_frame, corners, ids)
    
    # Camera feed preview:
    cv2.imshow("Camera Feed, Press 'N' to add drawing", resized_frame)

def capture_camera_frame(cam):
    ''' Captures a frame from the camera and returns the transformed image based on ArUco markers. '''
    s, img = cam.read()
    if s:    # frame captured without any errors
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # get aruco markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(img_bw)
        # print("Detected markers:", ids)
        if ids is not None:
            top_left = [1, 1]
            top_right = [img_bw.shape[1], 1]
            bottom_right = [img_bw.shape[1], img_bw.shape[0]]
            bottom_left = [1, img_bw.shape[0]]
            for i in range(ids.shape[0]):
                # print(ids[i], corners[i])
                if ids[i][0] == aruco_corner_order["bottom-right"]:
                    bottom_right = corners[i][0][0]
                if ids[i][0] == aruco_corner_order["top-right"]:
                    top_right = corners[i][0][0]
                if ids[i][0] == aruco_corner_order["bottom-left"]:
                    bottom_left = corners[i][0][0]
                if ids[i][0] == aruco_corner_order["top-left"]:
                    top_left = corners[i][0][0]
            transformed_image = transform_image(img, [top_left, top_right, bottom_right, bottom_left])
            return  (ids, transformed_image)
    return None

def init_canvas():
    old_drawing = np.full(CANVAS_SIZE, 255, dtype=np.uint8) # White canvas
    new_drawing = np.full(CANVAS_SIZE, 255, dtype=np.uint8) # White canvas
    print("Starting AI Co-Painter...")
    print("Controls:")
    print("  - Press 'n' for next robot turn (i.e. capture camera image, send to Gemini, output new drawing/trajectory).")
    print("  - Press 'c' to clear the canvas.")
    print("  - Press 'q' or 'ESC' to quit.")
    print("  - Press 's' to save the current drawing as PNG.")
    return old_drawing, new_drawing

    
def run_application():
    cam = init_camera(1) # 1 -> index of camera (0 for device native, 1 for USB webcam)
    old_drawing, new_drawing = init_canvas()
    init_gemini_api(API_KEY)
    model = get_model()
    while True:
        # Show real-time camera feed in a separate window
        ret, frame = cam.read()
        if ret:
            update_camera_feed(frame)
        space = np.zeros((old_drawing.shape[0], 3, 3)).astype(type(old_drawing[0,0,0]))
        cv2.imshow("Old drawing / New drawing", np.hstack((old_drawing, space, new_drawing)))
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27 or key == ord('Q'):
            print("Exiting...")
            break
        elif key == ord('n') or key == ord('N'):
            # take a picture from the camera, add camera image to canvas
            captured_frame = capture_camera_frame(cam)
            if captured_frame is not None  and captured_frame[0].shape[0] == 4:
                # Ensure the captured drawing matches the canvas size
                captured_drawing = cv2.resize(captured_frame[1], (CANVAS_SIZE[1], CANVAS_SIZE[0]))
                captured_drawing = cv2.cvtColor(captured_drawing, cv2.COLOR_GRAY2BGR)
                canvas[:] = captured_drawing
                model = get_model()
                (old_drawing, new_drawing, combined_drawing) = get_gemini_drawing(canvas, prompt, model, condition)
                
                # Generate trajectory from new drawing 
                lines = im.image_to_lines(new_drawing, segments=10)
                
                # Target width and height should correspond to 8.5 x 11 inches in meters:
                traj = im.convert_pixels_to_meters(lines, new_drawing.shape[0], new_drawing.shape[1])

                ###########################
                ### TODO(@Jack): Add ROS2 logic here!
                ###########################
            else:
                print("Could not detect all 4 markers. Please adjust the camera and try again.")
        elif key == ord('c') or key == ord('C'):
            print("Canvas cleared.")
            canvas[:] = 255
            old_drawing[:] = 255
            new_drawing[:] = 255
            combined_drawing[:] = 255
        elif key == ord('s') or key == ord('S'):
            cv2.imwrite("canvas.png", canvas)
            cv2.imwrite("old_drawing.png", old_drawing)
            cv2.imwrite("new_drawing.png", new_drawing)
            cv2.imwrite("combined_drawing.png", combined_drawing)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_application()