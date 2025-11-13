import cv2

# initialize the camera
cam = cv2.VideoCapture(1)   # 1 -> index of camera (0 for device native, 1 for USB webcam)
s, img = cam.read()
if s:    # frame captured without any errors
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img_bw, 90, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("threshold.jpg", thresh) # save image

    # cv2.namedWindow("cam-test", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("cam-test", img_bw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("captured_image.jpg", img_bw) # save image