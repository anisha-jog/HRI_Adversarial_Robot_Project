import cv2

# initialize the camera
cam = cv2.VideoCapture(1)   # 1 -> index of camera (0 for device native, 1 for USB webcam)
s, img = cam.read()
if s:    # frame captured without any errors
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img_bw, 90, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("threshold.jpg", thresh) # save image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    c = max(contours, key=cv2.contourArea) # max contour
    f = open('path.svg', 'w+')
    f.write('<svg width="'+str(img.shape[1])+'" height="'+str(img.shape[0])+'" xmlns="http://www.w3.org/2000/svg">')
    f.write('<path d="M')

    for i in range(len(c)):
        #print(c[i][0])
        x, y = c[i][0]
        f.write(str(x)+  ' ' + str(y)+' ')

    f.write('"/>')
    f.write('</svg>')
    f.close()

    # cv2.namedWindow("cam-test", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("cam-test", img_bw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("captured_image.jpg", img_bw) # save image