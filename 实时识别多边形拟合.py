import cv2
import numpy as np

video_capture = cv2.VideoCapture("test.mp4")
while video_capture.isOpened():
    ret, frame = video_capture.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 20, 50])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel=np.ones((10, 10), np.uint8)
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour)>10:
         x,y,w,h = cv2.boundingRect(contour)
         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
         epsilon=0.03*cv2.arcLength(contour,True)
         approx = cv2.approxPolyDP(contour,epsilon,True)
         cv2.polylines(frame,[approx],True,(0,0,255),5)


    cv2.imshow("test",mask)
    cv2.imshow("result",frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
