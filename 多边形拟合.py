import cv2
import numpy as np

img = cv2.imread("kuang.PNG")
#创建蓝色掩膜（HSV）
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 150, 100])
upper_blue = np.array([140, 255, 255])
#二值纯膜生成
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
#形态学连接
kernel_large = np.ones((20, 20), np.uint8)
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_large)
#去除噪点（腐蚀和膨胀）
kernel_small = np.ones((3, 3), np.uint8)
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_small)
#找轮廓
contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#副本
result = img.copy()
#轮廓与面积处理
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area < 1000:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / min(w, h)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("first", blue_mask)
cv2.imshow("second", result)
cv2.waitKey(-1)
cv2.destroyAllWindows()