import cv2
img_BGR=cv2.imread('bed_pic.png')
#HSV函数
img_HSV=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)
img_h,img_s,img_v=cv2.split(img_HSV)
#HSV范围筛选
mask_h=cv2.inRange(img_h,150,230)
mask_s= cv2.inRange(img_s,50,255)
mask_v=cv2.inRange(img_v,50,255)
#掩膜合并
mask_h_and_s=cv2.bitwise_and(mask_h,mask_s)
mask=cv2.bitwise_and(mask_h_and_s,mask_v)
img_output=cv2.bitwise_and(img_BGR,img_BGR,mask=mask)
cv2.imshow("img",img_output)
cv2.waitKey(-1)
cv2.destroyAllWindows()
