import cv2
img=cv2.imread('123.jpg')    # 读取
px=img[199,100]    #坐标定位
print("The value of BGR is:",px)
px=[255,255,255]    #坐标修改
print("the value of BGR is:",px)