import cv2
#读取图像
image= cv2.imread("E:/test of lab/1.jpg")
#水平像素，垂直像素，通道数
print ("shape:",image.shape)
#类型
print("type:",image.dtype)
#大小
print("size:",image.size)