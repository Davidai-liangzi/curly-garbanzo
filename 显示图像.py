import cv2
#读取图像
image = cv2.imread('123.jpg')
#打开图像
cv2.imshow("flower",image)
#等待键盘输入
cv2.waitKey(-1)
#关闭窗口
cv2.destroyAllWindows()