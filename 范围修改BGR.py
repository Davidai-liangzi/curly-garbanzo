import cv2
#读取图像
img = cv2.imread("123.jpg")
# 展示
cv2.imshow('test1',img)
#范围+编辑
for i in range(111,127):
    for j in range(111,127):
        img[i,j]=[255,255,255]
#展示
cv2.imshow('test2',img)
#等待输入
cv2.waitKey(-1)
#关闭窗口
cv2.destroyAllWindows()