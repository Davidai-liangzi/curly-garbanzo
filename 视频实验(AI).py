import cv2

video_path = "test video.mp4"
cap = cv2.VideoCapture(video_path)  # 重命名变量避免混淆

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# 读取并显示视频帧
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    # 图像处理 - 对每一帧进行处理
    # 中值滤波
    blurred = cv2.medianBlur(frame, 5)
    # 转换为灰度图
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 在原图上绘制轮廓
    result_frame = frame.copy()  # 创建副本用于绘制结果
    cv2.drawContours(result_frame, contours, -1, (0, 255, 0), 3)

    # 为每个轮廓绘制边界矩形
