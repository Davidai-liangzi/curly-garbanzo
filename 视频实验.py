import cv2
import numpy as np

# 初始化视频捕获
cap = cv2.VideoCapture("video.mp4")

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()


def detect_football(frame):
    """
    使用颜色分割、最小框选和多边形识别来检测黑白相间的足球
    """
    result_frame = frame.copy()
    height, width = frame.shape[:2]

    # 1. 颜色分割 - 提取黑白区域
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 方法1: 使用灰度图进行阈值处理，提取黑白区域
    # 使用自适应阈值处理，适应光照变化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 方法2: 使用HSV颜色空间提取黑色和白色区域
    # 定义黑色的HSV范围
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # 定义白色的HSV范围
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])

    # 创建黑色和白色掩膜
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 合并黑白掩膜
    bw_mask = cv2.bitwise_or(black_mask, white_mask)

    # 合并两种方法的结果
    combined_mask = cv2.bitwise_or(bw_mask, binary)

    # 形态学操作，改善掩膜质量
    kernel_open = np.ones((3, 3), np.uint8)  # 开运算去除小噪点
    kernel_close = np.ones((15, 15), np.uint8)  # 闭运算连接断裂部分

    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

    # 2. 查找轮廓
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 处理每个轮廓
    football_contours = []

    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)

        # 面积过滤 - 根据实际足球大小调整
        if area < 300 or area > 30000:
            continue

        # 3.1 最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        # 计算最小外接矩形的长宽比
        width_rect = rect[1][0]
        height_rect = rect[1][1]
        aspect_ratio = max(width_rect, height_rect) / min(width_rect, height_rect)

        # 足球应该接近圆形，长宽比接近1
        if aspect_ratio > 1.8:  # 稍微放宽长宽比限制
            continue

        # 3.2 多边形逼近
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 计算圆形度
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # 3.3 综合判断
        # 条件1：圆形度 > 0.6 (稍微降低要求)
        # 条件2：多边形顶点数 > 5 (稍微降低要求)
        # 条件3：面积在合理范围内
        if circularity > 0.6 and len(approx) > 5:
            football_contours.append(contour)

            # 绘制最小外接矩形
            cv2.drawContours(result_frame, [box], 0, (0, 255, 0), 2)

            # 绘制多边形逼近
            cv2.drawContours(result_frame, [approx], -1, (255, 0, 0), 2)

            # 计算中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 绘制中心点
                cv2.circle(result_frame, (cx, cy), 5, (0, 0, 255), -1)

                # 添加标签
                cv2.putText(result_frame, f"Football", (cx - 30, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 4. 显示中间结果
    # 创建可视化掩膜
    mask_visual = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

    # 调整掩膜图像大小以匹配结果帧
    if mask_visual.shape != result_frame.shape:
        mask_visual = cv2.resize(mask_visual, (result_frame.shape[1], result_frame.shape[0]))

    # 在掩膜上绘制检测到的轮廓
    for contour in football_contours:
        cv2.drawContours(mask_visual, [contour], -1, (0, 255, 0), 2)

    # 水平拼接原始帧和掩膜
    combined_result = np.hstack([result_frame, mask_visual])

    return combined_result, len(football_contours)


# 主循环
while True:
    ret, frame = cap.read()

    if not ret:
        print("视频播放完毕或无法读取视频帧")
        break

    # 检测足球
    result_frame, count = detect_football(frame)

    # 显示结果
    cv2.imshow("Football Detection", result_frame)

    # 按'q'退出，按空格暂停/继续
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # 空格键暂停
        while True:
            key2 = cv2.waitKey(1) & 0xFF
            if key2 == ord(' '):  # 再次按空格继续
                break
            elif key2 == ord('q'):  # 暂停状态下也可以按q退出
                break
        if key2 == ord('q'):
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()