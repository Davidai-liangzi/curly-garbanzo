import cv2
import numpy as np
import os


def process_kuang_image(image_path="kuang.PNG"):
    """
    专门处理 kuang.PNG 图像的蓝色灯条六边形拟合
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件 {image_path}")
        print("请确保图像文件在当前工作目录中")
        print(f"当前工作目录: {os.getcwd()}")
        return None, []

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像文件 {image_path}")
        return None, []

    print(f"成功读取图像: {image_path}")
    print(f"图像尺寸: {image.shape}")

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义蓝色范围 - 根据kuang.PNG的实际蓝色调整
    # 通常蓝色在HSV中的H值在100-130之间
    blue_lower1 = np.array([100, 120, 70])  # 蓝色范围下限
    blue_upper1 = np.array([130, 255, 255])  # 蓝色范围上限

    # 创建蓝色掩码
    blue_mask1 = cv2.inRange(hsv, blue_lower1, blue_upper1)

    # 可选的第二个蓝色范围，用于捕捉不同亮度的蓝色
    blue_lower2 = np.array([90, 80, 50])
    blue_upper2 = np.array([110, 255, 200])
    blue_mask2 = cv2.inRange(hsv, blue_lower2, blue_upper2)

    # 合并两个蓝色掩码
    blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)

    # 形态学操作 - 根据kuang.PNG中灯条大小调整核大小
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)  # 填充小孔洞
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)  # 去除小噪声

    # 查找轮廓
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"找到 {len(contours)} 个蓝色区域")

    # 筛选六边形蓝色灯条
    blue_hexagons = []
    hexagon_approx = []

    for i, contour in enumerate(contours):
        # 面积筛选 - 根据kuang.PNG中灯条的实际大小调整
        area = cv2.contourArea(contour)
        if area < 100 or area > 10000:  # 调整面积阈值
            continue

        # 多边形拟合
        epsilon_ratio = 0.03  # 拟合精度
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 六边形顶点数量筛选 (5-7个顶点)
        vertex_count = len(approx)
        if 5 <= vertex_count <= 7:
            blue_hexagons.append(contour)
            hexagon_approx.append(approx)
            print(f"轮廓 {i}: 面积={area:.1f}, 顶点数={vertex_count} -> 符合六边形条件")
        else:
            print(f"轮廓 {i}: 面积={area:.1f}, 顶点数={vertex_count} -> 不符合六边形条件")

    # 创建结果图像
    result_image = image.copy()

    # 绘制所有蓝色轮廓（浅蓝色）
    cv2.drawContours(result_image, contours, -1, (255, 200, 100), 1)

    # 高亮显示六边形筛选结果
    for i, (contour, approx) in enumerate(zip(blue_hexagons, hexagon_approx)):
        # 绘制原始轮廓（红色）
        cv2.drawContours(result_image, [contour], 0, (0, 0, 255), 2)

        # 绘制拟合多边形（绿色）
        cv2.polylines(result_image, [approx], True, (0, 255, 0), 2)

        # 计算轮廓中心点
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 标记顶点数量和面积
            cv2.putText(result_image, f'Hexagon {i + 1}: {len(approx)} vertices',
                        (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 添加统计信息
    cv2.putText(result_image, f'Blue Regions: {len(contours)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_image, f'Hexagon Lights: {len(blue_hexagons)}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    print(f"\n最终结果: 找到 {len(blue_hexagons)} 个六边形蓝色灯条")

    # 显示中间结果（掩码）
    cv2.imshow("Blue Mask", blue_mask)

    return result_image, blue_hexagons


# 参数调优函数
def tune_parameters(image_path="kuang.PNG"):
    """
    交互式参数调优，帮助找到最佳参数
    """
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建调参窗口
    cv2.namedWindow("Parameter Tuning")

    # 创建滑动条
    cv2.createTrackbar("H Lower", "Parameter Tuning", 100, 180, lambda x: None)
    cv2.createTrackbar("H Upper", "Parameter Tuning", 130, 180, lambda x: None)
    cv2.createTrackbar("S Lower", "Parameter Tuning", 120, 255, lambda x: None)
    cv2.createTrackbar("S Upper", "Parameter Tuning", 255, 255, lambda x: None)
    cv2.createTrackbar("V Lower", "Parameter Tuning", 70, 255, lambda x: None)
    cv2.createTrackbar("V Upper", "Parameter Tuning", 255, 255, lambda x: None)
    cv2.createTrackbar("Kernel Size", "Parameter Tuning", 5, 15, lambda x: None)

    while True:
        # 获取滑动条值
        h_low = cv2.getTrackbarPos("H Lower", "Parameter Tuning")
        h_high = cv2.getTrackbarPos("H Upper", "Parameter Tuning")
        s_low = cv2.getTrackbarPos("S Lower", "Parameter Tuning")
        s_high = cv2.getTrackbarPos("S Upper", "Parameter Tuning")
        v_low = cv2.getTrackbarPos("V Lower", "Parameter Tuning")
        v_high = cv2.getTrackbarPos("V Upper", "Parameter Tuning")
        kernel_size = cv2.getTrackbarPos("Kernel Size", "Parameter Tuning")
        if kernel_size % 2 == 0:  # 确保核大小为奇数
            kernel_size += 1

        # 创建掩码
        lower_blue = np.array([h_low, s_low, v_low])
        upper_blue = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 形态学操作
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 显示结果
        result = cv2.bitwise_and(image, image, mask=mask)

        # 添加参数信息
        cv2.putText(result, f"H: [{h_low}, {h_high}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result, f"S: [{s_low}, {s_high}]", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result, f"V: [{v_low}, {v_high}]", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result, f"Kernel: {kernel_size}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Parameter Tuning", result)

        # 按ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# 主程序
if __name__ == "__main__":
    # 处理 kuang.PNG 图像
    result_img, detected_lights = process_kuang_image("kuang.PNG")

    if result_img is not None:
        # 显示结果
        cv2.imshow("Blue Hexagon Lights Detection - kuang.PNG", result_img)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果
        cv2.imwrite("kuang_result.jpg", result_img)
        print("结果已保存为 kuang_result.jpg")

        # 询问是否需要参数调优
        response = input("\n是否需要参数调优? (y/n): ")
        if response.lower() == 'y':
            print("启动参数调优工具...")
            print("使用滑动条调整参数，按ESC退出")
            tune_parameters("kuang.PNG")