import cv2
import numpy as np


def detect_and_box_hexagons(image, min_area=500, max_corners=8):
    """
    检测并框选抽象六边形

    参数:
        image: 输入图像
        min_area: 最小轮廓面积
        max_corners: 最大顶点数（用于筛选近似六边形）
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hexagons = []

    for contour in contours:
        # 面积筛选
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # 多边形近似
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 筛选顶点数接近6的多边形（5-7个顶点）
        if 5 <= len(approx) <= max_corners:
            # 计算轮廓的几何特征
            hull = cv2.convexHull(approx)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            # 筛选实心度较高的多边形
            if solidity > 0.8:
                hexagons.append({
                    'contour': approx,
                    'bounding_rect': cv2.boundingRect(approx)
                })

    return hexagons


# 使用示例
image = cv2.imread('kuang.PNG')
result = image.copy()

hexagons = detect_and_box_hexagons(image)

for i, hexagon in enumerate(hexagons):
    # 绘制六边形轮廓
    cv2.drawContours(result, [hexagon['contour']], -1, (0, 255, 0), 2)

    # 绘制边界框
    x, y, w, h = hexagon['bounding_rect']
    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 添加标签
    cv2.putText(result, f'Hex {i}', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.imshow('Hexagon Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()