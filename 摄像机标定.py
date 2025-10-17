import cv2
import numpy as np

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 相机内参矩阵 [fx, 0, cx; 0, fy, cy; 0, 0, 1]
camera_matrix = np.array([
    [1000, 0, 640],  # fx=1000, 光心cx=640
    [0, 1000, 360],  # fy=1000, 光心cy=360
    [0, 0, 1]  # 齐次坐标
], dtype=np.float32)

#这里假设没有畸变
dist_coeffs = np.zeros((4, 1))

#标准A4纸尺寸为210mm × 297mm
a4_width = 0.210
a4_height = 0.297

# 定义矩形在3D空间中的角点坐标（以矩形中心为原点）
object_points = np.array([
    [-a4_width / 2, -a4_height / 2, 0],  # 左下角
    [a4_width / 2, -a4_height / 2, 0],  # 右下角
    [a4_width / 2, a4_height / 2, 0],  # 右上角
    [-a4_width / 2, a4_height / 2, 0]  # 左上角
], dtype=np.float32)

cv2.namedWindow("first")

canny_low = 50
canny_high = 150
blur_size = 5


while True:

    _, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)  # 高斯模糊，减少噪声

    edges = cv2.Canny(blurred, canny_low, canny_high)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = frame.copy()

    rectangle_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            if cv2.isContourConvex(approx):
                rectangle_contours.append(approx)

    if rectangle_contours:
        largest_rect = max(rectangle_contours, key=cv2.contourArea)

        cv2.drawContours(img, [largest_rect], -1, (0, 255, 0), 3)

        # 对四个角点进行排序，确保顺序一致：左下、右下、右上、左上
        # 首先按照y坐标排序（图像坐标系y轴向下）
        sorted_points = sorted(largest_rect[:, 0, :], key=lambda point: point[1])

        # 下半部分两个点（y坐标较大）按x坐标排序
        bottom_points = sorted(sorted_points[2:], key=lambda point: point[0])
        # 上半部分两个点（y坐标较小）按x坐标排序
        top_points = sorted(sorted_points[:2], key=lambda point: point[0])

        image_points = np.array([
            bottom_points[0],  # 左下
            bottom_points[1],  # 右下
            top_points[1],  # 右上
            top_points[0]  # 左上
        ], dtype=np.float32)

        try:

            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

            if success:
                # 定义坐标轴3D点：原点 + X/Y/Z轴端点
                # X轴沿着矩形宽度方向，Y轴沿着矩形高度方向，Z轴垂直于纸张平面
                axis_length_x = a4_width / 3  # X轴长度
                axis_length_y = a4_height / 3  # Y轴长度
                axis_length_z = 0.05  # Z轴长度（垂直于纸张平面）

                axis_points = np.array([
                    [0, 0, 0],  # 原点（矩形中心）
                    [axis_length_x, 0, 0],  # X轴正方向（红色）- 沿着矩形宽度
                    [0, axis_length_y, 0],  # Y轴正方向（绿色）- 沿着矩形高度
                    [0, 0, axis_length_z]  # Z轴正方向（蓝色）- 垂直于纸张
                ], dtype=np.float32)


                projected_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

                origin = tuple(projected_points[0].ravel().astype(int))  # 原点
                x_axis = tuple(projected_points[1].ravel().astype(int))  # X轴端点
                y_axis = tuple(projected_points[2].ravel().astype(int))  # Y轴端点
                z_axis = tuple(projected_points[3].ravel().astype(int))  # Z轴端点

                cv2.arrowedLine(img, origin, x_axis, (0, 0, 255), 3)  # X轴：红色（沿着矩形宽度）
                cv2.arrowedLine(img, origin, y_axis, (0, 255, 0), 3)  # Y轴：绿色（沿着矩形高度）
                cv2.arrowedLine(img, origin, z_axis, (255, 0, 0), 3)  # Z轴：蓝色（垂直于纸张）

                cv2.putText(img, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.circle(img, origin, 8, (0, 255, 255), -1)

        except Exception as e:

            print(f"Error: {e}")
            pass

    cv2.imshow("result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()