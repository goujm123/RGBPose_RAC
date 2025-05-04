import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp


#  这是腾讯元宝生成的代码，错误较多，流程供参考

def gaussian_kernel(size, sigma):
    """生成二维高斯核"""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)


def draw_skeleton_heatmap(image, landmarks, sigma=5):
    """绘制带权热力图和骨架连线"""
    h, w, _ = image.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    # 定义BlazePose关键点连接关系(错的！ 该是 33个关键点)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 脊柱
        (1, 5), (5, 6), (6, 7),  # 左臂
        (1, 8), (8, 9), (9, 10),  # 右臂
        (1, 11), (11, 12), (12, 13),  # 左腿
        (1, 14), (14, 15), (15, 16)  # 右腿
    ]

    # 生成热力图
    for idx, (x, y, z, vis) in enumerate(landmarks):
        if vis > 0:
            # 创建高斯权重矩阵
            kernel = gaussian_kernel(15, sigma)
            x0, y0 = int(x * w), int(y * h)
            x1, y1 = min(x0 + kernel.shape[1] // 2, w), min(y0 + kernel.shape[0] // 2, h)
            x2, y2 = max(x0 - kernel.shape[1] // 2, 0), max(y0 - kernel.shape[0] // 2, 0)
            heatmap[y2:y1, x2:x1] += kernel * vis

    # 归一化热力图
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 绘制骨架连线
    overlay = image.copy()
    for conn in connections:
        pt1 = landmarks[conn[0]]
        pt2 = landmarks[conn[1]]

        if pt1[3] > 0 and pt2[3] > 0:
            # 插值计算中间点
            num_points = int(np.linalg.norm(np.array(pt1[:2]) - np.array(pt2[:2])) / 5)
            for t in np.linspace(0, 1, num_points):
                x = int((1 - t) * pt1[0] * w + t * pt2[0] * w)
                y = int((1 - t) * pt1[1] * h + t * pt2[1] * h)
                alpha = (1 - t) * pt1[3] + t * pt2[3]

                # 计算动态线宽（2倍标准差）
                std_dev = max(1, int(10 * alpha))  # 标准差与可见性相关
                kernel = gaussian_kernel(5, std_dev)

                # 绘制带权重的线段
                x0, y0 = x - kernel.shape[1] // 2, y - kernel.shape[0] // 2
                x1, y1 = x + kernel.shape[1] // 2, y + kernel.shape[0] // 2
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                overlay[y0:y1, x0:x1] += kernel * int(alpha * 255)

    # 合成最终图像
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    return result, heatmap_color


# 初始化BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5)

# 读取图像
image = cv2.imread('test.jpg')
h, w, _ = image.shape
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 处理图像
results = pose.process(image_rgb)
pose.close()

# 提取关键点
landmarks = []
if results.pose_landmarks:
    for landmark in results.pose_landmarks.landmark:
        landmarks.append((
            landmark.x,  # x坐标 (0-1)
            landmark.y,  # y坐标 (0-1)
            landmark.z,  # z坐标 (0-1)
            landmark.visibility  # 可见性 (0-1)
        ))

# 绘制结果
if landmarks:
    result_img, heatmap = draw_skeleton_heatmap(image, landmarks)

    # 显示结果
    plt.figure(figsize=(15, 5))

    # 原始图像
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 带权热力图
    plt.subplot(132)
    plt.imshow(heatmap)
    plt.title('Weighted Heatmap')
    plt.axis('off')

    # 最终合成图像
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('Final Result')
    plt.axis('off')

    plt.show()
else:
    print("No pose detected!")