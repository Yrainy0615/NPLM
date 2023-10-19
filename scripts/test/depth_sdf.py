import numpy as np
import matplotlib.pyplot as plt

# 读取 .npy 文件
data = np.load("dataset/depth.npy")


depth_data = data[:,:]

# 可视化深度数据
plt.imshow(depth_data, cmap='gray')  # 使用灰度图
plt.colorbar()
plt.title("Depth Visualization")
plt.show()