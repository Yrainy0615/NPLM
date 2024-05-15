
import cv2
import numpy as np
import pyvista as pv

# 读取mask
mask = cv2.imread('zebra/mask_vit_15.png')

# 找到所有非零像素
y_indices, x_indices,_ = np.where(mask != 0)

# 随机选择1000个点
random_index = np.random.randint(0, len(x_indices), 2000)
x_indices = x_indices[random_index]
y_indices = y_indices[random_index]
# 创建点云的坐标
points = np.zeros((len(x_indices), 3))
points[:, 0] = x_indices
points[:, 1] = y_indices
pv.start_xvfb()
# 创建黑色点云
colors_black = np.zeros((len(x_indices), 3))
cloud_black = pv.PolyData(points)
cloud_black['colors'] = colors_black

# 创建RGB点云
colors_rgb = mask[y_indices, x_indices] / 255.0
cloud_rgb = pv.PolyData(points)
cloud_rgb['colors'] = colors_rgb

# 渲染并保存图像
plotter = pv.Plotter(off_screen=True)
plotter.camera.position = (20,5,10)
plotter.camera.focal_point = (0, 10, 0)
plotter.camera.view_up = (0, 55, 2)
plotter.add_mesh(cloud_black, scalars='colors', rgb=False)
plotter.show(screenshot='black.png')

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(cloud_rgb, scalars='colors', rgb=False)
plotter.show(screenshot='rgb.png')