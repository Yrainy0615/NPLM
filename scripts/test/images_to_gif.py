import cv2
import os

directory = "/home/yang/projects/parametric-leaf/sample_result/udf_test_shape"
output_file = "output.avi"

images = []
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".png"):  
        filepath = os.path.join(directory, filename)
        images.append(filepath)

# 获取一张图像的信息
height, width, layers = cv2.imread(images[0]).shape

# 定义视频的编码，创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter(output_file, fourcc, 2, (width, height))

for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()