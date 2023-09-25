import torch
import numpy as np
import cv2
import os
from scipy import ndimage 
from skimage import morphology
import pathlib
import mcubes
import trimesh
from img_to_3dsdf import SDF2D, sdf2d_3d, mesh_from_sdf
import json
import random

def count_jpg_files(root_dir):
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.jpg'):  # 这里加了一个 lower() 来确保文件扩展名是不区分大小写的。
                count += 1
    return count

def save_file_info(root_dir):
    jpg_files_info = []

    # 遍历根目录
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.JPG') and 'mask' in filename:
                full_path = os.path.join(foldername, filename)
                jpg_files_info.append(full_path)
    jpg_files_info.sort()
    # 将信息写入到txt文件中
    with open('jpg_files_info.txt', 'w') as file:
        for info in jpg_files_info:
            file.write(f"{info}\n\n")

class ImageProcessor():
    """
    functions (need aligned 2d image dataset with mask)
        1. load 2d sdf from mask
        2. transpose 2d sdf to 3d voxel grid
        3. sampling points from voxel grid
    """
    def __init__(self, root_path):
        self.image_dirs = root_path
        self.all_species = os.listdir(self.image_dirs)
        pass
    
    def mask_to_sdf(self, mask):
        SDF = SDF2D(mask)
        sdf_2d = SDF.mask2sdf()
        sdf_3d = sdf2d_3d(sdf_2d)
        
    
    def sdf_to_voxel(self, sdf):
        pass
    
    def sampling_voxel(self, voxel):
        pass
    
    
class Image_preprocess():
    def __init__(self) -> None:
        pass
    """ 
    for preprocess raw mask
    1. check bg and fg
    """
    def invert_and_save_if_needed(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Unable to read {image_path}")
            return
        
        white_area = np.sum(img == 255)
        black_area = np.sum(img == 0)
        
        if white_area > black_area:
            inverted_img = cv2.bitwise_not(img)  # invert black and white
            cv2.imwrite(image_path, inverted_img)  # overwrite the original image
            print(f"Inverted and saved {image_path}")
            
    def process_images_in_directory(self, directory):
        for foldername, subfolders, filenames in os.walk(directory):
            for filename in filenames:
                if 'mask' in filename and filename.endswith('.JPG'):
                    file_path = os.path.join(foldername, filename)
                    self.remove_noise(file_path)  

    def remove_noise(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kernel_size = 50
        kernel = np.ones((kernel_size, kernel_size),np.int8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cv2.imwrite('test.png', closing)  # overwrite the original image
        # print(f"Inverted and saved {image_path}")        
        

    def resize_crop_mask(self,file):
        bin_img = cv2.imread(file)
        bin_img = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到面积最大的连通区域
        max_contour = max(contours, key=cv2.contourArea)

        # 计算最小外接矩形
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算长和宽
        width = int(rect[1][0])
        height = int(rect[1][1])

        # 创建一个新的黑色图像
        output_size = 256
        output_img = np.zeros((output_size, output_size), dtype=np.uint8)

        # 保持长宽比，计算新的长和宽
        if width > height:
            new_width = output_size
            new_height = int(output_size * (height / width))
        else:
            new_height = output_size
            new_width = int(output_size * (width / height))

        # 调整最大连通区域的大小
        resized_contour = cv2.resize(bin_img, (new_width, new_height))

        # 将调整大小后的连通区域放在黑色图像中
        x_offset = (output_size - new_width) // 2
        y_offset = (output_size - new_height) // 2
        output_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_contour
        cv2.imwrite('test.png',output_img )

        
                    
if __name__ == "__main__":
    root_path = '/home/yang/projects/parametric-leaf/dataset/LeafData'
    test_image = '/home/yang/projects/parametric-leaf/dataset/LeafData/Pomegranate/healthy/Pomegranate_healthy_0024_mask.JPG'
    preprocessor  = Image_preprocess()
    # preprocessor.process_images_in_directory(root_path)
    # print(count_jpg_files(root_path))
    # save_file_info(root_path)
    #preprocessor.remove_noise(test_image)
  #   preprocessor.resize_crop_mask(test_image)
  # 初始化字典
    # data_structure = {}

    # 从文件中读取行
    # with open('jpg_files_info.txt', 'r') as file:
    #     for line in file:
    #         line = line.strip()  # 去除行尾的换行符
    #         if not line:  # 跳过空行
    #             continue
    #         # 拆分路径以获取需要的信息
    #         parts = line.split('/')
    #         category = parts[-3]  # 例如：Alstonia
    #         health_status = parts[-2]  # 例如：healthy

    #         # 更新数据结构
    #         if category not in data_structure:
    #             data_structure[category] = {health_status: [line]}
    #         else:
    #             if health_status not in data_structure[category]:
    #                 data_structure[category][health_status] = [line]
    #             else:
    #                 data_structure[category][health_status].append(line)
    # with open('shape_label.json', 'w') as f:
    #     json.dump(data_structure, f)
    with open('shape_label.json', 'r') as f:
        data_structure = json.load(f)

    train_structure = {}
    test_structure = {}

    # 设置随机种子以确保可重复性
    random.seed(42)

    # 遍历数据结构并分配数据点到训练集和测试集
    for category, health_statuses in data_structure.items():
        train_structure[category] = {}
        test_structure[category] = {}
        
        for health_status, file_paths in health_statuses.items():
            # 随机打乱路径
            random.shuffle(file_paths)
            
            # 计算测试集的大小
            test_size = len(file_paths) // 10  # 10%的数据作为测试集
            
            # 分配到训练集和测试集
            test_structure[category][health_status] = file_paths[:test_size]
            train_structure[category][health_status] = file_paths[test_size:]

    # 将结果保存为JSON文件
    with open('train_shape.json', 'w') as f:
        json.dump(train_structure, f)
        
    with open('test_shape.json', 'w') as f:
        json.dump(test_structure, f)     
    pass
