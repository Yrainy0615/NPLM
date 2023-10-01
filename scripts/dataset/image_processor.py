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
from DataManager import LeafImageManger
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, M, (cols, rows))
    return translated

def rotate_image(image, angle, rot_point):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(rot_point, angle.astype(np.float32), 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    return cv2.warpAffine(image, M, (nW, nH))
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

def get_label():
    data_structure = {}

    # 从文件中读取行
    with open('/home/yang/projects/parametric-leaf/dataset/LeafData/jpg_files_info.txt', 'r') as file:
        for line in file:
            line = line.strip()  # 去除行尾的换行符
            if not line:  # 跳过空行
                continue
            # 拆分路径以获取需要的信息
            parts = line.split('/')
            category = parts[-3]  # 例如：Alstonia
            health_status = parts[-2]  # 例如：healthy

            # 更新数据结构
            if category not in data_structure:
                data_structure[category] = {health_status: [line]}
            else:
                if health_status not in data_structure[category]:
                    data_structure[category][health_status] = [line]
                else:
                    data_structure[category][health_status].append(line)
    with open('shape_label.json', 'w') as f:
        json.dump(data_structure, f)
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

class ImageProcessor():
    """
    functions (need aligned 2d image dataset with mask)
        1. load 2d sdf from mask
        2. transpose 2d sdf to 3d voxel grid
        3. sampling points from voxel grid
        4. image alignment
    """
    def __init__(self, root_path):
        self.root_dir = root_path
        self.all_species = os.listdir(self.root_dir)
        self.manager = LeafImageManger(self.root_dir)
        self.voxel_mini = [-0.5, -0.5,0.5]
        self.voxel_max = [0.5,0.5,0.5]
        self.resolution = 256
    
    def mask_to_sdf(self, mask, save_path):
        SDF = SDF2D(mask)
        sdf_2d = SDF.mask2sdf()
        sdf_3d = sdf2d_3d(sdf_2d)
        filename = save_path + '_sdf.npy'
        # np.save(filename, sdf_3d)
        # print(f'sdf file saced: {filename}')
        return sdf_3d
        
    
    def extract_mesh(self, sdf, save_path):
        mesh = mesh_from_sdf(sdf, self.voxel_mini, self.voxel_max, self.resolution)
        filename = save_path + '.obj'
        mesh.export(filename)
        print(f"{filename} is saved")
    
    def sampling_voxel(self, voxel):
        pass
    
    def image_alignment(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        leaf_points = np.column_stack(np.where(mask > 0))
        img_path = path.replace('_mask', '')
        image = cv2.imread(img_path)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pca= PCA(n_components=1)
        pca.fit(leaf_points)
        
        principal_axis = pca.components_[0]
        projection = np.dot(leaf_points - pca.mean_, pca.components_[0])
        min_point = leaf_points[np.argmin(projection)]
        max_point = leaf_points[np.argmax(projection)]
        # 计算旋转角度并进行旋转
       
        angle = np.arctan2(min_point[1]-max_point[1],min_point[0]-max_point[0]) * (180.0 / np.pi)
        print(f'axis info: {min_point[1]-max_point[1],min_point[0]-max_point[0]}')
        if min_point[1]-max_point[1]<=0:
            rotated_mask = rotate_image(mask,180-angle, pca.mean_)  # Negative as we need to rotate the other way
            rotated_image = rotate_image(image, 180-angle, pca.mean_)        
        else:
            rotated_mask = rotate_image(mask, -angle, pca.mean_) # np.degrees(angle)
            rotated_image = rotate_image(image, -angle, pca.mean_)    
    
        # 计算旋转后叶子的中心并进行平移
        h, w = mask.shape
        center_x, center_y = pca.mean_
        tx = w//2 -center_x
        ty = h//2 -center_y
        translated_mask =translate_image(rotated_mask, tx, ty)
        translated_image = translate_image(rotated_image, tx, ty)
        translated_image = cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB) 
        dirname, filename = os.path.split(path)
        filebase, ext = os.path.splitext(filename)
        new_name = filebase + '_aligned' + ext
        new_path = os.path.join(dirname, new_name)
        new_image = new_path.replace('_mask', '')
                # # 显示图像
        # plt.figure(figsize=(10,5))
        # plt.subplot(1,2,1)
        # plt.imshow(image, cmap='Greens')
        # plt.plot([min_point[1], max_point[1]], [min_point[0], max_point[0]], 'r')  
        # plt.subplot(1,2,2)
        # plt.imshow(translated_image, cmap='Greens')
        # plt.title('Aligned')
        # plt.title('Rotated')
        # plt.show()
        #cv2.imwrite(new_path, translated_mask)
        cv2.imwrite(new_image, translated_image)
        print(f"save image: {new_path}")

    
    def crop_image(self, path):
        assert isinstance(path, str), f"Expected a string for filename, but got {type(path)}: {path}"
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_path = path.replace('_mask', '')
        image = cv2.imread(img_path)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 获取 mask 的非零像素的坐标
        y, x = np.where(mask > 0)

        # 获取 bounding box
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        # 裁剪图像
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        cropped_image = image[y_min:y_max+1, x_min:x_max+1]

        # 计算新的大小，保持长宽比
        h, w = cropped_mask.shape
        aspect_ratio = w / h
        if h > w:
            new_h = 512
            new_w = int(new_h * aspect_ratio)
        else:
            new_w = 512
            new_h = int(new_w / aspect_ratio)

        # 调整大小
        resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # 在需要的地方填充0，以获得256x256的图像
        final_mask = np.zeros((512, 512), dtype=resized_mask.dtype)
        final_image = np.zeros((512,512,3), dtype=resized_image.dtype)
        y_offset = (512 - new_h) // 2
        x_offset = (512 - new_w) // 2
        final_mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_mask
        final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
        final_image[final_mask==0] =0
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(final_mask)
        # plt.subplot(1,2,2)
        # plt.imshow(final_image)
        # plt.show()
        cv2.imwrite(img_path,final_image)
        print(f"{img_path} is cropped.")
        cv2.imwrite(path, final_mask)

class Mask_preprocess():
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
    test_image = '/home/yang/projects/parametric-leaf/dataset/LeafData/Arjun/healthy/Arjun_healthy_0001_mask_aligned.JPG'
    # save_file_info(root_path)
    # get_label()
    manager = LeafImageManger(root_path)
    processor = ImageProcessor(root_path)
    # sdf_3d = processor.mask_to_sdf(test_image, base_path)
    #processor.extract_mesh(sdf_3d, base_path)
    pass
    all_healthy, all_diseased = manager.get_mask_train()
    for healthy in all_healthy:
        base_path, _ = os.path.splitext(healthy)
        base_path = base_path.rsplit('_mask', 1)[0]
        filename = base_path + '.obj'
        if not os.path.exists(filename):
            sdf_3d =processor.mask_to_sdf(healthy, base_path)
            processor.extract_mesh(sdf_3d, base_path)

            #processor.image_alignment(healthy)
    for diseased in all_diseased:
        base_path, _ = os.path.splitext(diseased)
        # base_path = base_path.rsplit('_aligned', 1)[0]
        filename = base_path + '.obj'
        if not os.path.exists(filename):
            #processor.image_alignment(diseased)
            sdf_3d = processor.mask_to_sdf(diseased, base_path)
            processor.extract_mesh(sdf=sdf_3d, save_path=base_path)
        




