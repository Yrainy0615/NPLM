import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import taichi as ti
import pathlib
import mcubes
import trimesh

ti.init(arch=ti.cpu)

MAX_DIST = 2147483647
null = ti.Vector([-1, -1, MAX_DIST])
vec3 = lambda scalar: ti.Vector([scalar, scalar, scalar])
eps = 1e-5


@ti.data_oriented
class SDF2D:
    def __init__(self, filename):
        self.filename = filename
        self.num = 0  # index of bit_pic

        self.im = cv2.imread(filename)
        self.width, self.height = self.im.shape[1], self.im.shape[0]
        self.pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
        self.bit_pic_white = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
        self.bit_pic_black = ti.Vector.field(3, dtype=ti.i32, shape=(2, self.width, self.height))
        self.output_pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
        self.output_linear = ti.Vector.field(3, dtype=ti.f32, shape=(self.width, self.height))
        self.max_reduction = ti.field(dtype=ti.i32, shape=self.width * self.height)

    def reset(self, filename):
        self.filename = filename
        self.num = 0  # index of bit_pic

        self.im = cv2.imread(filename)
        self.width, self.height = self.im.shape[1], self.im.shape[0]

    def output_filename(self, ins):
        path = pathlib.Path(self.filename)
        out_dir = path.parent / 'output'
        if not (out_dir.exists() and out_dir.is_dir()):
            out_dir.mkdir()
        return str(out_dir / (path.stem + ins + path.suffix))

    @ti.kernel
    def pre_process(self, bit_pic: ti.template(), keep_white: ti.i32):  # keep_white, 1 == True, -1 == False
        for i, j in self.pic:
            if (self.pic[i, j][0] - 127) * keep_white > 0:
                bit_pic[0, i, j] = ti.Vector([i, j, 0])
                bit_pic[1, i, j] = ti.Vector([i, j, 0])
            else:
                bit_pic[0, i, j] = null
                bit_pic[1, i, j] = null

    @ti.func
    def cal_dist_sqr(self, p1_x, p1_y, p2_x, p2_y):
        return (p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2

    @ti.kernel
    def jump_flooding(self, bit_pic: ti.template(), stride: ti.i32, n: ti.i32):
        # print('n =', n, '\n')
        for i, j in ti.ndrange(self.width, self.height):
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
                i_off = i + stride * di
                j_off = j + stride * dj
                if 0 <= i_off < self.width and 0 <= j_off < self.height:
                    dist_sqr = self.cal_dist_sqr(i, j, bit_pic[n, i_off, j_off][0],
                                                 bit_pic[n, i_off, j_off][1])
                    # print(i, ', ', j, ': ', 'dist_sqr: ', dist_sqr,', ', i_off, j_off)
                    if not bit_pic[n, i_off, j_off][0] < 0 and dist_sqr < bit_pic[1 - n, i, j][2]:
                        bit_pic[1 - n, i, j][0] = bit_pic[n, i_off, j_off][0]
                        bit_pic[1 - n, i, j][1] = bit_pic[n, i_off, j_off][1]
                        bit_pic[1 - n, i, j][2] = dist_sqr
                        # print(i, ', ', j, ': ', 'dist_sqr: ', dist_sqr, ', ', i_off, j_off)

    @ti.kernel
    def copy(self, bit_pic: ti.template()):
        for i, j in ti.ndrange(self.width, self.height):
            self.max_reduction[i * self.width + j] = bit_pic[self.num, i, j][2]

    @ti.kernel
    def max_reduction_kernel(self, r_stride: ti.i32):
        for i in range(r_stride):
            self.max_reduction[i] = max(self.max_reduction[i], self.max_reduction[i + r_stride])

    @ti.kernel
    def post_process_udf(self, bit_pic: ti.template(), n: ti.i32, coff: ti.f32, offset: ti.f32):
        for i, j in self.output_pic:
            self.output_pic[i, j] = vec3(ti.cast(ti.sqrt(bit_pic[n, i, j][2]) * coff + offset, ti.u32))

    @ti.kernel
    def post_process_sdf(self, bit_pic_w: ti.template(), bit_pic_b: ti.template(), n: ti.i32, coff: ti.f32,
                         offset: ti.f32):
        for i, j in self.output_pic:
            self.output_pic[i, j] = vec3(
                ti.cast((ti.sqrt(bit_pic_w[n, i, j][2]) - ti.sqrt(bit_pic_b[n, i, j][2])) * coff + offset, ti.u32))

    @ti.kernel
    def post_process_sdf_linear_1channel(self, bit_pic_w: ti.template(), bit_pic_b: ti.template(), n: ti.i32):
        for i, j in self.output_pic:
            self.output_linear[i, j][0] = ti.sqrt(bit_pic_w[n, i, j][2]) - ti.sqrt(bit_pic_b[n, i, j][2])

    # @ti.kernel
    # def print_p(self, n: ti.i32):
    #     print(n, '\n')
    #     for i, j in ti.ndrange(self.width, self.height):
    #         print('i:', i, 'j:', j, 'store:', self.bit_pic[n, i, j][0], self.bit_pic[n, i, j][1],
    #               self.bit_pic[n, i, j][2])
    #     print('\n')

    def gen_udf(self, dist_buffer, keep_white=True):

        keep_white_para = 1 if keep_white else -1
        self.pre_process(dist_buffer, keep_white_para)
        self.num = 0
        stride = self.width >> 1
        while stride > 0:
            self.jump_flooding(dist_buffer, stride, self.num)
            stride >>= 1
            self.num = 1 - self.num

        self.jump_flooding(dist_buffer, 2, self.num)
        self.num = 1 - self.num

        self.jump_flooding(dist_buffer, 1, self.num)
        self.num = 1 - self.num

    def find_max(self, dist_buffer):
        self.copy(dist_buffer)

        r_stride = self.width * self.height >> 1
        while r_stride > 0:
            self.max_reduction_kernel(r_stride)
            r_stride >>= 1

        return self.max_reduction[0]

    def mask2udf(self, normalized=(0, 1), to_rgb=True, output=True):  # unsigned distance
        self.pic.from_numpy(self.im)
        self.gen_udf(self.bit_pic_white)

        max_dist = ti.sqrt(self.find_max(self.bit_pic_white))

        if to_rgb:  # scale sdf proportionally to [0, 1]
            coefficient = 255.0 / max_dist
            offset = 0.0
        else:
            coefficient = (normalized[1] - normalized[0]) / max_dist
            offset = normalized[0]

        self.post_process_udf(self.bit_pic_white, self.num, coefficient, offset)
        if output:
            if to_rgb:
                cv2.imwrite(self.output_filename('_udf'), self.output_pic.to_numpy())

    def gen_udf_w_h(self):
        self.pic.from_numpy(self.im)
        self.gen_udf(self.bit_pic_white, keep_white=True)
        self.gen_udf(self.bit_pic_black, keep_white=False)

    def mask2sdf(self, to_rgb=True, output=True):
        self.gen_udf_w_h()

        if to_rgb:  # grey value == 0.5 means sdf == 0, scale sdf proportionally
            max_positive_dist = ti.sqrt(self.find_max(self.bit_pic_white))
            min_negative_dist = ti.sqrt(self.find_max(self.bit_pic_black))  # this value is positive
            coefficient = 127.5 / max(max_positive_dist, min_negative_dist)
            offset = 127.5
            self.post_process_sdf(self.bit_pic_white, self.bit_pic_black, self.num, coefficient, offset)
            if output:
                # cv2.imwrite(self.output_filename('_sdf'), self.output_pic.to_numpy())
                return self.output_pic.to_numpy()
        else:  # no normalization
            if output:
                pass
            else:
                self.post_process_sdf_linear_1channel(self.bit_pic_white, self.bit_pic_black, self.num)

@ti.data_oriented
class MultiSDF2D:
    def __init__(self, file_name, file_num, sample_num=256, thresholds=None):
        self.file_name = file_name
        self.file_path = pathlib.Path(file_name)
        self.thresholds_tuple = thresholds
        self.file_num = file_num
        self.sample_num = sample_num
        self.name_base = self.file_path.stem[:-2]
        self.file_name_list = self.gen_file_list()
        self.sdf_2d = SDF2D(self.file_name_list[0])
        self.width, self.height = self.sdf_2d.width, self.sdf_2d.height
        self.sdf_buffer = ti.field(dtype=ti.f32, shape=(self.width, self.height, file_num))
        self.output_pic = ti.Vector.field(3, dtype=ti.i32, shape=(self.width, self.height))
        self.thresholds = ti.field(dtype=ti.i32, shape=file_num)

    def calc_thresholds(self):
        if self.thresholds_tuple:
            diff = self.thresholds_tuple[-1] - self.thresholds_tuple[0]
            for i in range(self.file_num):
                self.thresholds[i] = int(self.thresholds_tuple[i] / diff * self.sample_num)
                print(self.thresholds[i])
        else:
            for i in range(self.file_num):
                self.thresholds[i] = ti.floor(i / (self.file_num - 1) * self.sample_num)
                print(self.thresholds[i])

    def output_filename(self, ins='output'):
        out_dir = self.file_path.parent / 'output'
        if not (out_dir.exists() and out_dir.is_dir()):
            out_dir.mkdir()
        return str(out_dir / (self.name_base + ins + self.file_path.suffix))

    def gen_file_list(self):
        lst = []
        for i in range(self.file_num):
            name = str(self.file_path.parent / f'{self.name_base}_{i + 1}{self.file_path.suffix}')
            lst.append(name)
        return lst

    def blur_mix_sdf(self):
        for k, sdf in enumerate(self.file_name_list):
            self.sdf_2d.reset(sdf)
            self.sdf_2d.gen_udf_w_h()
            self.create_sdf_buffer(k)
        self.calc_thresholds()
        self.blur_mix(self.thresholds)
        cv2.imwrite(self.output_filename('_blur_mix'), self.output_pic.to_numpy())

    def create_sdf_buffer(self, k):
        self.copy_sdf_buffer(k, self.sdf_2d.bit_pic_white, self.sdf_2d.bit_pic_black,
                             self.sdf_2d.num)

    @ti.kernel
    def copy_sdf_buffer(self, k: ti.i32, bit_pic_w: ti.template(), bit_pic_b: ti.template(), n: ti.i32):
        for i, j in ti.ndrange(self.width, self.height):
            self.sdf_buffer[i, j, k] = ti.sqrt(bit_pic_w[n, i, j][2]) - ti.sqrt(bit_pic_b[n, i, j][2])

    @ti.func
    def cal_grey_value(self, dis1, dis2, interval_l, interval_r):
        value = vec3(0)
        interval_len = interval_r - interval_l - 1
        if dis1 < -eps and dis2 < -eps:
            value = vec3(255) * (interval_len + 1)
        elif dis1 > 0.0 and dis2 > 0.0:
            pass
        else:
            res = 0
            for n in range(interval_l, interval_r):
                mix = (n - interval_l) / interval_len
                if (1 - mix) * dis1 + mix * dis2 < -eps:
                    res += 255
            value = vec3(res)
        return value

    @ti.kernel
    def blur_mix(self, thresholds: ti.template()):
        for i, j in self.output_pic:

            for k in range(self.file_num - 1):
                self.output_pic[i, j] += self.cal_grey_value(self.sdf_buffer[i, j, k], self.sdf_buffer[i, j, k + 1],
                                                             thresholds[k], thresholds[k + 1])
            self.output_pic[i, j] = int(self.output_pic[i, j] / self.sample_num)

def pixel_to_distance(pixel_value, coefficient, offset):
    return (pixel_value - offset) / coefficient

def image_to_sdf(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # 如果image是一个三通道图像，只取一个通道
        image = image[:, :, 0]
    sdf = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sdf[i, j] = pixel_to_distance(image[i, j], 127.5, 127.5)
    return sdf


def mesh_from_sdf(logits, mini, maxi, resolution):
    # logits = np.reshape(logits, (resolution,) * 3)

    # logits *= -1

    # padding to ba able to retrieve object close to bounding box bondary
    # logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=1000)
    threshold = 0.0
    vertices, triangles = mcubes.marching_cubes(-logits, threshold)

    # rescale to original scale
    step = (np.array(maxi) - np.array(mini)) / (resolution - 1)
    vertices = vertices * np.expand_dims(step, axis=0)
    vertices += [mini[0], mini[1], mini[2]]

    return trimesh.Trimesh(vertices, triangles)


def sdf2d_3d(sdf_image,viz_3d=False):

    # 定义z轴层数
    z_layers = 32
    sdf_2d = image_to_sdf(sdf_image)
    sdf_2d = cv2.resize(sdf_2d, (32,32),interpolation=cv2.INTER_AREA)
    # 创建3D Voxel Grid
    sdf_3d = np.zeros((sdf_2d.shape[0], sdf_2d.shape[1], z_layers))

    middle_layer = z_layers //2 # 选择z轴的中间层放置2D SDF
    sdf_3d[:, :, middle_layer] = sdf_2d  # 在中间层放置2D SDF
    def normalize_array_to_neg_pos_half(arr):
        min_vals = arr.min(axis=(0,1,2), keepdims=True)
        max_vals = arr.max(axis=(0,1,2), keepdims=True)
        
        # 归一化到 [0, 1]
        normalized_arr = (arr - min_vals) / (max_vals - min_vals)
        
        # 转换到 [-0.5, 0.5]
        normalized_arr = normalized_arr - 0.5
        
        return normalized_arr
    sdf_norm = normalize_array_to_neg_pos_half(sdf_3d)
    # 简单地可视化一下这个3D Voxel Grid
    if viz_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = np.where(sdf_3d < 0)  # 只展示SDF值小于0的voxels，您可以根据您的需求调整这一部分
        ax.scatter(x, y, z)

        plt.show()
    return sdf_3d

if __name__ == "__main__":
        # 假设您的2D SDF是一个圆形，中心点在(50, 50)，半径是40
    img_name = r'/home/yang/projects/parametric-leaf/dataset/LeafData/Bael/healthy/Bael_healthy_0001_mask_aligned.JPG'

    mySDF2D = SDF2D(img_name)
    sdf_2d = mySDF2D.mask2sdf()
    # sdf_2d = cv2.imread('/home/yang/projects/parametric-leaf/dataset/LeafData/Basil/healthy/mask/output/0_sdf.png')
    sdf_3d = sdf2d_3d(sdf_2d, viz_3d=False)
    
    mini = [-.95, -.95, -.95]
    maxi = [0.95, 0.95, 0.95]   
    mesh = mesh_from_sdf(sdf_3d,mini=mini, maxi=maxi , resolution=256)
    mesh.export('test_2.obj')
    
