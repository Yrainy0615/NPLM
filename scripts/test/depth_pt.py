import os
import OpenEXR
import cv2
import imageio
import numpy as np
import array
from matplotlib import pyplot as plt

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
def load_depth(path):
    depth_map = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # shape = (N, N, 3)
    return depth_map[..., 0]  # shape = (N, N)

if __name__ == "__main__":
    depth_test = 'depth_159.exr'
    depth_map = load_depth(depth_test)

  #   cv2.imshow('depth', depth_map)
    cv2.imwrite('depth.png', depth_map)
    #However, this works well

    #-> 0x0001 is the code for EXR_FLOAT in saving images: https://github.com/imageio/imageio/issues/356

    #Undistort the image based on pixel distance to center cx, cy and focal length in pixel units f_pix
    #cx, cy, f_pix are from the true camera intrinsics extracted from Blender
