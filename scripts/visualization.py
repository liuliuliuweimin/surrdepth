# plt.figure(1)
# plt.subplot(1, 2, 1)
# plt.title('Original image')
# plt.imshow(rgbd_image.color)
# plt.subplot(1, 2, 2)
# plt.title('Depth image')
# plt.imshow(rgbd_image.depth)
# plt.show()


import os
import cv2
import open3d as o3d
import numpy as np
import json
from packnet_sfm.geometry.pose_utils import quat2mat
from packnet_sfm.geometry.camera_utils import scale_intrinsics
import matplotlib.pyplot as plt


def visual_pointcloud_multi(depths_path, imgs_rgb_path, intrinsics, extrinsics, depth_scales=None):
    """
    :param depth: List of path of depth.png
    :param img_rgb: List of path of img_rgb.png
    :param intrinsics: List of np array [3,3]
    :param extrinsics: List of np array [4,4]
    :param depth_scales: List of floats, e.g. [2] for 2*depth[0]
    :return: List for Point Clouds
    """
    assert len(depths_path) == len(imgs_rgb_path) and len(depths_path) == len(intrinsics), \
        "wrong length for input List of path"
    T = extrinsics
    ext = [np.eye(4)] * len(T)
    if depth_scales is not None:
        ext = [ext[i] / depth_scales[i] for i in range(len(ext))]
    H, W, _ = cv2.imread(imgs_rgb_path[0]).shape
    pcds = []
    # Read depth and rgb images
    inter = o3d.camera.PinholeCameraIntrinsic()
    for idx, depth_path in enumerate(depths_path):
        img_o3d = o3d.io.read_image(imgs_rgb_path[idx])
        depth_o3d = o3d.io.read_image(depth_path)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img_o3d, depth_o3d, 1000,
                                                                        convert_rgb_to_intensity=False)
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.title('Original image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(rgbd_image.depth)
        plt.show()
        inter.set_intrinsics(height=H, width=W,
                             fx=intrinsics[idx][0, 0], fy=intrinsics[idx][1, 1],
                             cx=intrinsics[idx][0, 2], cy=intrinsics[idx][1, 2])
        pcd = o3d.geometry.PointCloud().create_from_rgbd_image(rgbd_image, inter, ext[idx])
        pcd = pcd.rotate(T[idx][:3, :3], center=[0, 0, 0]).translate(T[idx][3, :3]).transform(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)
    return pcds


if __name__ == '__main__':
    root_path = '/home/thuar/Desktop/surround_depth/data/save/depth/ddad_tiny-val-lidar/epoch=10_ddad_tiny-val-lidar-abs_rel=0'
    calibration_path = '/home/thuar/Desktop/surround_depth/data/datasets/DDAD_tiny/000150/calibration'
    cam = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
    rgb_filepath = [os.path.join(root_path, cam[i], '15616458296936490_rgb.png') for i in range(6)]
    depth_filepath = [os.path.join(root_path, cam[i], '15616458296936490_depth.png') for i in range(6)]
    intrinsics = [
        np.load(os.path.join(root_path, cam[i], '15616458296936490_depth.npz'))['intrinsics'].astype(np.float64) for i
        in range(6)]
    x_scale = 384 / 1216
    y_scale = 640 / 1936
    for intrinsic in intrinsics:
        intrinsic = scale_intrinsics(intrinsic, x_scale, y_scale)
    extrinsics = []
    with open(os.path.join(calibration_path, os.listdir(calibration_path)[0])) as jsonfile:
        calibration_dict = json.load(jsonfile)
        for ext_dict in calibration_dict['extrinsics'][1:]:
            extrinsics.append(
                quat2mat(**ext_dict['rotation'], **ext_dict['translation']).squeeze(0).numpy().astype(np.float64))
    # depth_scale = [11.479507530763469, 5.566449630803877, 5.075467275815063, 6.078948620498273, 5.243490168481688,
    #                5.960468297881719]
    depth_scale = [2, 1, 1, 1.2, 1, 1]
    visual_pointcloud_multi(depth_filepath, rgb_filepath, intrinsics, extrinsics, depth_scale)
