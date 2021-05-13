import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import os
from packnet_sfm.utils.image import write_image
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import prepare_dataset_prefix

rgbd_images = []
num_cameras = 6
sensor_names = ['CAMERA_01','CAMERA_05','CAMERA_06','CAMERA_07','CAMERA_08','CAMERA_09']
file_path = [os.path.join('/home/thuar/Desktop/surround_depth/data/save/depth/ddad_tiny-val-lidar/PackNet01_MR_selfsup_D'
						  , sensor_names[i]) for i in range(len(sensor_names))]

for i in range(num_cameras):
	rgb_filepath = os.path.join(file_path[i], '15616458297936490_rgb.png')
	depth_filepath = os.path.join(file_path[i], '15616458297936490_depth.png')
	color_raw = o3d.io.read_image(rgb_filepath)
	depth_raw = o3d.io.read_image(depth_filepath)
	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
	rgbd_images.append(rgbd_image)

plt.figure(1)
plt.subplot(1, 2, 1)
plt.title('Original image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd_image.depth)
plt.show()

inter = o3d.camera.PinholeCameraIntrinsic()
pcds = []
fx = [2181.530254382263, 1057.0685006265498, 1060.7556963722263
	, 1058.949432375444, 1057.2909120690622, 1063.4579724697585]
fy = [2181.6034466840038, 1055.9745605834144, 1059.2549314610033
	, 1056.7776258435003, 1060.1497929739965, 1065.2223608725776]
cx = [928.0218817103168, 964.6834850172629, 946.5584747928981
	, 966.0260232796186, 966.4159674505777, 944.6657652450222]
cy = [615.9567879463117, 588.6612498587956, 611.407084771089
	, 615.2019014245489, 619.2699631002707, 612.6983955581646]

x_scale, y_scale = 384/1216, 640/1936
for i in range(len(fx)):
	fx[i] = fx[i]*x_scale
	fy[i] = fy[i]*y_scale
	cx[i] = (cx[i]+0.5)*x_scale-0.5
	cy[i] = (cy[i]+0.5)*y_scale-0.5

# calibration.txt
t1 = [1.4855427639599839, 1.52093975696107, 1.5330512698187704
	,1.1047227373333044, 1.0954118728427602, 0.14706852359404365]
t2 = [0.28616353316692766, 0.4574157637366625, -0.4134779206162875
	,0.4272270943622516, -0.46058894121415506, 0.13320383159566518]
t3 = [1.5617304615771417, 1.5753242209317193, 1.5332483324642254
	,1.563046883152822, 1.55477381656209, 1.5298214057698232]

q1 = [0.512734825660536, 0.6607769368469352, 0.2256789490858021
	,0.6700935030735806, 0.21126882397726474, 0.4977373598124163]
q2 = [-0.5204131359731259, -0.6751131476133265, -0.22195853829756987
	,-0.6832162910666042, -0.2086369912947377, -0.4936700173681212]
q3 = [0.483559134543592, 0.23003455787756122, 0.6685661783754228
	,-0.20377261275794284, -0.6756492997311923, -0.49550743704266875]
q4 = [-0.48212418510268484, -0.2338379349739517, -0.6729210089964616
	, 0.2065597227185677, 0.6747844942287067, 0.5128545743523507]

for i in range(len(fx)):
	inter.set_intrinsics(height=384, width=640, fx=fx[i], fy=fy[i], cx=cx[i], cy=cy[i])
	pcd = o3d.geometry.PointCloud().create_from_rgbd_image(rgbd_images[i], inter)
	pcd = pcd.translate((t1[i], t2[i], t3[i]))
	R = pcd.get_rotation_matrix_from_quaternion((q1[i], q2[i], q3[i], q4[i]))
	pcd = pcd.rotate((R), center=(0, 0, 0))
	# pcd = pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	pcds.append(pcd)


o3d.visualization.draw_geometries([pcds[i] for i in range(len(pcds))])
# o3d.visualization.draw_geometries([pcds[0]])

