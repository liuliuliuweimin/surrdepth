# Copyright 2020 Toyota Research Institute.  All rights reserved.
import numpy as np
import math
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

from packnet_sfm.utils.image import match_scales
from packnet_sfm.geometry.camera import Camera
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera_utils import view_synthesis
from packnet_sfm.utils.depth import calc_smoothness, inv2depth
from packnet_sfm.losses.loss_base import LossBase, ProgressiveScaling

########################################################################################################################

def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim

########################################################################################################################

class MultiViewPhotometricLoss(LossBase):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them

    Parameters
    ----------
    num_scales : int
        Number of inverse depth map scalesto consider
    ssim_loss_weight : float
        Weight for the SSIM loss
    occ_reg_weight : float
        Weight for the occlusion regularization loss
    smooth_loss_weight : float
        Weight for the smoothness loss
    C1,C2 : float
        SSIM parameters
    photometric_reduce_op : str
        Method to reduce the photometric loss
    disp_norm : bool
        True if inverse depth is normalized for
    clip_loss : float
        Threshold for photometric loss clipping
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    padding_mode : str
        Padding mode for view synthesis
    automask_loss : bool
        True if automasking is enabled for the photometric loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, num_scales=4, ssim_loss_weight=0.85, occ_reg_weight=0.1, smooth_loss_weight=0.1,
                 consistency_loss_weight=0.1, C1=1e-4, C2=9e-4, photometric_reduce_op='mean', disp_norm=True,
                 clip_loss=0.5, progressive_scaling=0.0, padding_mode='zeros', t_loss_weight=0.1, R_loss_weight=0.1,
                 temporal_loss_weight=1.0, spatial_loss_weight=0.1, automask_loss=True,
                 consistency_loss=True, cameras=None, **kwargs):
        super().__init__()
        self.n = num_scales
        self.progressive_scaling = progressive_scaling
        self.ssim_loss_weight = ssim_loss_weight
        self.occ_reg_weight = occ_reg_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.C1 = C1
        self.C2 = C2
        self.photometric_reduce_op = photometric_reduce_op
        self.disp_norm = disp_norm
        self.clip_loss = clip_loss
        self.padding_mode = padding_mode
        self.automask_loss = automask_loss
        self.consistency_loss = consistency_loss
        self.t_loss_weight = t_loss_weight
        self.R_loss_weight = R_loss_weight
        self.temporal_loss_weight = temporal_loss_weight
        self.spatial_loss_weight = spatial_loss_weight
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)
        self.cameras = cameras

        # Asserts
        if self.automask_loss:
            assert self.photometric_reduce_op == 'min', \
                'For automasking only the min photometric_reduce_op is supported.'
        if self.consistency_loss:
            assert self.cameras is not None, 'Need camera number parameters for consistency loss'
            self.num_cameras = len(self.cameras)

########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'num_scales': self.n,
        }

########################################################################################################################

    def warp_ref_image_temporal(self, inv_depths, ref_image, K, ref_K, pose):
        """
        Warps a reference image to produce a reconstruction of the original one (temporal-wise).

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float()).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)
        ref_warped = []
        ref_coords = []
        for i in range(self.n):
            w,c = view_synthesis(ref_images[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode)
            ref_warped.append(w)
            ref_coords.append(c)
        return ref_warped

    def warp_ref_image_spatial(self, inv_depths, ref_image, K, ref_K, extrinsics_1,extrinsics_2):
        """
        Warps a reference image to produce a reconstruction of the original one (spatial-wise).

        Parameters
        ----------
        inv_depths : torch.Tensor [6,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [6,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        extrinsics_1: torch.Tensor [B,4,4]
            target image extrinsics
        extrinsics_2: torch.Tensor [B,4,4]
            context image extrinsics

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        valid_points_mask :
            valid points mask
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        # Generate cameras for all scales
        cams, ref_cams = [], []
        for i in range(self.n):
            _, _, DH, DW = inv_depths[i].shape
            scale_factor = DW / float(W)
            cams.append(Camera(K=K.float(),Tcw=extrinsics_1).scaled(scale_factor).to(device))
            ref_cams.append(Camera(K=ref_K.float(), Tcw=extrinsics_2).scaled(scale_factor).to(device))
        # View synthesis
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        ref_images = match_scales(ref_image, inv_depths, self.n)
        ref_warped = []
        ref_coords = []
        for i in range(self.n):
            w,c = view_synthesis(ref_images[i], depths[i], ref_cams[i], cams[i],
            padding_mode=self.padding_mode)
            ref_warped.append(w)
            ref_coords.append(c)
        # calculate valid_points_mask
        valid_points_masks = [ref_coords[i].abs().max(dim=-1)[0] <= 1 for i in range(self.n)]
        return ref_warped, valid_points_masks

########################################################################################################################

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in range(self.n)]

        # Visualization l1_loss
        # if temporal==True and spatial==True:
        #     pic = l1_loss[0][0].cpu().clone()
        #     pic = (pic.squeeze(0).permute(1,2,0).detach().numpy()*255).astype(np.uint8)
        #     cv2.imshow('pic_{}'.format(0), pic)
        #     cv2.waitKey()
        #
        #     pic = l1_loss[0][1].cpu().clone()
        #     pic = (pic.squeeze(0).permute(1,2,0).detach().numpy()*255).astype(np.uint8)
        #     cv2.imshow('pic_{}'.format(1), pic)
        #     cv2.waitKey()
        #
        #     pic = l1_loss[0][2].cpu().clone()
        #     pic = (pic.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        #     cv2.imshow('pic_{}'.format(2), pic)
        #     cv2.waitKey()
        #
        #     pic = l1_loss[0][3].cpu().clone()
        #     pic = (pic.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        #     cv2.imshow('pic_{}'.format(3), pic)
        #     cv2.waitKey()


        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3)
                         for i in range(self.n)]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in range(self.n)]
        else:
            photometric_loss = l1_loss
        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(self.n):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

    def reduce_photometric_loss(self, photometric_losses):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        """
        # Reduce function
        def reduce_function(losses):
            if self.photometric_reduce_op == 'mean':
                return sum([l.mean() for l in losses]) / len(losses)
            elif self.photometric_reduce_op == 'min':
                return torch.cat(losses, 1).min(1, True)[0].mean()
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))
        # Reduce photometric loss
        photometric_loss = sum([reduce_function(photometric_losses[i])
                                for i in range(self.n)]) / self.n
        # Store and return reduced photometric loss
        self.add_metric('photometric_loss', photometric_loss)
        return photometric_loss

########################################################################################################################

    def calc_smoothness_loss(self, inv_depths, images):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales

        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.n)]) / self.n
        # Apply smoothness loss weight
        smoothness_loss = self.smooth_loss_weight * smoothness_loss
        # Store and return smoothness loss
        self.add_metric('smoothness_loss', smoothness_loss)
        return smoothness_loss

########################################################################################################################
    def pose_conversion(self, pose, extrinsic_1, extrinsic_2):
        # converted_pose = extrinsic_1.inverse().bmm(extrinsic_2).bmm(pose)\
        #     .bmm(extrinsic_2.inverse()).bmm(extrinsic_1)
        converted_pose = extrinsic_1.inverse()@extrinsic_2@pose@extrinsic_2.inverse()@extrinsic_1
        return converted_pose  # transformation matrix (4*4)

    def calc_consistency_loss(self, poses, extrinsics):
        """
        Calculates the consistency loss for multi-cameras pose
        Parameters
        ----------
        poses : list of torch.Tensor [[6,4,4], [6,4,4]] (transformation matrix)
            Predicted poses for 6 cameras from the poseNet [[pose(t->t-1)], [pose(t->t+1)]]
        extrinsics: list of torch.Tensor [6,4,4]
            Extrinsics matrix for 6 cameras

        Returns
        -------
        consistency_loss : torch.Tensor [1]
            Consistency loss
        """
        converted_poses_forward, converted_poses_backward = [], []
        poses_forward = poses[0].item()   # in shape of torch.tensor [6,4,4]
        poses_backward = poses[1].item()  # in shape of torch.tensor [6,4,4]

        for i in range(poses_forward.shape[0]-1):
            converted_poses_forward.append(self.pose_conversion(poses_forward[i+1], extrinsics[0], extrinsics[i+1]))
            converted_poses_backward.append(self.pose_conversion(poses_backward[i+1], extrinsics[0], extrinsics[i+1]))
        t_consistency_loss = 0
        for i in range(len(converted_poses_forward)):  # should be 5
            t_original_forward = poses_forward[i+1][:3, -1]
            t_original_backward = poses_backward[i+1][:3, -1]
            t_converted_forward = converted_poses_forward[i][:3, -1]
            t_converted_backward = converted_poses_backward[i][:3, -1]
            t_consistency_loss = t_consistency_loss + torch.norm((t_original_forward-t_converted_forward), 2) \
                                 + torch.norm((t_original_backward-t_converted_backward), 2)

        rotation_original_forward = poses[0].item()[:, :3, :3]
        rotation_original_backward = poses[1].item()[:, :3, :3]
        rotation_converted_forward, rotation_converted_backward = [], []
        for i in range(len(converted_poses_forward)):
            rotation_converted_forward.append(converted_poses_forward[i][:3, :3])
            rotation_converted_backward.append(converted_poses_backward[i][:3, :3])
        euler_original_forward, euler_original_backward = [], []
        euler_converted_forward, euler_converted_backward = [], []
        for i in range(len(converted_poses_forward)):  # should be five
            euler_original_forward.append(Rotation2euler(rotation_original_forward[i+1]))
            euler_original_backward.append(Rotation2euler(rotation_original_backward[i+1]))
            euler_converted_forward.append(Rotation2euler(rotation_converted_forward[i]))
            euler_converted_backward.append(Rotation2euler(rotation_converted_backward[i]))
        R_consistency_loss = 0
        for i in range(len(euler_converted_forward)):  # should be 5
            R_consistency_loss = R_consistency_loss \
                                 + torch.sum(torch.tensor([torch.norm(torch.tensor(euler_original_forward[j]-euler_converted_forward[j]), 2)
                                                           for j in range(len(euler_original_forward))])) \
                                 + torch.sum(torch.tensor([torch.norm(torch.tensor(euler_original_backward[j]-euler_converted_backward[j]), 2)
                                                           for j in range(len(euler_original_forward))]))
        pose_consistency_loss = self.t_loss_weight * t_consistency_loss + self.R_loss_weight * R_consistency_loss
        return pose_consistency_loss

########################################################################################################################

    def forward(self, image, context, inv_depths, K, ref_K, extrinsics, poses, return_logs=False, progress=0.0):
        """
        Calculates training photometric loss.
        (Here we need to consider temporal, spatial and temporal-spatial wise losses)

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        context : list of torch.Tensor [B,3,H,W]
            Context containing a list of reference images
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        poses : list of Pose
            Camera transformation between original and context
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.n = self.progressive_scaling(progress)
        # Loop over all reference images
        photometric_losses = [[] for _ in range(self.n)]
        images = match_scales(image, inv_depths, self.n)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32, device="cuda")
        # Step 1: Calculate the losses temporal-wise
        for j, (ref_image, pose) in enumerate(zip(context, poses)):
            # Calculate warped images
            ref_warped = self.warp_ref_image_temporal(inv_depths, ref_image, K, ref_K, pose)
            # Calculate and store image loss

            # print('### poses shape', len(poses))
            # print('poses[0].shape:', poses[0].shape)
            # print('###multiview_photometric_loss printing ref_warped')
            # print('len of images: ',len(images))
            # print('shape of images[0]: ', images[0].shape)
            # print('len of context: ',len(context))
            # print('shape of context[0]:', context[0].shape)
            # print('len of ref_warped: ',len(ref_warped))
            # print('shape of ref_warped[0]', ref_warped[0].shape)

            # pic_orig = images[0][5].cpu().clone()
            # pic_orig = (pic_orig.squeeze(0).permute(1,2,0).detach().numpy()*255).astype(np.uint8)
            # pic_ref = context[0][5].cpu().clone()
            # pic_ref = (pic_ref.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
            # pic_warped = ref_warped[0][5].cpu().clone()
            # pic_warped = (pic_warped.squeeze(0).permute(1,2,0).detach().numpy()*255).astype(np.uint8)
            # final_frame = cv2.hconcat((pic_orig, pic_ref, pic_warped))
            # cv2.imshow('temporal warping', final_frame)
            # cv2.waitKey()

            photometric_loss = self.calc_photometric_loss(ref_warped, images)
            for i in range(self.n):
                photometric_losses[i].append(photometric_loss[i])
            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(ref_image, inv_depths, self.n)
                unwarped_image_loss = self.calc_photometric_loss(ref_images, images)
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[i])

        # Step 2: Calculate the losses spatial-wise
        # reconstruct context images
        num_cameras, C, H, W = image.shape  # should be (6, 3, H, W)


        # for i in range(num_cameras):
        #     pic = image[i].cpu().clone()
        #     pic = (pic.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        #     if i == 0:
        #         final_file = pic
        #     else:
        #         final_file = cv2.hconcat((final_file, pic))
        # cv2.imshow('6 images', final_file)
        # cv2.waitKey()





        sequence = [0,2,4,5,3,1]
        # left_swap = [i for i in range(-1, num_cameras-1)]
        # right_swap = [i % 6 for i in range(1, num_cameras+1)]
        left_swap = [1,3,0,5,2,4]
        right_swap = [2,0,4,1,5,3]

        context_spatial = [[], []]  # [[B,3,H,W],[B,3,H,W]]
        context_spatial[0] = image[left_swap, ...]
        context_spatial[1] = image[right_swap, ...]
        K_spatial = K  # tensor [B,3,3]
        ref_K_spatial = [[], []]  # [[B,3,3],[B,3,3]]
        ref_K_spatial[0] = K[left_swap, ...]
        ref_K_spatial[1] = K[right_swap, ...]
        # reconstruct extrinsics
        extrinsics_1 = extrinsics  # [B,4,4]
        extrinsics_2 = [[], []]  # [[B,4,4],[B,4,4]]
        extrinsics_2[0] = extrinsics_1[left_swap, ...]
        extrinsics_2[1] = extrinsics_1[right_swap, ...]
        # calculate spatial-wise photometric loss
        for j, ref_image in enumerate(context_spatial):
            # Calculate warped images
            ref_warped, valid_points_masks = self.warp_ref_image_spatial(inv_depths, ref_image, K_spatial
                                                                         , ref_K_spatial[j], Pose(extrinsics_1), Pose(extrinsics_2[j]))

            # pic_orig = images[0][5].cpu().clone()
            # pic_orig = (pic_orig.squeeze(0).permute(1,2,0).detach().numpy()*255).astype(np.uint8)
            # pic_ref = context_spatial[0][5].cpu().clone()
            # pic_ref = (pic_ref.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
            # pic_warped = ref_warped[0][5].cpu().clone()
            # pic_warped = (pic_warped.squeeze(0).permute(1,2,0).detach().numpy()*255).astype(np.uint8)
            # pic_valid = valid_points_masks[0][5].cpu().clone()
            # pic_valid = (pic_valid.permute(1,2,0).detach().numpy()*255).astype(np.uint8)
            # final_frame = cv2.hconcat((pic_orig, pic_ref, pic_warped))
            # cv2.imshow('spatial warping', final_frame)
            # cv2.waitKey()
            # # cv2.imshow('pic_valid', pic_valid)
            # # cv2.waitKey()

            # Calculate and store image loss
            photometric_loss = [self.calc_photometric_loss(ref_warped, images)[i] * valid_points_masks[i]
                                for i in range(len(valid_points_masks))]
            for i in range(self.n):
                photometric_losses[i].append(self.temporal_loss_weight*photometric_loss[i])

        # Step 3: Calculate the loss temporal-spatial wise
        # reconstruct context images
        context_temporal_spatial = []
        # [context_temporal_spatial_backward, context_temporal_spatial_forward]
        # [[left t-1, right t-1], [left t+1, right t+1]]
        # [[[B,H,W],[B,H,W]],[[B,H,W],[B,H,W]]]
        # reconstruct intrinsics
        K_temporal_spatial = K
        ref_K_temporal_spatial = []
        # reconstruct extrinsics
        extrinsics_1_ts = extrinsics
        extrinsics_2_ts = []
        # reconstruct pose
        poses_ts = []
        for l in range(len(context)):
            context_temporal_spatial.append([context[l][left_swap, ...], context[l][right_swap, ...]])
            ref_K_temporal_spatial.append([K_temporal_spatial[left_swap, ...], K_temporal_spatial[right_swap, ...]])
            extrinsics_2_ts.append([extrinsics[left_swap, ...], extrinsics[right_swap, ...]])
            poses_ts.append([Pose(poses[l].item()[left_swap, ...]), Pose(poses[l].item()[right_swap, ...])])
        # calculate spatial-wise photometric loss
        for j, (ref_image, pose) in enumerate(zip(context_temporal_spatial, poses_ts)):
            # Calculate warped images
            for k in range(len(ref_image)):
                ref_warped, valid_points_masks = self.warp_ref_image_spatial(
                    inv_depths, ref_image[k], K_temporal_spatial, ref_K_temporal_spatial[j][k]
                    , Pose(extrinsics_1_ts), Pose(extrinsics_2_ts[j][k])@pose[k].inverse())

                # for i in range(6):
                #     pic_orig = images[0][i].cpu().clone()
                #     pic_orig = (pic_orig.squeeze(0).permute(1,2,0).detach().numpy()*255).astype(np.uint8)
                #     pic_ref = context_spatial[0][i].cpu().clone()
                #     pic_ref = (pic_ref.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
                #     pic_warped = ref_warped[0][i].cpu().clone()
                #     pic_warped = (pic_warped.squeeze(0).permute(1,2,0).detach().numpy()*255).astype(np.uint8)
                #     pic_valid = valid_points_masks[0][i].cpu().clone()
                #     pic_valid = (pic_valid.permute(1,2,0).detach().numpy()*255).astype(np.uint8)
                #     final_frame = cv2.hconcat((pic_orig, pic_ref, pic_warped))
                #     cv2.imshow('temporal spatial warping', final_frame)
                #     cv2.waitKey()
                #     cv2.imshow('pic_valid', pic_valid)
                #     cv2.waitKey()




                # Calculate and store image loss
                photometric_loss = [self.calc_photometric_loss(ref_warped, images)[i] * valid_points_masks[i] \
                                    for i in range(len(valid_points_masks))]
                for i in range(self.n):
                    photometric_losses[i].append(self.spatial_loss_weight*photometric_loss[i])

        # Step 4: Calculate reduced photometric loss
        loss = self.reduce_photometric_loss(photometric_losses)
        # Include smoothness loss if requested
        if self.smooth_loss_weight > 0.0:
            loss += self.calc_smoothness_loss(inv_depths, images)
        if self.consistency_loss:
            # poses: type List [[pose(t->t-1)], [pose(t->t+1)]]
            loss += self.calc_consistency_loss(poses, extrinsics)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }

########################################################################################################################

def set_id_grid(depth):
    b, _, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)
    return torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

def pixel2cam(depth, intrinsics_inv,image):
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, _, h, w = depth.size()
    if (image is None) or image.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = image[..., :h, :w].expand(b, 3, h, w).reshape(b, 3, -1).double()# [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth

def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)

def Rotation2euler(R):
    assert (R.shape == (3,3))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])
