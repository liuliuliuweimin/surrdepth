# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch

from packnet_sfm.models.SfmModel import SfmModel
from packnet_sfm.losses.supervised_loss import SupervisedLoss
from packnet_sfm.utils.depth import depth2inv
from packnet_sfm.losses.multiview_photometric_loss import MultiViewPhotometricLoss
from packnet_sfm.models.model_utils import merge_outputs
from packnet_sfm.geometry.pose import Pose
import torch


class SupModel(SfmModel):
    """
    Model that only inherits a depth networks, and includes the supervision loss of depth

    Parameters
    ----------
    supervised_loss_weight : float
        Weight for the supervised loss
    kwargs : dict
        Extra parameters
    """

    def __init__(self, supervised_loss_weight=1.0, **kwargs):
        # Initializes SelfSupModel
        super().__init__(**kwargs)
        assert supervised_loss_weight != 0, "Model requires supervision"

        # Initializes the photometric loss
        self._photometric_loss = MultiViewPhotometricLoss(**kwargs)

        # Initializes the supervision loss
        self.supervised_loss_weight = supervised_loss_weight
        self._supervised_loss = SupervisedLoss(**kwargs)

        # Pose network is only required if there is self-supervision
        self._network_requirements['pose_net'] = 0

        # GT depth and pose are both required if there is supervision
        self._train_requirements['gt_depth'] = self.supervised_loss_weight != 0
        self._train_requirements['gt_pose'] = self.supervised_loss_weight != 0

    @property
    def logs(self):
        """Return logs."""
        return {
            **super().logs,
            **self._photometric_loss.logs,
            **self._supervised_loss.logs
        }

    def multiview_photometric_loss(self, image, ref_images, inv_depths, poses,
                                   intrinsics, extrinsics, return_logs=False, progress=0.0):
        """
        Calculates the multiview photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        ref_images : list of torch.Tensor [B,3,H,W]
            Reference images from context
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        poses : list of Pose groundtruth [Pose, Pose] each Pose object is in shape of [6,4,4]
            List containing predicted poses between original and context images
        intrinsics : torch.Tensor [B,3,3]
            Camera intrinsics
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._photometric_loss(
            image, ref_images, inv_depths, intrinsics, intrinsics, extrinsics, poses,
            return_logs=return_logs, progress=progress)

    def supervised_loss(self, inv_depths, gt_inv_depths,
                        return_logs=False, progress=0.0):
        """
        Calculates the supervised loss.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        gt_inv_depths : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth maps from the original image
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._supervised_loss(
            inv_depths, gt_inv_depths,
            return_logs=return_logs, progress=progress)

    def forward(self, batch, return_logs=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        if not self.training:
            # If not training, no need for self-supervised loss
            return SfmModel.forward(self, batch)
        else:
            # Calculate predicted depth and pose output
            output = super().forward(batch, return_logs=return_logs)

            # Introduce poses ground_truth
            poses_gt = [[], []]
            poses_gt[0], poses_gt[1] = torch.zeros((6, 4, 4)), torch.zeros((6, 4, 4))
            for i in range(6):
                poses_gt[0][i] = batch['pose_context'][0][i].inverse() @ batch['pose'][i]
                poses_gt[1][i] = batch['pose_context'][1][i].inverse() @ batch['pose'][i]
            poses_gt = [Pose(poses_gt[0]), Pose(poses_gt[1])]

            multiview_loss = self.multiview_photometric_loss(
                batch['rgb_original'], batch['rgb_context_original'],
                output['inv_depths'], poses_gt, batch['intrinsics'], batch['extrinsics'],
                return_logs=return_logs, progress=progress)

            # loss = multiview_loss['loss']
            loss = 0.
            # Calculate supervised loss
            supervision_loss = self.supervised_loss(output['inv_depths'], depth2inv(batch['depth']),
                                                    return_logs=return_logs, progress=progress)
            loss += self.supervised_loss_weight * supervision_loss['loss']

            # Return loss and metrics
            return {
                'loss': loss,
                **merge_outputs(merge_outputs(multiview_loss, supervision_loss), output)
            }
