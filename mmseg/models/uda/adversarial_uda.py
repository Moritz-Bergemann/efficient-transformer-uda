from mmseg.models import UDA
from mmseg.models.uda.uda_decorator import UDADecorator
import torch
from torch import nn
from mmcv.runner import BaseModule
from copy import deepcopy


@UDA.register_module()
class AdversarialUDA(UDADecorator):

    def __init__(self, **cfg):
        # Build model etc in UDADecorator
        super(AdversarialUDA, self).__init__(**cfg)

        # M-TODO add something about the loss weighting schedule here

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        optimizer.zero_grad()
        log_vars = self(**data_batch) # call forward_train
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img, # M: THIS TAKES IN A BATCH OF IMAGES, IS CALLED BY TRAIN_STEP
                    target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Source dataset
        # Compute batch source segmentation and adversarial loss
        gt_src_disc = torch.zeros(batch_size, device=dev, dtype=torch.long)

        clean_losses = self.get_model().forward_train( # M-NOTE: Losses returned as defined in BaseSegmentor
        img, img_metas, gt_semantic_seg, gt_src_disc) # M: Do training, calc losses & other data
        src_loss, src_log_vars = self._parse_losses(clean_losses)

        log_vars.update(src_log_vars)

        # Compute batch target adversarial loss
        gt_target_disc = torch.ones(batch_size, device=dev, dtype=torch.long)
        clean_target_disc_losses = self.get_model().forward_train(
            target_img, target_img_metas, None, gt_target_disc)
        target_disc_loss, target_disc_log_vars = self._parse_losses(clean_target_disc_losses)
        log_vars.update(target_disc_log_vars)

        # Compute final loss
        loss = src_loss + target_disc_loss # M-TODO probably make this weighted with CFG params - but idk, should the weighting go here? Maybe the weighting that increases by total steps should
        loss.backward()

        return log_vars