from mmseg.models import UDA
from mmseg.models.uda.uda_decorator import UDADecorator
import torch
from copy import deepcopy

@UDA.register_module()
class AdversarialUDA(UDADecorator):

    @staticmethod
    def build_discriminator(cfg):
        raise NotImplementedError("TODO - make this make a discriminator")

    def __init__(self, **cfg):
        super(AdversarialUDA, self).__init__(**cfg)

        self.discriminator = AdversarialUDA.build_discriminator(deepcopy(cfg['discriminator']))

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
        log_vars = self(**data_batch) # M: I'M PRETTY SURE THIS CALLS FORWARD_TRAIN
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
        # Compute batch source segmentation loss
        clean_losses = self.get_model().forward_train( # M: NOTE: Losses returned as defined in BaseSegmentor
        img, img_metas, gt_semantic_seg, return_feat=False) # M: Do training, calc losses & other data
        src_feat = clean_losses.pop('features') # M: Get source dataset features (for calculating adversarial loss)
        src_seg_loss, clean_log_vars = self._parse_losses(clean_losses)

        # Compute batch source adversarial loss
        src_disc_labels = torch.zeros(batch_size)
        src_disc_loss, src_disc_log_vars = self.discriminator.forward_train(src_feat, src_disc_labels)
        log_vars.update(src_disc_log_vars)

        # Compute batch target adversarial loss
        # First, produce features for target images 
        target_feat = self.get_model().extract_feat(target_img)

        target_disc_labels = torch.ones(batch_size)
        target_disc_loss, target_disc_log_vars = self.discriminator.forward_train(target_feat, target_disc_labels)
        log_vars.update(target_disc_log_vars)

        loss = src_seg_loss + src_disc_loss + target_disc_loss

        loss.backward()

        return log_vars

        # TODO potential experiment - is it better to just do objectives one at a time or mix different ones between each other?
            # ANSWER - don't think it matters because this is doing one batch at a time

class AdversarialDiscriminator:
    def __init__(input_shape, weights=None, classes=2):
        layers = torch.linear()
    
    def forward_train(img, labels):
        raise NotImplementedError("TODO")