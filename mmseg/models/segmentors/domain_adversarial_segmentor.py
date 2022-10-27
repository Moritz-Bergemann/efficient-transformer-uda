from mmseg.core import add_prefix

from .encoder_decoder import EncoderDecoder
from ..builder import SEGMENTORS

@SEGMENTORS.register_module()
class DomainAdversarialSegmentor(EncoderDecoder):
    """Domain adversarial segmentors.
    The adversarial discriminator isn't actually defined here. 
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DomainAdversarialSegmentor, self).__init__(backbone,
                                                         decode_head,
                                                         neck,
                                                         auxiliary_head,
                                                         train_cfg,
                                                         test_cfg,
                                                         pretrained,
                                                         init_cfg)

        # M-TODO something probably goes here, stuff like the weights to each thing

    def _decode_head_forward_train(self,
                                x,
                                img_metas,
                                gt_semantic_seg,
                                gt_disc,
                                seg_weight=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                    gt_semantic_seg,
                                                    gt_disc,
                                                    self.train_cfg,
                                                    seg_weight)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses


    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      gt_disc,
                      seg_weight=None,
                      return_feat=False):
        """Updated forward function for domain adversarial training.
        If either gt_semantic_seg or gt_disc are None, the model will train with only one or the other.

        New Args:
            gt_dict - discriminator labels
        """
        # Get model features
        x = self.extract_feat(img)

        losses = dict()
        if return_feat: # M-TODO I feel like we needed something like this, can't remember
            losses['features'] = x

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      gt_disc,
                                                      seg_weight)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        return losses