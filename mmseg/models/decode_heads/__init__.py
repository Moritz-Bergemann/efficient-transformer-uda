# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .aspp_head import ASPPHead
from .da_head import DAHead
from .daformer_head import DAFormerHead
from .dlv2_head import DLV2Head
from .fcn_head import FCNHead
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead
from .das_head import DASHead
from .das_basic_head import DASBasicHead
from .das_multi_head import DASMultiHead
from .das_output_head import DASOutputHead
from .simple_head import SimpleHead
from .sep_fcn_head import DepthwiseSeparableFCNHead

__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'DAHead',
    'DLV2Head',
    'SegFormerHead',
    'DAFormerHead',
    'ISAHead',
    'DASHead',
    'DASBasicHead',
    'DASMultiHead',
    'DASOutputHead',
    'SimpleHead',
    'DepthwiseSeparableFCNHead',
]
