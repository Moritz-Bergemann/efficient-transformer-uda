_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/daformer_sepaspp_mitb0.py",
    "../_base_/datasets/gta_to_cityscapes_512x512.py",
    "../_base_/schedules/adamw.py",
    "../_base_/schedules/poly10warm.py"
]
n_gpus = 1
seed = 0

data = dict(
    samples_per_gpu = 4,
    workers_per_gpu = 8,
    train = dict()
)
optimizer = dict(
    lr = 6e-05,
    paramwise_cfg = dict(
        custom_keys = dict(
            head = dict(
                lr_mult = 10.0
            ),
            pos_block = dict(
                decay_mult = 0.0
            ),
            norm = dict(
                decay_mult = 0.0
            )
        )
    )
)
runner = dict(
    type = "IterBasedRunner",
    max_iters = 40000
)
checkpoint_config = dict(
    by_epoch = False,
    interval = 20000,
    max_keep_ckpts = 1
)
evaluation = dict(
    interval = 4000,
    metric = "mIoU"
)
name = "gta2cs_sourceonnly_daformer_mitb0"
exp = 'baseline'
name_dataset = "gta2cityscapes"
name_architecture = "segformer_mitb0"
name_encoder = "segformer_mitb0"
name_decoder = "daformer"
name_uda = "source-only"
name_opt = "adamw_6e-05_pmTrue_poly10warm_1x2_0k"