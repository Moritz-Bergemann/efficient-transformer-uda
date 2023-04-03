_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/fast_scnn.py",
    "../_base_/datasets/cityscapes_pseudolabel_512x512.py",
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
    lr = 0.00012,
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
name = "gta2cs_pseudoteacher_fastscnn"
exp = 'pseudo-teacher'
name_dataset = "gta2cityscapes"
name_architecture = "fast-scnn"
name_encoder = "fast-scnn"
name_decoder = "sep_fcn_head"
name_uda = "pseudo-teacher"
name_opt = "adamw_6e-05_pmTrue_poly10warm_1x2_0k"