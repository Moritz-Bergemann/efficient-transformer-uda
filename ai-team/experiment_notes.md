# Experiments
## General Repo Architecture Notes
[`UDADecorator`](mmseg/models/uda/uda_decorator.py) is a wrapper around a model for the performing of UDA.
[`DACS`](mmseg/models/uda/dacs.py) (Domain Adaptation via Cross-domain mixed Sampling) is a particular method for performing pseudolabelling-based UDA that was adapted for DAFormer in this repository.

### Common methods
`forward_train` is the method called by `BaseSegmentor` when `forward` (i.e. `__call__`) is called on it during training.

## General Setup
Install miniconda (on my Linux, I installed using the source sh script.)

Create a new conda environment (python 3.8)

```sh
conda create --name das python=3.8; conda activate das
conda create --name efficient-das python=3.8; conda activate efficient-das
```

Install pytorch
```sh
conda install pytorch==1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch-lts -c nvidia
```

Install mmcv locally (mmsegmentation is in this repository, so does not need to be installed)
```sh
pip install -U openmim
mim install mmcv-full==1.3.7
```

Install other dependencies (this is how we're currently doing it)
```
pip install -r requirements.txt
```

Install the cityscapes and GTA datasets from [here](https://www.cityscapes-dataset.com/downloads/) and [here](https://download.visinf.tu-darmstadt.de/data/from_games/). Symlink/move them to `data/citsyscapes` and `data/gta`.

Then, run the dataset conversion scripts:
```sh
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```
# Things to note
SegFormer's attention is performed along channels rather than along features.

Why don't we just do the domain classifier like SegFormer do theirs?
- Alternatively, just do a real simple conv based one
    - In the thesis would need to discuss effect of type of features on the conv 

For utility:
```
--gpu-id 0
--gpu-id 1
```

# Experiments
## Experiment 0 - Baseline


### Notes
- Training is quite unstable - large (~5% differences in mIoU between runs)

## GTA2CS, MiTB5
```sh
python run_experiments.py --config configs/baseline/gta2cs_sourceonly_mitb5.py --gpu-id 0
```

## GTA2CS, MiTB4
```sh
python run_experiments.py --config configs/baseline/gta2cs_sourceonly_mitb4.py --gpu-id 0
```

## Experiment 1 - Domain Adversarial Loss

### With RCS
```
python run_experiments.py --config configs/domain_adversarial_segformer/gta2cs_das_basic_convdisc_rcs_mitb4.py --gpu-id 1
```
## Experiment 2 - Patch-wise Domain Adversarial Loss
## Experiment 3 - Patch-wise Domain Invariant Feature Encouragement (a la TVT)
## Experiemnt 3 - Patch-wise Feature Distances
## Experiment 4 - Cross-Attention

# A new network architecture