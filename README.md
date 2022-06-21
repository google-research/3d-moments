# 3D Moments from Near-Duplicate Photos
This repository contains code for paper "3D Moments from Near-Duplicate Photos", CVPR 2022.

> 3D Moments from Near-Duplicate Photos  
> [Qianqian Wang](https://www.cs.cornell.edu/~qqw/), 
> [Zhengqi Li](https://zhengqili.github.io/), 
> [David Salesin](http://salesin.cs.washington.edu/), 
> [Noah Snavely](https://www.cs.cornell.edu/~snavely/),
> [Brian Curless](https://homes.cs.washington.edu/~curless/), 
> [Janne Kontkanen](https://www.linkedin.com/in/jannekontkanen/)   
> CVPR 2022

#### [Project Page](https://3d-moments.github.io/) | [Paper](https://arxiv.org/abs/2205.06255)
![video](assets/teaser.gif)

## Prerequisites
The code is tested with Python3.7, PyTorch == 1.7.1 and CUDA == 10.1. 
   We recommend you to use anaconda to make sure that all dependencies are in place. 
   To create an anaconda environment:
```
conda env create -f environment.yml
conda activate 3d_moments
```

Download pretrained models. 
The following script will download pretrained models for: 1) our pretrained model, 2) optical flow estimator [RAFT](https://github.com/princeton-vl/RAFT), 3) monocular depth estimator [DPT](https://github.com/isl-org/DPT), and 4) [RGBD-inpainting networks](https://github.com/vt-vl-lab/3d-photo-inpainting).
```
./download.sh
```

## Demos
We provided some sample input pairs in the `demo/` folder. You can render space-time videos on them using our pretrained model:


```
python demo.py --input_dir demo/001/ --config configs/render.txt
```


## Training
### Training datasets
Our pretrained model is trained on both the [Mannequin Challenge Dataset](https://google.github.io/mannequinchallenge/www/index.html)
and the [Vimeo-90k Dataset](http://toflow.csail.mit.edu/). 
The original Mannequin Challenge Dataset only provides the camera parameters. 
To train on the Mannequin Challenge dataset, 
one would need to first run Structure from Motion (SfM) to get sparse point clouds, and then compute a scale and a shift 
vector to align SfM depths with monocular depths.
Unfortunately, we do not plan to release our generated SfM point clouds on Mannequin Challenge dataset 
due to privacy concerns, and one would
need to run SfM (or triangulation) themselves. 
However, one can still train a model only on the Vimeo-90k Dataset.

To train on Vimeo-90k dataset, you should first download the dataset with our 
pre-computed DPT depths:
```
gdown https://drive.google.com/uc?id=1EhKJE27SVc32XFjYJJpDbD4ZXO5t8dnX -O data/
unzip data/vimeo_sequences.zip -d data/vimeo/
```


We recommend you to train the model with multiple GPUs:
```
# this example uses 8 GPUs (nproc_per_node=8) 
python -m torch.distributed.launch --nproc_per_node=8 train.py --config configs/train.txt
```
Alternatively, you can train with a single GPU by setting distributed=False in
`configs/train.txt` and running:

```
python train.py --config configs/train.txt
```


## Citation
```
@inproceedings{wang2022_3dmoments,
  title     = {3D Moments from Near-Duplicate Photos},
  author    = {Wang, Qianqian and Li, Zhengqi and Curless, Brian and Salesin, David and Snavely, Noah and Kontkanen, Janne},
  booktitle = {CVPR},
  year      = {2022}
}
```

## Disclaimers
Open source release prepared by Qianqian Wang.

This is not an official Google product.
