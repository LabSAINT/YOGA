** Source code and evaluation scripts for our Pattern Recognition (journal) 2023 paper. **

[Link to paper](https://www.sciencedirect.com/science/article/pii/S0031320323001516)<br>
[arXiv](https://arxiv.org/abs/2307.05945)

## YOGA: Deep Object Detection in the Wild with Lightweight Feature Learning and Multiscale Attention

### Abstract

We introduce YOGA, a deep learning based yet lightweight object detection model that can operate on low-end edge devices while still achieving competitive accuracy. The YOGA architecture consists of a two-phase feature learning pipeline with a cheap linear transformation, which learns feature maps using only half of the convolution filters required by conventional convolutional neural networks. In addition, it performs multi-scale feature fusion in its neck using an attention mechanism instead of the naive concatenation used by conventional detectors. YOGA is a flexible model that can be easily scaled up or down by several orders of magnitude to fit a broad range of hardware constraints. We evaluate YOGA on COCO-val and COCO-testdev datasets with over 10 state-of-the-art object detectors. The results show that YOGA strikes the best trade-off between model size and accuracy (up to 22% increase of AP and 23–34% reduction of parameters and FLOPs), making it an ideal choice for deployment in the wild on low-end edge devices. This is further affirmed by our hardware implementation and evaluation on NVIDIA Jetson Nano.

### Citation

```
@article{sunkara2023yoga,
  title={YOGA: Deep Object Detection in the Wild with Lightweight Feature Learning and Multiscale Attention},
  author={Sunkara, Raja and Luo, Tie},
  journal={Pattern Recognition},
  volume={139},
  pages={109451},
  year={2023},
  publisher={Elsevier},
  doi= {10.1016/j.patcog.2023.109451},
}
```

<!---
<embed src="./images/yolov5-spd_final.pdf" type="application/pdf">
-->

### YOGA Building Block:

![losses](https://github.com/raja-sunkara/pictures/blob/main/YOGA-1.png&raw=true)



### Installation

```
# Download the code 
git clone https://github.com/LabSAINT/YOGA

# Create an environment
cd SPD-Conv
conda create -n YOGA python==3.7.4
conda activate YOGA
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip3 install -r requirements.txt

# note will be added to handle Upsampling.py file 
```


### YOGA

YOGA is trained and evaluated on evaluated using COCO-2017 dataset. Below are the pretrained models and training scripts.


##### Pre-trained models

The table below gives an overview of the results of our models


| $$\textbf{Model}$$ | $$\textbf{AP}$$ | $$\textbf{AP}_\textbf{S}$$ |  $$\textbf{Params (M)}$$ | $$\textbf{FLOPs (B)}$$ |
|----	|:-:|:-:|:-:|:-:|
|  [YOGA-n](https://drive.google.com/drive/u/2/folders/1K2rKYY9p3wmA6-rN0Jx8g_6pBazyGZjO) |  32.3 | 15.2 | 1.9   | 4.9|
|  [YOGA-s](https://drive.google.com/drive/u/2/folders/1K2rKYY9p3wmA6-rN0Jx8g_6pBazyGZjO) | 40.7 | 23.0 | 7.6 |  16.6  |
|  [YOGA-m](https://drive.google.com/drive/u/2/folders/1K2rKYY9p3wmA6-rN0Jx8g_6pBazyGZjO) | 45.2|28.0|16.3|34.6
|  YOGA-l | 48.9|31.8|33.6|71.8


##### Evaluation

The script `val.py` can be used to evaluate the pre-trained models

```
  $ python val.py --weights 'pth_models/YOGA-n.pt' --img 640 --iou 0.65 --half --batch-size 1 --data data/coco.yaml
  $ python val.py --weights 'pth_models/YOGA-s.pt' --img 640 --iou 0.65 --half --batch-size 1 --data data/coco.yaml
  $ python val.py --weights 'pth_models/YOGA-m.pt' --img 640 --iou 0.65 --half --batch-size 1 --data data/coco.yaml   
```

##### Training 

The script train.py is used to train YOGA models

```
# nano model
python3 train.py --data coco.yaml --cfg ./models/YOGA-n.yaml --hyp ./data/hyps/YOGA-n.yaml --weights '' --batch-size 128 --epochs 300 --sync-bn --project YOGA --name YOGA-n --label-smoothing 0.01

# small model
python3 train.py --data coco.yaml --cfg ./models/YOGA-s.yaml --hyp ./data/hyps/YOGA-s.yaml --weights '' --batch-size 128 --epochs 300 --sync-bn --project YOGA --name YOGA-s --linear-lr

# medium model
python3 train.py --data coco.yaml --cfg ./models/YOGA-m.yaml --hyp ./data/hyps/YOGA-m.yaml --weights '' --batch-size 64 --epochs 300 --sync-bn --project YOGA --name YOGA-m

# large model

python3 train.py --data coco.yaml --cfg ./models/YOGA-l.yaml --hyp ./data/hyps/YOGA-m.yaml --weights '' --batch-size 20 --epochs 250 --sync-bn --project YOGA --name YOGA-l

```
 




