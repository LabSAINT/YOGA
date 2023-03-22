** Code and scripts to be uploaded soon. **

[Link to paper](https://www.sciencedirect.com/science/article/pii/S0031320323001516)

[Full text](https://authors.elsevier.com/a/1ghEz77nKkWoi) (accessible until April 22, 2023 only)

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
