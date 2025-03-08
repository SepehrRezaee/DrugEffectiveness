# LyAm: A Robust Optimization Algorithm for Deep Learning

## Overview
This repository contains the implementation of **LyAm**, a novel optimization algorithm inspired by Lyapunov stability principles, designed to enhance deep neural network training, particularly in noisy and non-convex environments. The project includes training scripts for CIFAR-10 and CIFAR-100 datasets using various architectures (ResNet, ViT, MobileNetV2, EfficientNet) and optimizers (LyAm, Adam, AdamW, AdaBelief, Adan, AdaGrad).

## Features
- **Novel Lyapunov-based optimization (LyAm)**: Ensures robust convergence in deep learning tasks.
- **Support for multiple architectures**: ResNet-50, ResNet-101, ViT, MobileNetV2, EfficientNet-B0.
- **Dataset augmentation**: Merges FashionMNIST into CIFAR for extended classification tasks.
- **Multiple optimizer support**: LyAm, Adam, AdamW, AdaBelief, Adan, AdaGrad.
- **Performance tracking**: Logs training and validation metrics, generates plots.
- **Reproducibility**: Implements random seed control for consistent experiments.

## File Structure
```
├── first_ex.py       # Training script for CIFAR architectures (ResNet, ViT, MobileNet, EfficientNet)
├── second_ex.py      # Training script including dataset merging and LyAm optimizer implementation
├── test.ipynb        # Jupyter notebook for interactive model training and evaluation
├── LyapunovOptimizer_ICCV2025.pdf  # Research paper draft explaining LyAm in detail
└── README.md         # Documentation (this file)
```

## Installation
To set up the environment, install the necessary dependencies:

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Usage
### Running Training Experiments
To train a model using LyAm on CIFAR-10 with ResNet-50:

```bash
python first_ex.py --dataset cifar10 --arch resnet50 --optimizer lyam --epochs 10 --lr 0.001
```

To train on CIFAR-100 using MobileNetV2 and AdaBelief:

```bash
python second_ex.py --dataset cifar100 --arch mobilenet_v2 --optimizer adabelief --epochs 20 --lr 0.0005
```

### Running Jupyter Notebook
To run the notebook interactively:

```bash
jupyter notebook test.ipynb
```

## Optimizers
The repository supports:
- **LyAm** (Lyapunov-Guided Adam)
- **Adam**
- **AdamW**
- **AdaBelief**
- **Adan**
- **AdaGrad**

## Results & Performance
Experiments in the **LyapunovOptimizer_ICCV2025.pdf** paper show that **LyAm outperforms traditional optimizers in stability and accuracy under noisy conditions**. The optimizer dynamically adjusts learning rates using Lyapunov principles, ensuring better performance in complex deep learning scenarios.

## Citation
If you use LyAm in your research, please cite our **ICCV 2025 submission**:

```
@article{LyAm2025,
  title={LyAm: Robust Non-Convex Optimization for Stable Learning in Noisy and Anomalous Environments},
  author={Anonymous},
  journal={ICCV},
  year={2025}
}
```

## License
This project is released under the **MIT License**.
