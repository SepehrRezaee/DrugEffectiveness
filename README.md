# **DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion**

## **Overview**
This repository contains code for **DISTIL**, a novel data-free trigger inversion approach designed to detect and mitigate trojan attacks in deep learning models. The project leverages diffusion models to generate suspicious trigger patterns without requiring access to clean training data, making it a powerful tool for backdoor defense.

## **Key Features**
- **Data-Free Trojan Detection:** Uses guided diffusion to generate adversarial triggers.
- **No Clean Data Required:** Avoids reliance on extensive datasets for backdoor scanning.
- **High Accuracy & Robustness:** Outperforms existing trigger inversion methods by significant margins.
- **Scalability:** Supports a variety of datasets and neural architectures.
- **Evaluation on TrojAI & BackdoorBench Datasets:** Demonstrates superior detection across different attack scenarios.

---

## **Installation**
To set up the environment, run the following:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

Ensure that **PyTorch**, **Torchvision**, and other dependencies are installed.

---

## **Usage**
### **1. Listing Available GPUs**
To check if CUDA-enabled GPUs are available:
```bash
python list_gpus.py
```

### **2. Running Trigger Inversion**
To perform trojan scanning and trigger inversion on a trained model:
```bash
python run_inversion.py --model_path <path-to-model> --dataset <dataset-name>
```

Supported datasets:
- **CIFAR-10**
- **GTSRB**
- **Tiny-ImageNet**
- **Imagenet**
- **Custom datasets (with minor modifications)**

### **3. Backdoor Mitigation**
To fine-tune a model and remove trojan triggers:
```bash
python mitigation.py --model_path <path-to-model> --dataset <dataset-name>
```

---

## **Project Structure**
```
├── notebooks/                      # Jupyter Notebooks for experiments
│   ├── DiffTrojAI_R3.ipynb
│   ├── DiffTrojAI_R4.ipynb
│   ├── DiffTrojAI_Round10.ipynb
│   ├── DiffTrojAI_R11.ipynb
├── src/                             # Source code directory
│   ├── backdoor_detection.py        # Backdoor detection using diffusion models
│   ├── mitigation.py                # Trojan mitigation strategies
│   ├── model_utils.py               # Model loading and processing utilities
│   ├── dataset_utils.py             # Dataset loading and preprocessing
│   ├── visualization.py             # Visualizations for triggers and model predictions
├── models/                          # Pre-trained and backdoored models
├── scripts/                         # Helper scripts
│   ├── list_gpus.py                 # Script to check available GPUs
│   ├── train_model.py               # Script to train a clean or trojaned model
├── Diffus_Trojan.pdf                # Paper describing the methodology
├── requirements.txt                 # Required dependencies
├── README.md                        # This file
```

---

## **Evaluation**
We evaluate **DISTIL** on the **BackdoorBench** and **TrojAI** datasets. Below are some key results:

| Dataset      | Attack Method | DISTIL Accuracy | Improvement Over Baseline |
|-------------|--------------|----------------|-------------------------|
| CIFAR-10    | BadNets      | **94.9%**      | +7.1%                    |
| CIFAR-10    | Blended      | **93.4%**      | +8.2%                    |
| TinyImageNet| InputAware   | **90.2%**      | +12.4%                   |
| GTSRB       | TrojanNN     | **86.1%**      | +9.7%                    |

DISTIL successfully reconstructs and identifies triggers while reducing false positives.

---

## **Citation**
If you use this work, please cite:

```
@article{anonymous2025distil,
  title={DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion},
  author={Anonymous},
  journal={ICCV 2025 Submission},
  year={2025}
}
```

---

## **Acknowledgments**
This work was conducted as part of the **ICCV 2025 submission**. We thank the **TrojAI** and **BackdoorBench** teams for providing datasets and benchmarks.
