# DataLabAssignment2: Precision-Recall GAN (PRGAN)

This repository implements the Precision-Recall GAN (PRGAN) approach as described in the paper [Precision and Recall for Generative Models](https://arxiv.org/abs/1904.06991). The implementation includes training a GAN with the PR-divergence, generating fake samples, and evaluating the model using k-NN Precision and Recall metrics.

## Approach Overview
The PRGAN approach uses a tunable parameter `lambda` to balance precision and recall during training. The generator and discriminator are trained using a modified loss function based on the PR-divergence. The implementation also includes a reimplementation of k-NN Precision and Recall metrics for evaluation.

---

## How to Use This Repository

### 1. Download the MNIST Dataset
To download the MNIST dataset and save real samples for evaluation:
1. Run the `train.py` or `train_monitored.py` script. The dataset will be automatically downloaded into the `data` folder.
2. The first time you run the script, 10,000 real samples will be saved in the `real_samples` folder for evaluation.

---

### 2. Train the GAN with PRGAN
You can train the GAN using the PRGAN approach with a selected `lambda` value.

#### **Option 1: Basic Training**
Use the `train.py` script for basic training:
```bash
python train.py --lambda_ <lambda_value>
```
- Replace `<lambda_value>` with the desired value of `lambda` (e.g., `2.0` or `10.0`).
- The script will save the trained generator and discriminator models in the `checkpoints` folder.

#### **Option 2: Monitored Training**
Use the `train_monitored.py` script for training with additional monitoring:
```bash
python train_monitored.py --lambda_ <lambda_value>
```
- This script will:
  - Save generated samples at each epoch in the `progress` folder.
  - Evaluate the model regularly (e.g., every 25 epochs) and log metrics such as FID, Precision, and Recall.

---

### 3. Generate Fake Samples
After training, you can generate fake samples using the checkpointed generator model:
```bash
python generate.py --batch_size <batch_size>
```
- Replace `<batch_size>` with the desired batch size (default: `4096`).
- The script will generate 10,000 fake samples and save them in the `samples` folder.

---

### 4. Evaluate the Model
Evaluate the trained model using the reimplementation of k-NN Precision and Recall metrics:
```bash
python evaluate.py --k <k_value>
```
- Replace `<k_value>` with the desired neighborhood size for k-NN (default: `3`).
- The script will compute:
  - **FID (Fréchet Inception Distance)** between real and fake samples.
  - **Precision and Recall** using the k-NN approach (based on [Kynkäänniemi et al., NeurIPS 2019](https://arxiv.org/abs/1904.06991)).

---

## Example Workflow
1. **Train the GAN**:
   ```bash
   python train_monitored.py --lambda_ 2.0 --epochs 250
   ```
2. **Generate Fake Samples**:
   ```bash
   python generate.py
   ```
3. **Evaluate the Model**:
   ```bash
   python evaluate.py --k 3
   ```

---

## Requirements
To set up the environment, use the following commands:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## References
- **PRGAN Paper**: [Precision and Recall for Generative Models](https://arxiv.org/abs/1904.06991)
- **k-NN Precision and Recall**: [Kynkäänniemi et al., NeurIPS 2019](https://arxiv.org/abs/1904.06991)
- **FID Metric**: [Heusel et al., NeurIPS 2017](https://arxiv.org/abs/1706.08500)

---
