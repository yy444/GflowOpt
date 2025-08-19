# GFlowOpt: Bayesian Network Structure Optimization Framework

GFlowOpt is a framework for optimizing Bayesian Network (BN) structures using GFlowNet, proxy models and hill climbing. The framework employs a three-step approach: training a GFlowNet model to learn the distribution of BN structures, using this model to generate samples to train a proxy model, and finally applying Hill Climbing optimization to find high-scoring BN structures.
[modelfig3.pdf](https://github.com/user-attachments/files/21848706/modelfig3.pdf)

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Required Dependencies

```
numpy
scipy
jax
gym>=0.22.0
pandas
dm-haiku
optax
pgmpy
networkx
tqdm
scikit-learn
jraph
```
The detailed environment configuration can be found in jsp1.yml
## Workflow

The process of finding high-scoring Bayesian Network structures consists of three main steps:

### 1. Train GFlowNet Model

First, train a GFlowNet model on your dataset:

```bash
python train.py --lr 1e-4 --lr_scheduler reduce_on_plateau --lr_patience 100 --lr_factor 0.3 --lr_min 1e-7 --batch_size 256 asia_interventional_bic
```

### 2. Train Proxy Model

Use the trained GFlowNet model to sample structures and train a proxy model:

```bash
python train_proxy_from_gflownet.py \
  train \
  --gflownet_model_path output/model.npz \
  --output_dir output/proxy \
  --num_samples 5000 \
  --num_epochs 800 \
  --batch_size 256 \
  --proxy_lr 1e-3 \
  --normalization_method standard \
  --norm_scale_factor 1 \
  --lr_patience 100 \
  --lr_factor 0.3 \
  --lr_threshold 0.0001 \
  --lr_min 1e-7 \
  asia_interventional_bic
```

### 3. Optimize Structures

Finally, optimize the structures using Hill Climbing:

```bash
python HC.py optimize \
  --gflownet_model_path output/model.npz \
  --proxy_model_path output/proxy/proxy_model.pkl \
  --output_dir output/proxy_optimize \
  asia_interventional_bic
```

## Available Datasets

The framework supports the following datasets:

- **Standard Benchmark Networks**:
  - `asia_interventional_bic`
  - `sachs_interventional_bic`
  - `child_interventional_bic`
  - `alarm_interventional_bic`
  - `hailfinder_interventional_bic`
  - `win95pts_interventional_bic`

- **Custom Networks**:
  - `sports_custom`
  - `property_custom`
  - `formed_custom`

## Important: Dataset Configuration

When changing datasets, you must update the `fixed_order` variable in `dag_gflownet/scores/bic_score.py` to match the node ordering of your target dataset. The file contains commented examples for each supported dataset.

For example, for the `asia` dataset:

```python
# In bic_score.py
fixed_order = ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
```

For the `sachs` dataset, uncomment:

```python
fixed_order = ['Akt', 'Erk', 'Jnk', 'Mek', 'P38', 'PIP2', 'PIP3', 'PKA', 'PKC', 'Plcg', 'Raf']
```

Ensure that only one `fixed_order` list is active at a time by commenting out all others.

## Example: Finding High-Scoring BN Structure for Asia Dataset

The following commands demonstrate the complete workflow for finding a high-scoring Bayesian Network structure for the Asia dataset:

```bash
# Step 1: Train GFlowNet model
python train.py --lr 1e-4 --lr_scheduler reduce_on_plateau --lr_patience 100 --lr_factor 0.3 --lr_min 1e-7 --batch_size 256 asia_interventional_bic

# Step 2: Train proxy model
python train_proxy_from_gflownet.py train --gflownet_model_path output/model.npz --output_dir output/proxy --num_samples 5000 --num_epochs 800 --batch_size 256 --proxy_lr 1e-3 --normalization_method standard --norm_scale_factor 1 --lr_patience 100 --lr_factor 0.3 --lr_threshold 0.0001 --lr_min 1e-7 asia_interventional_bic

# Step 3: Optimize structure
python HC.py optimize --gflownet_model_path output/model.npz --proxy_model_path output/proxy/proxy_model.pkl --output_dir output/proxy_optimize asia_interventional_bic
```

The optimized structures will be stored in the specified output directory.

