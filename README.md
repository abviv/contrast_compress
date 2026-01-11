# Contrast & Compress

**Learning Lightweight Embeddings for Short Trajectories**

[![arXiv](https://img.shields.io/badge/arXiv-2506.02571-b31b1b.svg)](https://arxiv.org/abs/2506.02571)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Compact, semantically meaningful trajectory embeddings for real-time motion forecasting and autonomous navigation.

---

## Overview

Retrieving similar trajectories efficiently is critical for motion forecasting and autonomous driving. Traditional approaches rely on computationally expensive heuristics or opaque latent representations. **Contrast & Compress** offers a better way:

- **Transformer encoder** trained with contrastive triplet loss
- **Ultra-compact embeddings** (as low as 4-16 dimensions) with competitive performance
- **Real-time capable** for deployment in autonomous systems
- **Interpretable** similarity-based retrieval

### Key Results

| Embedding Dim | minADE | minFDE | Comments        |
|---------------|--------|--------|-----------------|
| 256           | Best   | Best   | Baseline        |
| 128           | Good   | Good   | Available for Inference  |
| 16            | ~Best  | ~Best  | Request for Access      |
| 4             | Good   | Good   | Request for Access      |

---

## Architecture

```
                                    Contrast & Compress Pipeline
    
    Raw Trajectory          Transformer Encoder              Embedding Space
    ┌─────────────┐        ┌───────────────────┐           ┌─────────────────┐
    │ (x,y) seq   │   -->  │  Positional Enc   │   -->     │   Compact       │
    │ T timesteps │        │  Self-Attention   │           │   Embedding     │
    │             │        │  Attentive Pool   │           │   (4-256 dim)   │
    └─────────────┘        └───────────────────┘           └─────────────────┘
                                    │
                                    v
                           ┌───────────────────┐
                           │   Triplet Loss    │
                           │   + Mining        │
                           └───────────────────┘
```

**Training Strategy**: Dynamic triplet mining that transitions from hard -> semi-hard -> random mining across epochs for stable convergence.

---

## Quick Start

### Installation

**Option 1: Conda (Recommended)**

```bash
conda env create -f env.yml
conda activate traj_embeddings
```

**Option 2: Docker (for ClearML integration)**

```bash
docker build -t contrast_compress:latest .
```

### Download Data

1. Download the processed AV2 dataset from [Google Drive](https://drive.google.com/drive/folders/1qZI_jOsqy6jV6puMsXTViULS0JdYGrH3?usp=sharing)
2. Organize files to: `data/t_set_av2/`

```
data/
├── t_set_av2/
│   ├── train_*.pkl
│   └── test_*.pkl
├── t_set_womd/          # Optional: Waymo dataset
└── trained_models/      # Pre-trained checkpoints
```

---

## Usage

### Dry Run (Validate Setup)

Test your setup with a small sample before full training:

```bash
python train_and_eval.py args.action=dry_run args.batch_size=32 args.epochs=10
```

### Training

**Basic training:**

```bash
python train_and_eval.py --config-name config_march_pudding_transformer
```

**Custom configuration:**

```bash
python train_and_eval.py \
  --config-name config_march_pudding_transformer \
  model.embedding_dim=16 \
  model.num_heads=4 \
  model.num_layers=1
```

**With experiment tracking (ClearML):**

```bash
python train_and_eval.py \
  --config-name config_march_pudding_transformer \
  args.use_clearml=True \
  'clearml.additional_tags=[emb_dim_16]' \
  clearml.append_to_task_name=my-experiment
```

### Inference & Embedding Generation

**Generate embeddings for a dataset:**

```bash
python core/inference/faiss_eval_search.py \
  --model-dir data/trained_models/spice-slush-cosine/ \
  generate \
  --data-path data/t_set_av2/test.pkl
```

**Search for similar trajectories:**

```bash
python core/inference/faiss_eval_search.py \
  --model-dir data/trained_models/spice-slush-cosine/ \
  search
```

---

## Configuration

Configurations are managed via [Hydra](https://hydra.cc/). Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.embedding_dim` | Output embedding dimension | 16 |
| `model.num_heads` | Transformer attention heads | 4 |
| `model.num_layers` | Transformer encoder layers | 1 |
| `model.hidden_dim` | Hidden layer dimension | 512 |
| `args.batch_size` | Training batch size | 512 |
| `args.epochs` | Number of training epochs | 100 |
| `loss.similarity_fn` | Similarity function (`cosine`/`fft`) | cosine |

See `config/` directory for all available configurations.

---

## Project Structure

```
contrast_compress/
├── train_and_eval.py          # Main training entry point
├── config/                    # Hydra configuration files
│   ├── config_march_pudding_base.yaml
│   └── config_march_pudding_transformer.yaml
├── core/
│   ├── models/
│   │   └── transformer_encoders.py   # Transformer architecture
│   ├── loss.py                       # Triplet loss & mining strategies
│   ├── data_module_av2.py            # Data loading & preprocessing
│   ├── custom_callbacks.py           # Training callbacks
│   └── inference/
│       └── faiss_eval_search.py      # FAISS-based retrieval
├── data/                      # Datasets & trained models
└── outputs/                   # Training logs & checkpoints
```

---

## Pre-trained Models

Pre-trained embeddings are available for:

- [x] Argoverse 2 (AV2)
- [x] Waymo Open Motion Dataset (WOMD)

Download from the [Google Drive link](https://drive.google.com/drive/folders/1qZI_jOsqy6jV6puMsXTViULS0JdYGrH3?usp=sharing).

---

## Docker Usage

For containerized training with GPU support:

```bash
# Build the image
docker build -t contrast_compress:latest .

# Run with GPU access
docker run -it --rm \
  --gpus all \
  -u $(id -u):$(id -g) \
  -v $(pwd):/opt/project \
  contrast_compress:latest \
  python train_and_eval.py args.action=dry_run
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{vivekanandan2025contrastcompress,
  title={Contrast & Compress: Learning Lightweight Embeddings for Short Trajectories}, 
  author={Abhishek Vivekanandan and Christian Hubschneider and J. Marius Z{\"o}llner},
  year={2025},
  eprint={2506.02571},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.02571}, 
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Contrast & Compress</b> - Efficient trajectory embeddings for the real world.
</p>
