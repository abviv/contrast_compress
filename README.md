# Contrast &amp; Compress: Learning Lightweight Embeddings for Short Trajectories

Abstract:The ability to retrieve semantically and direction-
ally similar short-range trajectories with both accuracy and
efficiency is foundational for downstream applications such
as motion forecasting and autonomous navigation. However,
prevailing approaches often depend on computationally in-
tensive heuristics or latent anchor representations that lack
interpretability and controllability. In this work, we propose a
novel framework for learning fixed-dimensional embeddings for
short trajectories by leveraging a Transformer encoder trained
with a contrastive triplet loss that emphasize the importance of
discriminative feature spaces for trajectory data. We analyze
the influence of Cosine and FFT-based similarity metrics within
the contrastive learning paradigm, with a focus on captur-
ing the nuanced directional intent that characterizes short-
term maneuvers. Our empirical evaluation on the Argoverse
2 dataset demonstrates that embeddings shaped by Cosine
similarity objectives yield superior clustering of trajectories
by both semantic and directional attributes, outperforming
FFT-based baselines in retrieval tasks. Notably, we show that
compact Transformer architectures, even with low-dimensional
embeddings (e.g., 16 dimensions, but qualitatively down to 4),
achieve a compelling balance between retrieval performance
(minADE, minFDE) and computational overhead, aligning with
the growing demand for scalable and interpretable motion
priors in real-time systems. The resulting embeddings provide a
compact, semantically meaningful, and efficient representation
of trajectory data, offering a robust alternative to heuristic
similarity measures and paving the way for more transparent
and controllable motion forecasting pipelines.

Link to our paper: https://arxiv.org/pdf/2506.02571

## Setup Instructions for Env

There are 2 options to create and use this. If you're not planning to use clearml then build a miniforge through conda-forge channels using the following command. Its tested only for the python 3.8 version but works until python 3.10.

```bash
$ conda env create -f env.yml
```

If you're using clearml then I would suggest you to build the dockerfile at the root of this repo and use it as a devcontainer if you're using vscode or as a local docker interpreter for Pycharm. In this way you can easily push to clearml server for remote training.

```bash
$ docker build -t env_traj_embeddings:latest .
```

## Before Full Training

Before training on server there is a dry_run option which is geared towards using a small sample of the train and test dataset and runs it for defined epochs. Use this option to test any new functionality before running a full training on the server.

```bash
python train_and_eval.py args.action=dry_run args.batch_size=32 args.epochs=10
```

## Training

### Local or Server

By default the `args.use_clearml=False` which means you dont have to use any of the ML pipelines associated with clearml.

```python
python train_and_eval.py --config-name config_march_pudding_transformer 'clearml.additional_tags=[emb_dim_16, running_inference]' model.embedding_dim=16 model.num_heads=4 model.num_layers=1 clearml.append_to_task_name=march-pudding-emb-16-head-4-layer-1
```

If you're running the python interpreter inside the container then you should run it with the user privilege. Add the following option while starting your container.

### Optional: With Docker Option

```bash
$ docker run -it --rm --gpus all -u 1000:1000 --entrypoint= -v /home/abishek/contrast_compress:/opt/project 
```

### Generate Embeddings with Trained Models

Look into the file `core/inference/faiss_eval_search.py` for more help on how to put the model into inference mode and generate embeddings.
There are multiple modes including `search` and `generate`.

```python
python core/inference/faiss_eval_search.py --model-dir /home/abishek/contrast_compress/data/trained_models/spice-slush-cosine/ generate --data-path /home/abishek/contrast_compress/data/t_set_womd/train_500k_womd.pkl
```

`t_set_womd` raw trajectories will be updated soon. You can change this to av1/av2 for more analysis.

## Dataset

1. For training we used the processed future state of the AV2 dataset which you can access through the following [link](https://drive.google.com/drive/folders/1qZI_jOsqy6jV6puMsXTViULS0JdYGrH3?usp=sharing).

2. Download the files from the link and organize the files such that the contents can be reached  via the path: `/home/abhishek/contrast_compress/data/t_set_av2`

## Pre-Trained Embeddings

You can find the pre-trained embeddings from the above link for WOMD dataset. TODO: Will upload the AV2 soon.

## Release Notes

If you find our work useful, please consider citing our paper through the following BibTex

```latex
@misc{vivekanandan2025contrastcompresslearning,
      title={Contrast & Compress: Learning Lightweight Embeddings for Short Trajectories}, 
      author={Abhishek Vivekanandan and Christian Hubschneider and J. Marius Zöllner},
      year={2025},
      eprint={2506.02571},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.02571}, 
}
```