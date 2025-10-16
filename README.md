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

### Release Notes: 
Embeddings used within the paper and code will be updated soon.
