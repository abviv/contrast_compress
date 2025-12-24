import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class EmbeddingVisualizationCallback(Callback):
    def __init__(self, num_samples=10):
        super().__init__()
        self.num_samples = num_samples

    def on_train_end(self, trainer, pl_module):
        pl_module.eval()
        embeddings, trajectories = [], []

        with torch.no_grad():
            for batch in trainer.val_dataloaders:  # type: ignore
                traj = batch.to(pl_module.device)
                emb = pl_module(traj)
                embeddings.append(emb)
                trajectories.append(traj)

        embeddings = torch.cat(embeddings, dim=0)
        trajectories = torch.cat(trajectories, dim=0)
        
        # Save embeddings
        epoch = trainer.current_epoch
        # torch.save(embeddings, f'embeddings_epoch_{epoch}.pt')
        
        # Visualize evenly spaced samples instead of random ones
        total_samples = embeddings.shape[0]
        step = total_samples // self.num_samples
        sample_indices = list(range(0, total_samples, step))[:self.num_samples]
        print(f"Sample indices shape: {len(sample_indices)} and the indices are {sample_indices}")
        
        for idx in sample_indices:
            similar_indices, similarities = self._find_similar_embeddings(embeddings, idx)
            self._plot_trajectories(trajectories, idx, similar_indices, similarities, epoch)

    def _find_similar_embeddings(self, embeddings, idx, top_k=16):
        target_embedding = embeddings[idx].unsqueeze(0)
        similarities = F.cosine_similarity(target_embedding, embeddings)
        similarities[idx] = -float('inf')  # Exclude self-similarity
        top_k_indices = torch.topk(similarities, k=top_k)[1]
        return top_k_indices, similarities[top_k_indices]

    def _plot_trajectories(self, trajectories, reference_index, similar_indices, similarities, epoch):
        trajectories = trajectories.cpu().numpy()
        similarities_np = similarities.cpu().numpy()
        
        # Add safety check for normalization
        similarity_range = similarities_np.max() - similarities_np.min()
        if similarity_range > 1e-10:  # Check if range is non-zero (using small epsilon)
            norm_similarities = (similarities_np - similarities_np.min()) / similarity_range
        else:
            # If all similarities are practically equal, use equal weights
            norm_similarities = np.ones_like(similarities_np)

        plt.figure(figsize=(10, 6))
        # Plot reference trajectory in black
        ref_traj = trajectories[reference_index]
        plt.plot(ref_traj[:, 0], ref_traj[:, 1], 'r-', linewidth=2, label='Reference')

        # Plot similar trajectories with viridis colormap
        cmap = plt.get_cmap('viridis')
        for idx, norm_similarity in zip(similar_indices, norm_similarities):
            traj = trajectories[idx]
            color = cmap(norm_similarity)
            plt.plot(traj[:, 0], traj[:, 1], color=color,
                    label=f'Index: {idx} Normalized Similarity: {norm_similarity:.2f}')

        plt.title(f"Trajectory Similarities (Sample {reference_index})")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'trajectory_viz_epoch_{epoch}_sample_{reference_index}.png', bbox_inches='tight')
        plt.close()


class TestMetricsCallback(Callback):
    """
    This callback is used to compute the average displacement error (ADE) for the test set.
    In this eval mode, Query trajectories come from the test set and the "database" come from the train set.
    We perform an equivalent comparison of how a new trajectory compares against the training set.
    """
    def __init__(self):
        super().__init__()
        self.train_trajectories = None
        self.train_embeddings = None

    def on_test_end(self, trainer, pl_module):
        # Compute embeddings for test trajectories
        if self.train_trajectories is None:
            print("Warning: No training data available. Computing embeddings from train dataloader...")
            # If no stored train data (e.g., when running test only), compute it now
            train_trajectories, train_embeddings = [], []
            
            # Get train dataloader using a more robust approach
            train_loader = None
            if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'train_dataloader'):
                # Best approach - use the datamodule
                train_loader = trainer.datamodule.train_dataloader()
            elif hasattr(trainer, 'train_dataloaders') and trainer.train_dataloaders is not None:
                # Fallback to trainer's dataloaders (newer Lightning versions)
                train_loader = trainer.train_dataloaders
            elif hasattr(trainer, '_data_connector') and hasattr(trainer._data_connector, '_train_dataloader_source'):
                # Another fallback for some Lightning versions
                train_loader = trainer._data_connector._train_dataloader_source.dataloader()
            
            if train_loader is None:
                raise ValueError("Cannot access train dataloader. Make sure to run fit() before test() "
                                "or provide training data to the callback directly.")
            
            # Compute embeddings
            with torch.no_grad():
                for batch in train_loader:
                    batch = batch.to(pl_module.device)
                    embeddings = pl_module(batch)
                    train_trajectories.append(batch)
                    train_embeddings.append(embeddings)
                
            self.train_trajectories = torch.cat(train_trajectories, dim=0)
            self.train_embeddings = torch.cat(train_embeddings, dim=0)

        # Rest of the test code remains the same
        print(f"==============================> Entering test dataloader")
        test_dataloader = trainer.test_dataloaders
        test_trajectories, test_embeddings = [], []
        
        with torch.no_grad():
            for batch in test_dataloader: # type: ignore
                batch = batch.to(pl_module.device)
                embeddings = pl_module(batch)
                test_trajectories.append(batch)
                test_embeddings.append(embeddings)
        
        test_trajectories = torch.cat(test_trajectories, dim=0)
        test_embeddings = torch.cat(test_embeddings, dim=0)

        # save things to disk
        torch.save({
            'test_trajectories': test_trajectories,
            'test_embeddings': test_embeddings,
            'train_trajectories': self.train_trajectories,
            'train_embeddings': self.train_embeddings
        }, f'embeddings_trajectory_{trainer.current_epoch}.pt')

        print(f"==============================>Test trajectories shape: {test_trajectories.shape}")
        print(f"==============================>Test embeddings shape: {test_embeddings.shape}")
        print(f"==============================>Train trajectories shape: {self.train_trajectories.shape}")
        print(f"==============================>Train embeddings shape: {self.train_embeddings.shape}") # type: ignore
        
        # Initialize metrics
        total_avg_fde = 0
        total_min_fde = 0
        total_avg_ade = 0
        total_min_ade = 0
        num_queries = 0
        
        for idx in range(len(test_trajectories)):
            query_traj = test_trajectories[idx]
            query_emb = test_embeddings[idx].unsqueeze(0)
            similarities = F.cosine_similarity(query_emb, self.train_embeddings, dim=1) # type: ignore
            top_k_indices = torch.topk(similarities, k=6)[1]
            similar_trajs = self.train_trajectories[top_k_indices]
            errors = torch.norm(query_traj.unsqueeze(0) - similar_trajs, dim=2)
            ade = errors.mean(dim=1) # [:, K]
            fde = errors[:, -1] 
            
            total_min_ade += ade.min().item() # pick the minimum error
            total_avg_ade += ade.mean().item() # apply mean
            total_min_fde += fde.min().item() # pick the minimum error
            total_avg_fde += fde.mean().item() # apply mean
            
            num_queries += 1
        
        avg_ade = total_avg_ade / num_queries
        avg_min_ade = total_min_ade / num_queries
        avg_fde = total_avg_fde / num_queries
        avg_min_fde = total_min_fde / num_queries

        print(f"==============================>Test Average Displacement Error (ADE): {avg_ade:.4f}")
        print(f"==============================>Test Final Displacement Error (FDE): {avg_fde:.4f}")
        print(f"==============================>Test Minimum Displacement Error (ADE): {avg_min_ade:.4f}")
        print(f"==============================>Test Minimum Final Displacement Error (FDE): {avg_min_fde:.4f}")
        
        trainer.logger.experiment.add_scalar("test_ade", avg_ade, trainer.current_epoch) # type: ignore
        trainer.logger.experiment.add_scalar("test_fde", avg_fde, trainer.current_epoch) # type: ignore
        trainer.logger.experiment.add_scalar("test_min_ade", avg_min_ade, trainer.current_epoch) # type: ignore
        trainer.logger.experiment.add_scalar("test_min_fde", avg_min_fde, trainer.current_epoch) # type: ignore

# Same example as that of pl
class LitProgressBar(pl.callbacks.progress.TQDMProgressBar): # type: ignore
    def __init__(self):
        super().__init__()
        self.enable = True

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if trainer.global_step % 100 == 0:
            # Check if outputs is not None and contains 'loss'
            if outputs is not None and isinstance(outputs, dict) and 'loss' in outputs:
                print(f"Step {trainer.global_step}, Loss: {outputs['loss'].item():.4f}")
            else:
                print(f"Step {trainer.global_step}, Loss: N/A")