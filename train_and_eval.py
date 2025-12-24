#!/usr/bin/env python3

"""
Run this train_and_aval.py with the help of config/config_local.yaml to train and generate an embedding model
for trajectories.
"""
from typing import Any

import torch
import torch.optim as optim
from torch.optim.adamw import AdamW
from clearml import Task
from clearml import Dataset as Dataset_clearml

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelSummary
from core import data_module_av2
from core import custom_callbacks
from core.loss import generate_triplets
from core.utils_clearml import generate_task_name
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from core.data_processing import apply_preprocessing_steps, apply_postprocessing_steps
import torch.nn.functional as F
# Set seeds for reproducibility
seed = 2024
pl.seed_everything(seed, workers=True)

class LightningEncoder(pl.LightningModule):
    def __init__(self, model, cfg, loss_fn):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.distance_threshold = 0.8
        self.best_loss = float('inf')
        self.save_path = 'best_model.pth'
        self.loss_fn = loss_fn
        
    def training_step(self, batch, batch_idx):
        
        if self.cfg.args.mining_strategy == 'dynamic_mining':
            current_epoch = self.trainer.current_epoch
            max_epochs = self.cfg.trainer.max_epochs

            hard_mining_epochs = int(max_epochs * 0.2)
            semi_hard_mining_epochs = int(max_epochs * 0.7)

            if current_epoch < hard_mining_epochs:        
                self.mining_strategy = "hard_mining"
            elif current_epoch < semi_hard_mining_epochs:
                self.mining_strategy = "semi_hard_mining"
            else:
                self.mining_strategy = "random_mining"
        else:
            self.mining_strategy = self.cfg.args.mining_strategy

        # Generate triplets
        triplets = generate_triplets(batch, self.distance_threshold, self.cfg, self.mining_strategy)
        if not triplets:
            return None
        anchors, positives, negatives = zip(*triplets)
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        anchor_emb = self.model(anchors)
        positive_emb = self.model(positives)
        negative_emb = self.model(negatives)

        # calculate loss at the embedding layers
        loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Match training strategy during validation
        if self.cfg.args.mining_strategy == 'dynamic_mining':
            # Use same strategy progression as training
            current_epoch = self.trainer.current_epoch
            max_epochs = self.cfg.trainer.max_epochs
            hard_mining_epochs = int(max_epochs * 0.2)
            semi_hard_mining_epochs = int(max_epochs * 0.7)
            
            if current_epoch < hard_mining_epochs:        
                val_strategy = "hard_mining"
            elif current_epoch < semi_hard_mining_epochs:
                val_strategy = "semi_hard_mining"
            else:
                val_strategy = "random_mining"
        else:
            val_strategy = self.cfg.args.mining_strategy
        
        triplets = generate_triplets(batch, self.distance_threshold, self.cfg, val_strategy)

        if not triplets:
            return None
        anchors, positives, negatives = zip(*triplets)
        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        anchor_emb = self.model(anchors)
        positive_emb = self.model(positives)
        negative_emb = self.model(negatives)

        loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            if self.loss_fn.emb_norm:  # Check loss's normalization setting
                anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
                positive_emb = F.normalize(positive_emb, p=2, dim=1)
                negative_emb = F.normalize(negative_emb, p=2, dim=1)
            
            pos_distance = F.pairwise_distance(anchor_emb, positive_emb).mean()
            neg_distance = F.pairwise_distance(anchor_emb, negative_emb).mean()
            self.log('val_pos_dist', pos_distance)
            self.log('val_neg_dist', neg_distance)
            self.log('val_dist_gap', neg_distance - pos_distance)
            pos_cosine = F.cosine_similarity(anchor_emb, positive_emb).mean()
            neg_cosine = F.cosine_similarity(anchor_emb, negative_emb).mean()
            self.log('val_pos_cos', pos_cosine)
            self.log('val_neg_cos', neg_cosine)

    def test_step(self, batch, batch_idx):
        return self(batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():  # Disable gradient computation
            x, transformation_info = apply_preprocessing_steps(batch)
            print(f"Input shape: {x.shape}, Input dtype: {x.dtype}")
            print(f"Input min: {x.min()}, Input max: {x.max()}, Input mean: {x.mean()}")
            
            embeddings = self(x)
            print(f"Embeddings shape: {embeddings.shape}, Embeddings dtype: {embeddings.dtype}")
            print(f"Embeddings min: {embeddings.min()}, Embeddings max: {embeddings.max()}, Embeddings mean: {embeddings.mean()}")
            
            transformation_info.embeddings = embeddings
            apply_postprocessing_steps(transformation_info)
            return embeddings

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            dict: Configuration dictionary for optimizer and scheduler
        """
        # Initialize optimizer with model parameters
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.args["lr"],
            weight_decay=0.01  # Add weight decay for regularization
        )
        
        # Calculate total steps and warmup steps
        if self.trainer is None:
            raise RuntimeError("Trainer is not initialized. This method should be called after trainer is set.")
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.1)  # 10% warmup
        
        # Configure scheduler with warmup and cosine decay
        scheduler = {
            'scheduler': optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.args["lr"],
                total_steps=int(total_steps),
                pct_start=warmup_steps/total_steps,
                anneal_strategy='cos',
                cycle_momentum=False,
                div_factor=25.0,  # Initial lr = max_lr/25
                final_div_factor=10000.0  # Final lr = initial_lr/10000
            ),
            'interval': 'step',  # Update lr every step
            'frequency': 1,
            'name': 'learning_rate',
            'monitor': 'train_loss'  # Monitor training loss for scheduling
        }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
    # Required for the callbacks to use embeddings to generate plots.
    def forward(self, batch):
        x = batch.to(self.device)
        embeddings = self.model(x)
        # print(embeddings.shape)
        
        return embeddings


@hydra.main(version_base=None, config_path="config", config_name="config_local")
def main(cfg: DictConfig):
    # Set up logging
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    # For general clearml.TASK related
    if cfg.args["use_clearml"]:
        
        # auto populate task name if not provided
        task_name = cfg.clearml["task_name"] if cfg.clearml["task_name"] else generate_task_name()
        # Append additional string to task name if specified in config
        if "append_to_task_name" in cfg.clearml and cfg.clearml["append_to_task_name"]:
            task_name = f"{task_name}({cfg.clearml['append_to_task_name']})"
            log.info(f"Task name appended: {task_name}")
        # Get base tags
        base_tags = cfg.clearml["tags"]
        
        # Process additional tags
        additional_tags = cfg.clearml.get("additional_tags", [])
        if isinstance(additional_tags, str):
            # Split by comma if it's a string
            additional_tags = [tag.strip() for tag in additional_tags.split(',')]
        
        # Combine tags
        all_tags = base_tags + additional_tags
        
        Task.force_requirements_env_freeze(force=False, requirements_file="requirements.txt")
        task = Task.init(project_name=cfg.clearml["project_name"],
                         task_name=task_name,
                         tags=all_tags)

        task.set_base_docker(cfg.clearml["base_docker_img"],
                             docker_setup_bash_script="apt-get update && apt-get install -y libfreetype6-dev && "
                                                      "apt-cache search freetype | grep dev && "
                                                      "apt-get install -y python3-opencv && "
                                                      "conda install -y virtualenv && "
                                                      "conda install -y matplotlib",
                             docker_arguments="-e NVIDIA_DRIVER_CAPABILITIES=all")
        task.execute_remotely(cfg.clearml["queue_name"],
                              clone=False,
                              exit_process=True)
    # For clearml.dataset related
    if cfg.args["use_clearml"]:
        train_dataset = Dataset_clearml.get(dataset_project=cfg.clearml["dataset_project"],
                                            dataset_name=cfg.clearml["dataset_name"])
        preprocessed_data_path = train_dataset.get_local_copy()
        print('[INFO] Default location of dataset within clearml:: {}'.format(preprocessed_data_path))

        train_data_path = preprocessed_data_path + '/AV2/from_datasets/train_199908.pickle'
        # val_data_path = preprocessed_data_path + '/trajectories_train_dist_seed_2023.npy'
        val_data_path = preprocessed_data_path + '/AV2/from_datasets/val_24988.pickle'

    else:
        train_data_path = "data/t_set_av2/train_199908.pickle"
        val_data_path = 'data/t_set_av2/val_24988.pickle'
        

    datamodule = data_module_av2.TrajectoryDataModule(train_data_path=train_data_path,
                                                    val_data_path=val_data_path,
                                                    batch_size=cfg.args["batch_size"],
                                                    num_workers=cfg.args["num_workers"],
                                                    use_first_half_mask=cfg.args["use_first_half_mask"])
    model = hydra.utils.instantiate(cfg.model)
    loss_fn = hydra.utils.instantiate(cfg.loss)
    lightning_model = LightningEncoder(model, cfg, loss_fn)
    
    # Logging Functionality
    logger = TensorBoardLogger(**cfg.trainer.logger.tensorboard)
    lr_monitor = LearningRateMonitor(**cfg.trainer.callbacks.learning_rate_monitor)

    # Callbacks
    progress_bar = custom_callbacks.LitProgressBar()
    embedding_viz = custom_callbacks.EmbeddingVisualizationCallback()
    test_metrics_callback = custom_callbacks.TestMetricsCallback()
    checkpoint_callback = ModelCheckpoint(**cfg.trainer.callbacks.model_checkpoint)
    model_summary = ModelSummary(**cfg.trainer.callbacks.model_summary)
    
    trainer = pl.Trainer(
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        deterministic=cfg.trainer.deterministic,
        callbacks=[lr_monitor, progress_bar,
                    embedding_viz,
                    checkpoint_callback,
                    test_metrics_callback,
                    model_summary],
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )

    if cfg.args.action=='fit':
        trainer.fit(lightning_model, datamodule=datamodule)
        trainer.test(lightning_model, datamodule=datamodule)

    elif cfg.args.action=='validate':
        trainer.validate(lightning_model, datamodule=datamodule)
    
    elif cfg.args.action=='test':
        # Load checkpoint if provided
        if os.path.exists(cfg.args.ckpt_path):
            print(f"Loading model from checkpoint: {cfg.args.ckpt_path}")
            checkpoint = torch.load(
                cfg.args.ckpt_path,
                map_location='cpu',
                weights_only=True
            )
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            lightning_model.load_state_dict(state_dict)
            
        trainer.test(lightning_model, datamodule=datamodule)
    
    elif cfg.args.action=='dry_run':
        trainer = pl.Trainer(
            precision=cfg.trainer.dry_run.precision,
            limit_train_batches=cfg.trainer.dry_run.limit_train_batches,
            log_every_n_steps=cfg.trainer.dry_run.log_every_n_steps,
            limit_val_batches=cfg.trainer.dry_run.limit_val_batches,
            max_epochs=cfg.trainer.max_epochs,
            logger=logger,
            callbacks=[lr_monitor, progress_bar, test_metrics_callback, model_summary],
            check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
            enable_progress_bar=cfg.trainer.enable_progress_bar
        )
        trainer.fit(lightning_model, datamodule=datamodule)
        trainer.test(lightning_model, datamodule=datamodule)
    
    elif cfg.args.action == 'predict':
        """
        If the config is set to predict then we will load the checkpoints and will predict
        """
        # Initialize trainer with deterministic behavior
        trainer = pl.Trainer(
            precision=cfg.trainer.precision,
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            enable_progress_bar=cfg.trainer.enable_progress_bar,
            limit_predict_batches=cfg.trainer.limit_predict_batches
        )

        if not os.path.exists(cfg.args.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {cfg.args.ckpt_path}")
        
        # Load checkpoint
        checkpoint = torch.load(cfg.args.ckpt_path,
                                map_location='cpu',
                                weights_only=True)
        
        # Extract state dict
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        lightning_model.load_state_dict(state_dict)
        lightning_model.eval()
        datamodule_inference = hydra.utils.instantiate(cfg.data_module)
        # Make sure predict_step is implemented in your model
        if not hasattr(lightning_model, 'predict_step'):
            class PredictModule(type(lightning_model)):
                def predict_step(self, batch, batch_idx, dataloader_idx=0): # type: ignore
                    return self(batch)

            lightning_model.__class__ = PredictModule
        
        trainer.predict(lightning_model, datamodule=datamodule_inference)
        

if __name__ == "__main__":
    main()