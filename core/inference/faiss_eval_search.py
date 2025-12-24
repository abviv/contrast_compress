import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import cm for color maps
import random
import torch
from pathlib import Path
import os
import sys
import argparse
from scipy.interpolate import interp1d
from tqdm import tqdm
from omegaconf import OmegaConf
import logging # Add logging import

# Setup logger
logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    logger.error("faiss not found. Please install with 'pip install faiss-gpu' or 'pip install faiss-cpu'") # Use logger.error
    sys.exit(1)

# Define paths
DEFAULT_AV1_DATA_PATH = Path("/home/abishek/git_area/xgen_trajectory_embeddings/data/t_set_av1/tset_val_40k_av1.pickle")
DEFAULT_ROOT_DIR = Path("/media/abishek/just_envs/Paper_4_Trajectory_embeddings_artifacts/trained_embs")
DEFAULT_MODEL_DIR = Path("/home/abishek/git_area/xgen_trajectory_embeddings/data/trained_models/lucky-blame")

class ModelRepository:
    """
    Central repository to manage model references, embeddings, and associated trajectories.
    """
    def __init__(self, root_dir=DEFAULT_ROOT_DIR, model_dir=DEFAULT_MODEL_DIR):
        self.root_dir = Path(root_dir)
        self.model_dir = Path(model_dir)
        # is deprecated
        self.trained_models = {
            "10-misery-excess": self.model_dir / "embeddings_trajectory_10-misery-excess.pt",
            "10-pyramid-catalog": self.model_dir / "embeddings_trajectory_10-pyramid-catalog.pt",
            "10-afford-there": self.model_dir / "embeddings_trajectory_10-afford-there.pt",
            "20-garage-crack": self.model_dir / "embeddings_trajectory_20-garage-crack.pt",
            "30-jelly-lift": self.model_dir / "embeddings_trajectory_30-jelly-lift.pt",
            "40-omni-gospel": self.model_dir / "embeddings_trajectory_40-omni-gospel.pt",
            "10-unknown": self.model_dir / "embeddings_trajectory_10.pt",
            "develop-decline-100": self.model_dir / "embeddings_trajectory_develop-decline-100.pt",
            "100-thinking-boston": self.model_dir / "embeddings_trajectory_100-thinking-boston.pt",
            "100-march-pudding-8": self.model_dir / "embeddings_trajectory_100-march-pudding-emb-8.pt", #cosine
            "100-march-pudding-16": self.model_dir / "embeddings_trajectory_100-march-pudding-emb-16.pt", #cosine
            "100-march-pudding-32": self.model_dir / "embeddings_trajectory_100-march-pudding-32.pt", #cosine
            "100-march-pudding-64": self.model_dir / "embeddings_trajectory_100-march-pudding-64.pt", #cosine
            "100-march-pudding-256": self.model_dir / "embeddings_trajectory_100-march-pudding-256.pt", #cosine
            "100-upon-gasp-8": self.model_dir / "embeddings_trajectory_100-upon-gasp-8.pt", #fft
            "100-upon-gasp-16": self.model_dir / "embeddings_trajectory_100-upon-gasp-16.pt", #fft
            "100-upon-gasp-32": self.model_dir / "embeddings_trajectory_100-upon-gasp-32.pt", #fft
            "100-upon-gasp-64": self.model_dir / "embeddings_trajectory_100-upon-gasp-64.pt", #fft
            "100-upon-gasp-128": self.model_dir / "embeddings_trajectory_100-upon-gasp-128.pt", #fft
            "100-receive-carbon": self.model_dir / "embeddings_trajectory_100-receive-carbon.pt", #cosine
            "100-school-eight": self.model_dir / "embeddings_trajectory_100-school-eight.pt", #cosine
            "100-camera-subway": self.model_dir / "embeddings_trajectory_100-camera-subway.pt", #fft
            # "100-sudden-derive": self.model_dir / "embeddings_trajectory_100-sudden-derive.pt", #fft
            "100-clone-pool-more": self.model_dir / "embeddings_trajectory_100-clone-pool-more.pt", 
            "50-hour-cricket": self.model_dir / "embeddings_trajectory_50-hour-cricket.pt",
            "50-ugly-melody": self.model_dir / "embeddings_trajectory_50-ugly-melody.pt",
        }
        
        # Find config file (*.yaml) in the model directory
        self.config_files = list(self.model_dir.glob("*.yaml"))
        if not self.config_files:
            logger.warning(f"No config file (*.yaml) found in {self.model_dir}") # Use logger.warning
            self.config_path = None
        else:
            self.config_path = self.config_files[0]
            
        # Find checkpoint file (*.ckpt) in the model directory
        self.checkpoint_files = list(self.model_dir.glob("*.ckpt"))
        if not self.checkpoint_files:
            logger.warning(f"No checkpoint file (*.ckpt) found in {self.model_dir}") # Use logger.warning
            self.trained_model_path = None
        else:
            # Sort by modification time to get the latest checkpoint
            self.checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            self.trained_model_path = self.checkpoint_files[0]
    
    def list_available_models(self):
        """List all available trained models in the repository"""
        logger.info("\nAvailable trained models:") # Use logger.info
        for model_name, path in self.trained_models.items():
            status = "✓" if path.exists() else "✗"
            logger.info(f"- {model_name} [{status}]: {path}") # Use logger.info
    
    def get_model_embeddings_path(self, model_name):
        """Get path to the embeddings file for a specific model.
        
        Checks the predefined `self.trained_models` first. If not found,
        attempts to find a file named `embeddings_trajectory_{model_name}.pt` 
        or `{model_name}.pt` directly within `self.model_dir`.
        """
        # Check predefined dictionary first
        if model_name in self.trained_models:
            model_path = self.trained_models[model_name]
            if model_path.exists():
                return model_path
            else:
                logger.warning(f"Model '{model_name}' found in dictionary but file missing: {model_path}") # Use logger.warning

        # If not in dict or file missing, check model_dir directly
        potential_path1 = self.model_dir / f"embeddings_trajectory_{model_name}.pt"
        if potential_path1.exists():
            logger.info(f"Found model '{model_name}' directly in model directory: {potential_path1}") # Use logger.info
            return potential_path1
            
        potential_path2 = self.model_dir / f"{model_name}.pt"
        if potential_path2.exists():
            logger.info(f"Found model '{model_name}' directly in model directory: {potential_path2}") # Use logger.info
            return potential_path2

        # List available files for better error message
        available_pts = list(self.model_dir.glob("*.pt"))
        available_files_str = "\\n  ".join([f.name for f in available_pts]) if available_pts else "None"
        
        raise FileNotFoundError(
            f"Model '{model_name}' not found in predefined list or directly in model directory '{self.model_dir}'. "
            f"Available .pt files in model directory:\\n  {available_files_str}"
        )
    
    def get_evaluation_results_path(self, model_name, suffix=""):
        """Get path where evaluation results should be stored"""
        results_dir = self.model_dir / "evaluation_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine base filename
        if suffix:
            base_filename = f"{model_name}_{suffix}.pickle"
        else:
            base_filename = f"{model_name}.pickle"
        
        # Check if file already exists, if so add a running number
        file_path = results_dir / base_filename
        counter = 1
        while file_path.exists():
            # Extract name without extension
            name_part = base_filename.rsplit('.', 1)[0]
            # Create new filename with counter
            new_filename = f"{name_part}_{counter}.pickle"
            file_path = results_dir / new_filename
            counter += 1
            
        return file_path
    
    def get_plots_dir(self, model_name, suffix=""):
        """Get directory where visualization plots should be stored"""
        plots_dir = self.model_dir / "visualization_plots"
        if suffix:
            plots_dir = plots_dir / f"{model_name}_{suffix}"
        else:
            plots_dir = plots_dir / model_name
            
        plots_dir.mkdir(parents=True, exist_ok=True)
        return plots_dir

    def get_default_model_name(self):
        """Get the name of the model based on the model directory"""
        # Look for .pt files in the model directory
        pt_files = list(self.model_dir.glob("*.pt"))
        if pt_files:
            # Return the name extracted from the first pt file
            model_name = pt_files[0].stem.replace("embeddings_trajectory_", "")
            return model_name
        
        # If no pt files in model_dir, look for evaluation results
        eval_results_dir = self.model_dir / "evaluation_results"
        if eval_results_dir.exists():
            result_files = list(eval_results_dir.glob("*.pickle"))
            if result_files:
                # Extract model name from the first pickle file
                model_name = result_files[0].stem.split("_eval")[0]
                return model_name
        
        # Default model name if nothing found
        return "100-march-pudding-64"

    def get_latest_evaluation_result(self):
        """Get the path to the latest evaluation result file"""
        eval_results_dir = self.model_dir / "evaluation_results"
        if not eval_results_dir.exists():
            return None
            
        result_files = list(eval_results_dir.glob("*.pickle"))
        if not result_files:
            return None
            
        # Sort by modification time to get the latest result
        result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return result_files[0]

def load_av1_dataset(data_path=DEFAULT_AV1_DATA_PATH):
    """
    Load the AV1 dataset from the specified path.
    
    Args:
        data_path: Path to the AV1 dataset pickle file
        
    Returns:
        av1_data: Dictionary containing trajectories
    """
    try:
        with open(data_path, 'rb') as f:
            av1_data = pickle.load(f)
        logger.info(f"Successfully loaded AV1 dataset from {data_path}") # Use logger.info
        
        # Check the structure of the loaded data
        if isinstance(av1_data, list):
            logger.info(f"Dataset contains {len(av1_data)} trajectories") # Use logger.info
            
            # Create a dictionary structure to match the rest of the code
            trajectories = av1_data
            av1_data = {'trajectories': trajectories}
            
            # Print the shape of the first few trajectories
            for i in range(min(3, len(av1_data['trajectories']))):
                logger.info(f"Trajectory {i} shape: {av1_data['trajectories'][i].shape}") # Use logger.info
        else:
            logger.info(f"Dataset is not a list, but a {type(av1_data)}") # Use logger.info
            logger.info(f"Keys in the dataset: {av1_data.keys() if isinstance(av1_data, dict) else 'N/A'}") # Use logger.info
    except Exception as e:
        logger.error(f"Error loading AV1 dataset: {e}") # Use logger.error
        av1_data = {'trajectories': []}
    
    return av1_data

def plot_random_trajectories(trajectories, num_trajectories=1000, figsize=(12, 10), alpha=0.3, title="Random Trajectories"):
    """
    Plot a random sample of trajectories from the dataset, rotated to point eastwards using a rotation matrix.

    Args:
        trajectories: List of trajectory arrays
        num_trajectories: Number of random trajectories to plot
        figsize: Size of the figure (width, height)
        alpha: Transparency of trajectory lines
        title: Title for the plot
    """
    # Sample random trajectories if we have more than requested
    if len(trajectories) > num_trajectories:
        indices = np.random.choice(len(trajectories), size=num_trajectories, replace=False)
        sample_trajectories = [trajectories[i] for i in indices]
    else:
        sample_trajectories = trajectories
        logger.warning(f"Only {len(trajectories)} trajectories available, plotting all of them.") # Use logger.warning

    # Create the plot
    plt.figure(figsize=figsize)

    # Define the rotation angle (-90 degrees in radians)
    theta = -np.pi / 2
    # Construct the rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Plot each trajectory
    for traj in sample_trajectories:
        # Ensure traj is a numpy array
        traj_np = np.array(traj)

        # Center the trajectory at its starting point (translate to origin)
        start_point = traj_np[0]
        centered_traj = traj_np - start_point

        # Apply the rotation matrix to each point in the centered trajectory
        # We use np.dot(centered_traj, rotation_matrix.T) for batch matrix multiplication
        # where centered_traj is (N, 2) and rotation_matrix.T is (2, 2)
        rotated_traj = np.dot(centered_traj, rotation_matrix.T)

        # Extract transformed x and y coordinates
        x = rotated_traj[:, 0]
        y = rotated_traj[:, 1]
        plt.plot(x, y, alpha=alpha)
    
    plt.title(title)
    plt.xlabel("X coordinate (East)") 
    plt.ylabel("Y coordinate (North)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')

    return plt.gcf()

def plot_query_and_nearest_trajectories(query_trajectory, nearest_trajectories, figsize=(12, 10), title="Query and Nearest Trajectories"):
    """
    Plot a query trajectory and its nearest neighbors with different colors and styles.
    
    Args:
        query_trajectory: The query trajectory array (already transformed)
        nearest_trajectories: List of nearest trajectory arrays (already transformed)
        figsize: Size of the figure (width, height)
        title: Title for the plot
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot query trajectory in red and thicker
    ax.plot(query_trajectory[:, 0], query_trajectory[:, 1], color='red', linewidth=3, label='Query')
    
    # Create colormap for nearest trajectories
    colormap = cm.get_cmap('viridis', len(nearest_trajectories))
    
    # Plot each nearest trajectory with a different color
    for i, traj in enumerate(nearest_trajectories):
        # Plot with faded color from colormap
        ax.plot(traj[:, 0], traj[:, 1], color=colormap(i), alpha=0.7, 
                linewidth=1.5, label=f'Match {i+1}' if i < 5 else "")
    
    ax.set_title(title)
    ax.set_xlabel("X coordinate (East)")
    ax.set_ylabel("Y coordinate (North)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axis('equal')
    
    # Add legend for first 5 matches only to avoid clutter
    ax.legend(loc='best')
    
    return fig

def plot_random_trajectories_from_av1_dataset(av1_data):
    # Plot 1000 random trajectories from the AV1 dataset
    if len(av1_data['trajectories']) > 0:
        fig = plot_random_trajectories(
            av1_data['trajectories'], 
            num_trajectories=1000, 
            title="1000 Random Trajectories from AV1 Dataset"
            )
        plt.show()
    else:                       
        logger.info("No trajectories available to plot.") # Use logger.info

def load_model_from_config(config_path, trained_model_path):
    """
    Load the transformer model from config and checkpoint.

    Args:
        config_path: Path to the config file
        trained_model_path: Path to the trained model checkpoint

    Returns:
        model: The loaded model based on the config
    """
    # Add the root directory to the path to import core modules
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Import the model definition
    from core.models.transformer_encoders import SelfAttentionTrajectoryEncoder
    
    # Load the config
    cfg = OmegaConf.load(config_path)
    
    # Initialize the model from config
    model = SelfAttentionTrajectoryEncoder(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        embedding_dim=cfg.model.embedding_dim,
        dropout=cfg.model.dropout,
        norm_first=cfg.model.norm_first,
        use_input_dropout=cfg.model.use_input_dropout,
        input_dropout_rate=cfg.model.input_dropout_rate,
        trajectory_pooling=cfg.model.trajectory_pooling
    )
    
    # Load the checkpoint
    logger.info(f"Loading model from checkpoint: {trained_model_path}") # Use logger.info
    checkpoint = torch.load(
        trained_model_path,
        map_location='cpu'
    )
    
    # Extract the state dict
    if 'state_dict' in checkpoint:
        # Handle Lightning checkpoint format
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() 
                    if k.startswith('model.')}
    else:
        # Handle direct model state dict
        state_dict = checkpoint
    
    # Load state dict into model
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def generate_and_save_embeddings(model, av1_data, repo, model_name, semi_percentage=100, interpolate=False):
    """
    Generate embeddings for each trajectory and save the results to disk.
    Trajectories are interpolated to shape (60,2) before being passed to the model.
    
    Args:
        model: The loaded transformer model
        av1_data: Dictionary containing trajectories
        repo: ModelRepository instance for managing file paths
        model_name: Name of the model to use for saving results
        semi_percentage: Percentage of data to process (1-100)
        interpolate: Whether to interpolate the trajectories to shape (60,2)

    Returns:
        results: Dictionary containing original trajectories, transformed trajectories, and embeddings
    """
    # Determine the suffix for the output file
    base_suffix = f"eval_{semi_percentage}p"
    if interpolate:
        # Assuming interpolation target is fixed at (60, 2) as per interpolate_trajectory helper
        final_suffix = f"{base_suffix}_interp_60x2"
    else:
        final_suffix = base_suffix
        
    # Get output path from repository using the determined suffix
    output_file = repo.get_evaluation_results_path(model_name, final_suffix)
    
    # Make sure model is in evaluation mode
    model.eval()
    
    # Get trajectories and calculate how many to process
    trajectories = av1_data['train_trajectories']
    num_trajectories = len(trajectories)
    num_to_process = int(num_trajectories * (semi_percentage / 100))
    
    # Randomly select trajectories if using semi-supervised mode
    if semi_percentage < 100:
        indices = np.random.choice(num_trajectories, size=num_to_process, replace=False)
        trajectories = [trajectories[i] for i in indices]
        logger.info(f"Semi-supervised mode: Processing {num_to_process} trajectories ({semi_percentage}% of {num_trajectories})") # Use logger.info
    
    # Define the rotation angle (-90 degrees in radians) - same as in visualization
    theta = -np.pi / 2
    # Construct the rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Process in batches to avoid memory issues
    batch_size = 256
    embeddings = []
    # Store original and transformed trajectories
    original_trajectories = []
    transformed_trajectories = []
    
    logger.info(f"Generating embeddings for {len(trajectories)} trajectories...") # Use logger.info
    
    def interpolate_trajectory(traj, target_length=60):
        """Helper function to interpolate a trajectory to target length"""
        # Create time points for interpolation
        t_original = np.linspace(0, 1, len(traj))
        t_target = np.linspace(0, 1, target_length)
        
        # Interpolate x and y coordinates separately
        interpolator_x = interp1d(t_original, traj[:, 0], kind='linear')
        interpolator_y = interp1d(t_original, traj[:, 1], kind='linear')
        
        # Generate interpolated trajectory
        x_interp = interpolator_x(t_target)
        y_interp = interpolator_y(t_target)
        
        return np.column_stack((x_interp, y_interp))
    
    with torch.no_grad():
        for i in range(0, len(trajectories), batch_size):
            batch = trajectories[i:i+batch_size]
            transformed_batch = []
            
            # Apply transformations to each trajectory (center, rotate, and interpolate)
            for traj in batch:
                # Store original trajectory
                original_trajectories.append(traj)
                
                # Ensure trajectory is a numpy array
                traj_np = np.array(traj)
                
                # Center the trajectory at its starting point (translate to origin)
                start_point = traj_np[0]
                centered_traj = traj_np - start_point
                
                # Apply the rotation matrix to each point in the centered trajectory
                rotated_traj = np.dot(centered_traj, rotation_matrix.T)
                
                # Interpolate if requested
                if interpolate: 
                    # Interpolate the rotated trajectory to shape (60,2)
                    interpolated_traj = interpolate_trajectory(rotated_traj, target_length=60)
                else:
                    interpolated_traj = rotated_traj # Use rotated trajectory directly
                
                # Store the transformed and interpolated trajectory
                transformed_batch.append(interpolated_traj)
                transformed_trajectories.append(interpolated_traj)
            
            # Convert batch to tensor
            batch_tensor = torch.tensor(np.stack(transformed_batch, axis=0), dtype=torch.float32)
            
            # Forward pass
            batch_embeddings = model(batch_tensor)
            
            # Store embeddings
            embeddings.append(batch_embeddings.cpu().numpy())
            
            if (i // batch_size) % 10 == 0:
                logger.info(f"Processed {i}/{len(trajectories)} trajectories") # Use logger.info
                if i == 0:  # Print shape information for the first batch
                    logger.info(f"Original trajectory shape: {traj_np.shape}") # Use logger.info
                    # Print interpolated shape only if interpolation happened
                    if interpolate: 
                        logger.info(f"Interpolated trajectory shape: {interpolated_traj.shape}") # Use logger.info
                    else:
                        logger.info(f"Transformed (but not interpolated) trajectory shape: {interpolated_traj.shape}") # Use logger.info
                    logger.info(f"Batch tensor shape: {batch_tensor.shape}") # Use logger.info
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings, axis=0)
    
    # Create results dict
    results = {
        'original_trajectories': original_trajectories,
        'transformed_trajectories': transformed_trajectories,
        'embeddings': all_embeddings,
        'semi_percentage': semi_percentage,  # Store the percentage used
        'model_name': model_name  # Store the model name used
    }
    
    # Save to disk
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Saved trajectories and embeddings to {output_file}") # Use logger.info
    logger.info(f"Embeddings shape: {all_embeddings.shape}") # Use logger.info
    
    return results

def load_trained_model_embeddings(repo, model_name):
    """
    Load embeddings from a trained model file
    
    Args:
        repo: ModelRepository instance
        model_name: Name of the model to load embeddings for
        
    Returns:
        dict: Dictionary containing train_trajectories and train_embeddings
    """
    model_path = repo.get_model_embeddings_path(model_name)
    logger.info(f"Loading trained model embeddings from: {model_path}") # Use logger.info
    
    # Load the embeddings dictionary
    embeddings_dict = torch.load(model_path, weights_only=False)
    
    # Check for expected keys
    if 'train_embeddings' not in embeddings_dict:
        raise KeyError(f"'train_embeddings' key not found in model file. Available keys: {embeddings_dict.keys()}")
    
    # Convert to numpy if needed
    train_embeddings = embeddings_dict['train_embeddings'].cpu().numpy() if torch.is_tensor(embeddings_dict['train_embeddings']) else embeddings_dict['train_embeddings']
    train_trajectories = embeddings_dict['train_trajectories'].cpu().numpy() if torch.is_tensor(embeddings_dict['train_trajectories']) else embeddings_dict['train_trajectories']
    
    logger.info(f"Loaded {train_embeddings.shape[0]} training embeddings with dimension {train_embeddings.shape[1]}") # Use logger.info
    
    return {
        'train_trajectories': train_trajectories,
        'train_embeddings': train_embeddings
    }

def perform_faiss_search(query_embeddings, reference_embeddings, k=16):
    """
    Use FAISS to find k-nearest embeddings for each query embedding
    
    Args:
        query_embeddings: Query embedding vectors
        reference_embeddings: Reference embedding vectors to search in
        k: Number of nearest neighbors to retrieve
        
    Returns:
        distances: Distances to the k nearest neighbors
        indices: Indices of the k nearest neighbors
    """
    # Get embedding dimensions
    query_dim = query_embeddings.shape[1]
    ref_dim = reference_embeddings.shape[1]
    
    # Check dimension match
    if query_dim != ref_dim:
        raise ValueError(
            f"Dimension mismatch: Query embeddings have dimension {query_dim} "
            f"but reference embeddings have dimension {ref_dim}. "
            "Please ensure both embeddings have the same dimension."
        )
    
    # Print dimensions for debugging
    logger.info(f"Query embeddings shape: {query_embeddings.shape}") # Use logger.info
    logger.info(f"Reference embeddings shape: {reference_embeddings.shape}") # Use logger.info
    
    # Ensure data is in float32 format as required by FAISS
    query_embeddings = query_embeddings.astype('float32')
    reference_embeddings = reference_embeddings.astype('float32')
    
    # Build the FAISS index
    logger.info(f"Building FAISS index for {len(reference_embeddings)} embeddings...") # Use logger.info
    try:
        # Try to use GPU first
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(ref_dim)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(reference_embeddings)
        
        # Search for the nearest neighbors
        distances, indices = gpu_index_flat.search(query_embeddings, k)
        logger.info("FAISS search completed using GPU") # Use logger.info
    except Exception as e:
        logger.warning(f"GPU FAISS failed with error: {str(e)}. Falling back to CPU.") # Use logger.warning
        try:
            # Fall back to CPU if GPU fails
            index_flat = faiss.IndexFlatL2(ref_dim)
            index_flat.add(reference_embeddings)
            distances, indices = index_flat.search(query_embeddings, k)
            logger.info("FAISS search completed using CPU") # Use logger.info
        except Exception as e:
            logger.error(f"FAISS search failed on both GPU and CPU: {str(e)}") # Use logger.error
            raise RuntimeError(f"FAISS search failed on both GPU and CPU: {str(e)}")
    
    return distances, indices

def plot_embedding_nearest_neighbors(query_embeddings, query_trajectories, 
                                    ref_embeddings, ref_trajectories,
                                    num_queries=10, k=16, 
                                    output_dir=None):
    """
    Randomly select query embeddings, find their nearest neighbors, and plot the results
    
    Args:
        query_embeddings: Embeddings generated from our model
        query_trajectories: Trajectories corresponding to query_embeddings
        ref_embeddings: Reference embeddings to search in
        ref_trajectories: Trajectories corresponding to ref_embeddings
        num_queries: Number of random query embeddings to select
        k: Number of nearest neighbors to find
        output_dir: Directory to save plots (if None, plots are displayed)
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Randomly select query indices
    query_indices = np.random.choice(len(query_embeddings), size=num_queries, replace=False)
    
    # Extract query embeddings for selected indices
    selected_query_embeddings = query_embeddings[query_indices]
    
    # Perform FAISS search
    distances, neighbor_indices = perform_faiss_search(selected_query_embeddings, ref_embeddings, k=k)
    
    # Plot each query and its nearest neighbors
    for i in range(num_queries):
        query_trajectory = query_trajectories[query_indices[i]]
        neighbors = [ref_trajectories[idx] for idx in neighbor_indices[i]]
        
        # Create plot
        fig = plot_query_and_nearest_trajectories(
            query_trajectory, 
            neighbors, 
            title=f"Query {i+1} and its {k} Nearest Neighbors"
        )
        
        # Save or display the plot
        if output_dir:
            fig.savefig(os.path.join(output_dir, f"query_{i+1}_neighbors.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            
        # Print distance information
        logger.info(f"Query {i+1} - Min distance: {distances[i].min():.4f}, Max distance: {distances[i].max():.4f}") # Use logger.info

def run_embedding_search_analysis(repo, query_model_name=None, ref_model_name=None, num_queries=10, k=6, results_path=None):
    """
    Load generated embeddings and trained model embeddings,
    perform FAISS search, and plot results
    
    Args:
        repo: ModelRepository instance
        query_model_name: Name of the model used for query embeddings (auto-detected if None)
        ref_model_name: Name of the model to use for reference embeddings (auto-detected if None)
        num_queries: Number of random query embeddings to select
        k: Number of nearest neighbors to find
        results_path: Path to the generated embeddings file (auto-detected if None)
    """
    # Auto-detect model names if not provided
    if query_model_name is None:
        query_model_name = repo.get_default_model_name()
        logger.info(f"Using auto-detected query model: {query_model_name}") # Use logger.info
    
    if ref_model_name is None:
        ref_model_name = query_model_name
        logger.info(f"Using auto-detected reference model: {ref_model_name}") # Use logger.info
    
    # Auto-detect results path if not provided
    if results_path is None:
        latest_result = repo.get_latest_evaluation_result()
        if latest_result:
            target_path = latest_result
            logger.info(f"Using latest evaluation result: {target_path}") # Use logger.info
        else:
            target_path = repo.get_evaluation_results_path(query_model_name)
            logger.info(f"Using default evaluation path: {target_path}") # Use logger.info
    else:
        target_path = Path(results_path)
    
    if not target_path.exists():
        logger.error(f"Results file not found at {target_path}. Please run the embedding generation first.") # Use logger.error
        return
    
    logger.info(f"Loading generated embeddings from {target_path}") # Use logger.info
    with open(target_path, 'rb') as f:
        results = pickle.load(f)
    
    # Use transformed trajectories for consistent visualization
    if 'transformed_trajectories' in results:
        results_trajectories = results['transformed_trajectories']
    else:
        # Fallback for compatibility with older saved results
        results_trajectories = results.get('trajectories', [])
        
    results_embeddings = results['embeddings']
    logger.info(f"Loaded {len(results_embeddings)} generated embeddings") # Use logger.info
    if len(results_embeddings) > 0:
        logger.info(f"Shape of the first query embedding sample: {results_embeddings[0].shape}") # Use logger.info
    else:
        logger.warning("Warning: No query embeddings loaded, cannot print shape.") # Use logger.warning
    
    # Load trained model embeddings
    trained_data = load_trained_model_embeddings(repo, ref_model_name)
    
    # Create output directory for plots
    plots_dir = repo.get_plots_dir(query_model_name, f"vs_{ref_model_name}" if ref_model_name != query_model_name else "")
    
    # Run embedding search and plot
    plot_embedding_nearest_neighbors(
        results_embeddings,
        results_trajectories,
        trained_data['train_embeddings'],
        trained_data['train_trajectories'],
        num_queries=num_queries,
        k=k,
        output_dir=plots_dir
    )
    
    logger.info(f"Plots saved to {plots_dir}") # Use logger.info

def calculate_ade(traj1, traj2):
    """
    Calculate Average Displacement Error (ADE) between two trajectories.
    
    Args:
        traj1 (numpy.ndarray): First trajectory of shape (num_steps, 2)
        traj2 (numpy.ndarray): Second trajectory of shape (num_steps, 2)
        
    Returns:
        float: ADE value
    """
    # Ensure both trajectories have the same length by truncating to the shorter one if needed
    min_length = min(len(traj1), len(traj2))
    traj1 = traj1[:min_length]
    traj2 = traj2[:min_length]
    
    # Calculate Euclidean distances between corresponding points
    distances = np.linalg.norm(traj1 - traj2, axis=1)
    
    # Calculate average
    return np.mean(distances)

def calculate_fde(traj1, traj2):
    """
    Calculate Final Displacement Error (FDE) between two trajectories.
    
    Args:
        traj1 (numpy.ndarray): First trajectory of shape (num_steps, 2)
        traj2 (numpy.ndarray): Second trajectory of shape (num_steps, 2)
        
    Returns:
        float: FDE value
    """
    # Ensure both trajectories have the same length by truncating to the shorter one if needed
    min_length = min(len(traj1), len(traj2))
    traj1 = traj1[:min_length]
    traj2 = traj2[:min_length]
    
    # Calculate Euclidean distance between final points
    return np.linalg.norm(traj1[-1] - traj2[-1])

def calculate_min_ade(query_trajectory, neighbor_trajectories):
    """
    Calculate the minimum ADE between a query trajectory and a set of neighbor trajectories.
    
    Args:
        query_trajectory (numpy.ndarray): Query trajectory of shape (num_steps, 2)
        neighbor_trajectories (list): List of neighbor trajectories, each of shape (num_steps, 2)
    
    Returns:
        float: Minimum ADE value
        int: Index of the trajectory with the minimum ADE
    """
    min_ade = float('inf')
    min_ade_idx = -1
    
    for i, neighbor_traj in enumerate(neighbor_trajectories):
        ade = calculate_ade(query_trajectory, neighbor_traj)
        if ade < min_ade:
            min_ade = ade
            min_ade_idx = i
            
    return min_ade, min_ade_idx

def perform_minADE_analysis(query_trajectories, query_embeddings, ref_trajectories, ref_embeddings, k=16):
    """
    Perform minADE and minFDE analysis by finding the K nearest neighbor trajectories for each query 
    and calculating the minimum ADE and FDE.
    
    Args:
        query_trajectories (numpy.ndarray): Array of query trajectories
        query_embeddings (numpy.ndarray): Array of query trajectory embeddings
        ref_trajectories (numpy.ndarray): Array of reference trajectories
        ref_embeddings (numpy.ndarray): Array of reference trajectory embeddings
        k (int): Number of nearest neighbors to find
        
    Returns:
        dict: Dictionary containing analysis results
    """
    logger.info(f"Performing minADE and minFDE analysis with k={k}") # Use logger.info
    
    # Perform FAISS search to find nearest neighbors
    distances, indices = perform_faiss_search(query_embeddings, ref_embeddings, k=k)
    
    # Calculate minADE and minFDE for each query trajectory
    min_ades = []
    avg_ades = []
    min_fdes = []
    avg_fdes = []
    
    for i, query_traj in enumerate(tqdm(query_trajectories, desc="Calculating metrics")):
        # Get the K nearest neighbor trajectories
        neighbor_indices = indices[i]
        neighbor_trajs = [ref_trajectories[idx] for idx in neighbor_indices]
        
        # Calculate ADE and FDE for each neighbor trajectory
        ades = []
        fdes = []
        for neighbor_traj in neighbor_trajs:
            ade = calculate_ade(query_traj, neighbor_traj)
            fde = calculate_fde(query_traj, neighbor_traj)
            ades.append(ade)
            fdes.append(fde)
        
        # Store minimum and average values
        min_ades.append(min(ades))
        avg_ades.append(np.mean(ades))
        min_fdes.append(min(fdes))
        avg_fdes.append(np.mean(fdes))
        
        # Print progress every 1000 trajectories
        if i % 1000 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(query_trajectories)} trajectories") # Use logger.info
            logger.info(f"Current mean minADE: {np.mean(min_ades):.4f}, mean minFDE: {np.mean(min_fdes):.4f}") # Use logger.info
    
    # Calculate statistics
    mean_min_ade = np.mean(min_ades)
    median_min_ade = np.median(min_ades)
    std_min_ade = np.std(min_ades)
    
    mean_avg_ade = np.mean(avg_ades)
    median_avg_ade = np.median(avg_ades)
    std_avg_ade = np.std(avg_ades)
    
    mean_min_fde = np.mean(min_fdes)
    median_min_fde = np.median(min_fdes)
    std_min_fde = np.std(min_fdes)
    
    mean_avg_fde = np.mean(avg_fdes)
    median_avg_fde = np.median(avg_fdes)
    std_avg_fde = np.std(avg_fdes)
    
    results = {
        'min_ades': min_ades,
        'avg_ades': avg_ades,
        'min_fdes': min_fdes,
        'avg_fdes': avg_fdes,
        'stats': {
            'mean_min_ade': mean_min_ade,
            'median_min_ade': median_min_ade,
            'std_min_ade': std_min_ade,
            'mean_avg_ade': mean_avg_ade,
            'median_avg_ade': median_avg_ade,
            'std_avg_ade': std_avg_ade,
            'mean_min_fde': mean_min_fde,
            'median_min_fde': median_min_fde,
            'std_min_fde': std_min_fde,
            'mean_avg_fde': mean_avg_fde,
            'median_avg_fde': median_avg_fde,
            'std_avg_fde': std_avg_fde,
            'k': k
        }
    }
    
    return results

def run_minADE_analysis(repo, query_model_name=None, ref_model_name=None, k=16, results_path=None, save_results=True):
    """
    Load generated embeddings and trained model embeddings,
    perform FAISS search, and calculate minADE and minFDE statistics
    
    Args:
        repo: ModelRepository instance
        query_model_name: Name of the model used for query embeddings (auto-detected if None)
        ref_model_name: Name of the model to use for reference embeddings (auto-detected if None)
        k: Number of nearest neighbors to find
        results_path: Path to the generated embeddings file (auto-detected if None)
        save_results: Whether to save the results to disk
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Auto-detect model names if not provided
    if query_model_name is None:
        query_model_name = repo.get_default_model_name()
        logger.info(f"Using auto-detected query model: {query_model_name}") # Use logger.info
    
    if ref_model_name is None:
        ref_model_name = query_model_name
        logger.info(f"Using auto-detected reference model: {ref_model_name}") # Use logger.info
    
    # Auto-detect results path if not provided
    if results_path is None:
        latest_result = repo.get_latest_evaluation_result()
        if latest_result:
            target_path = latest_result
            logger.info(f"Using latest evaluation result: {target_path}") # Use logger.info
        else:
            target_path = repo.get_evaluation_results_path(query_model_name)
            logger.info(f"Using default evaluation path: {target_path}") # Use logger.info
    else:
        target_path = Path(results_path)
    
    if not target_path.exists():
        logger.error(f"Results file not found at {target_path}. Please run the embedding generation first.") # Use logger.error
        return
    
    logger.info(f"Loading generated embeddings from {target_path}") # Use logger.info
    with open(target_path, 'rb') as f:
        results = pickle.load(f)
    
    # Use transformed trajectories for consistent visualization
    if 'transformed_trajectories' in results:
        results_trajectories = results['transformed_trajectories']
    else:
        # Fallback for compatibility with older saved results
        results_trajectories = results.get('trajectories', [])
        
    results_embeddings = results['embeddings']
    logger.info(f"Loaded {len(results_embeddings)} generated embeddings") # Use logger.info
    
    # Load trained model embeddings
    trained_data = load_trained_model_embeddings(repo, ref_model_name)
    
    # Perform minADE and minFDE analysis
    analysis_results = perform_minADE_analysis(
        results_trajectories,
        results_embeddings,
        trained_data['train_trajectories'],
        trained_data['train_embeddings'],
        k=k
    )
    
    # Print results
    stats = analysis_results['stats']
    logger.info("\n--- Trajectory Metrics Analysis Results ---") # Use logger.info
    logger.info(f"Number of query trajectories: {len(results_trajectories)}") # Use logger.info
    logger.info(f"Number of reference trajectories: {len(trained_data['train_trajectories'])}") # Use logger.info
    logger.info(f"K nearest neighbors: {k}") # Use logger.info
    logger.info("\n-- Average Displacement Error (ADE) --") # Use logger.info
    logger.info(f"Mean minADE: {stats['mean_min_ade']:.4f}") # Use logger.info
    logger.info(f"Median minADE: {stats['median_min_ade']:.4f}") # Use logger.info
    logger.info(f"Std minADE: {stats['std_min_ade']:.4f}") # Use logger.info
    logger.info(f"Mean avgADE: {stats['mean_avg_ade']:.4f}") # Use logger.info
    logger.info(f"Median avgADE: {stats['median_avg_ade']:.4f}") # Use logger.info
    logger.info(f"Std avgADE: {stats['std_avg_ade']:.4f}") # Use logger.info
    logger.info("\n-- Final Displacement Error (FDE) --") # Use logger.info
    logger.info(f"Mean minFDE: {stats['mean_min_fde']:.4f}") # Use logger.info
    logger.info(f"Median minFDE: {stats['median_min_fde']:.4f}") # Use logger.info
    logger.info(f"Std minFDE: {stats['std_min_fde']:.4f}") # Use logger.info
    logger.info(f"Mean avgFDE: {stats['mean_avg_fde']:.4f}") # Use logger.info
    logger.info(f"Median avgFDE: {stats['median_avg_fde']:.4f}") # Use logger.info
    logger.info(f"Std avgFDE: {stats['std_avg_fde']:.4f}") # Use logger.info
    
    # Save results to disk if requested
    if save_results:
        output_file = repo.get_evaluation_results_path(
            query_model_name,
            f"metrics_analysis_k{k}_vs_{ref_model_name}"
        )
        with open(output_file, 'wb') as f:
            pickle.dump(analysis_results, f)
        logger.info(f"Saved analysis results to {output_file}") # Use logger.info
    
    return analysis_results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trajectory embedding generation and FAISS similarity search')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate embeddings command
    generate_parser = subparsers.add_parser('generate', help='Generate embeddings from trajectories')
    generate_parser.add_argument('--model-name', type=str, default=None,
                               help='Name of the model to use for saving results (auto-detected if not provided)')
    generate_parser.add_argument('--semi', type=int, default=100,
                               help='Percentage of data to process (1-100). Default: 100')
    generate_parser.add_argument('--data-path', type=str, default=None,
                               help='Path to the AV1 dataset. If not provided, uses the default path.')
    generate_parser.add_argument('--interpolate', type=bool, default=False,
                               help='Whether to interpolate the trajectories to shape (60,2). Default: False')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Perform similarity search on embeddings')
    search_parser.add_argument('--query-model', type=str, default=None,
                              help='Name of the model used for query embeddings (auto-detected if not provided)')
    search_parser.add_argument('--ref-model', type=str, default=None,
                              help='Name of the model to use for reference embeddings (auto-detected if not provided)')
    search_parser.add_argument('--num-queries', type=int, default=10,
                              help='Number of random query embeddings to select')
    search_parser.add_argument('--k', type=int, default=16,
                              help='Number of nearest neighbors to find')
    search_parser.add_argument('--results-path', type=str, default=None,
                              help='Path to the generated embeddings file (auto-detected if not provided)')
    
    # Perform statistical analysis command
    stats_parser = subparsers.add_parser('perform-stat', help='Perform statistical analysis on embeddings')
    stats_parser.add_argument('--query-model', type=str, default=None,
                             help='Name of the model used for query embeddings (auto-detected if not provided)')
    stats_parser.add_argument('--ref-model', type=str, default=None,
                             help='Name of the model to use for reference embeddings (auto-detected if not provided)')
    stats_parser.add_argument('--k', type=int, default=16,
                             help='Number of nearest neighbors to find')
    stats_parser.add_argument('--results-path', type=str, default=None,
                             help='Path to the generated embeddings file (auto-detected if not provided)')
    stats_parser.add_argument('--no-save', action='store_true',
                             help='Do not save analysis results to disk')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available trained models')
    
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Path to the model directory containing checkpoints and config')
    
    return parser.parse_args()

def set_seed(seed=2024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

if __name__ == "__main__":

    """
    Standalone script for generating and searching embeddings using FAISS. Requires the following directory structure:
    use conda env: faiss_sim_search
    
    trained_models/
     └── model_name/ <--- manual (this is the --model_dir)
         ├── config.yaml <--- manual
         ├── checkpoint.ckpt <--- manual
         ├── embeddings_trajectory_model_name.pt # contains [train_trajectories,train_embeddings, test_trajectories, test_embeddings] <--- manual
         └── evaluation_results/
             └── eval_results.pickle # will be created by the script


    ## Generate: 
    ### In this mode, the script will generate embeddings for the trajectories in the AV1/AV2/WOMD dataset.
    The embeddings will be saved to the model directory. All the paths are auto-detected other than the model_dir path. 
    The interpolate flag is optional and defaults to False. But if you set this true you're using the trained_trajectories shape (60,2). 
    Although this is not required as the model can handle the variable length of trajectories upto a flattend length of 1024 (512,2).
    --model-dir : Path to the model directory containing checkpoints and config files

    $ python core/data_analysis/faiss_eval_search.py \
    --model-dir /home/abishek/git_area/contrast_compress/data/trained_models/lucky-blame \
    generate --semi 100 --interpolate True

    ## Search: In this mode, the script will perform a similarity search on the embeddings of the query model
    with the embeddings of the reference model. The results will be saved to the model directory.
    In the following example, the query and reference models are optional and auto-detected based on the model_dir path

    $ python core/data_analysis/faiss_eval_search.py \
    --model-dir /home/abishek/git_area/contrast_compress/data/trained_models/lucky-blame \
    search --query-model 100-march-pudding-64 --ref-model 100-march-pudding-64 --num-queries 10 --k 6
    
    ## Perform-stat: In this mode, the script will calculate minADE and minFDE between query trajectories 
    and their K nearest neighbors found through FAISS search.
    
    $ python core/data_analysis/faiss_eval_search.py \
    --model-dir /home/abishek/git_area/contrast_compress/data/trained_models/lucky-blame \
    perform-stat --query-model 100-march-pudding-64 --ref-model 100-march-pudding-64 --k 6

    Switch to a different evaluation result file by performing the touch following command. Since the code picks the latest evaluation result file so a `touch` ensures it.
    $ touch /home/abishek/git_area/contrast_compress/data/trained_models/ugly-melody-cosine/evaluation_results/eval_results_1.pickle
    
    and then you can run the following command to perform minADE and minFDE analysis.
    $ python core/data_analysis/faiss_eval_search.py --model-dir /home/abishek/git_area/contrast_compress/data/trained_models/ugly-melody-cosine/ perform-stat --k 6
    """
    set_seed()
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    repo = ModelRepository(model_dir=args.model_dir)

    logger.info(f"Set model directory to: {args.model_dir}") # Use logger.info
    logger.info(f"Config file: {repo.config_path}") # Use logger.info
    logger.info(f"Latest checkpoint: {repo.trained_model_path}") # Use logger.info
        
    if args.command == 'generate':
        # Validate semi-supervised percentage
        if not (0 < args.semi <= 100):
            logger.error("--semi must be between 1 and 100") # Use logger.error
            sys.exit(1)
        
        # Check for required files
        if not repo.config_path or not repo.trained_model_path:
            logger.error("Could not find config file or checkpoint in model directory") # Use logger.error
            sys.exit(1)
            
        # Load the dataset
        data_path = args.data_path if args.data_path else DEFAULT_AV1_DATA_PATH
        av1_data = load_av1_dataset(data_path)
        
        # Load the model
        model = load_model_from_config(repo.config_path, repo.trained_model_path)
        logger.info(f"Loaded model from {repo.trained_model_path}") # Use logger.info
        
        # Generate and save embeddings
        if args.model_name is None:
            args.model_name = repo.get_default_model_name()
        
        results = generate_and_save_embeddings(
            model, 
            av1_data, 
            repo, 
            args.model_name, 
            semi_percentage=args.semi,
            interpolate=args.interpolate
        )
        
        logger.info(f"Done! To perform embedding search analysis, run with 'search --query-model {args.model_name}' command.") # Use logger.info
        
    elif args.command == 'search':
        # Run embedding search analysis with specified parameters
        run_embedding_search_analysis(
            repo,
            query_model_name=args.query_model,
            ref_model_name=args.ref_model,
            num_queries=args.num_queries,
            k=args.k,
            results_path=args.results_path
        )
    
    elif args.command == 'perform-stat':
        # Run minADE analysis with specified parameters
        run_minADE_analysis(
            repo,
            query_model_name=args.query_model,
            ref_model_name=args.ref_model,
            k=args.k,
            results_path=args.results_path,
            save_results=not args.no_save
        )
        
    elif args.command == 'list-models':
        # List available models
        repo.list_available_models()
            
    else:
        # No command specified, print help
        logger.info("Please specify a command: generate, search, perform-stat, list-models") # Use logger.info
        logger.info("Run with --help for more information") # Use logger.info