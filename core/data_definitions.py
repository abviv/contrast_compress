from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional

@dataclass
class TransformationInfo:
    origin: np.ndarray  # Reference point (last observed position).
    rot_matrix: np.ndarray  # Rotation matrix.
    scenario_id: List[str]  # List of unique scenario identifiers.
    clipped_cl: List[Dict[int, np.ndarray]]  # List of clipped lane centerlines.
    full_lane_cl: Optional[List[Dict[int, np.ndarray]]] = None  # List of full lane centerlines.
    embeddings: List[np.ndarray] = field(default_factory=list)  # Embeddings with default empty list  # Embeddings.
    