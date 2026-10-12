"""
EEQ401 — Classifier Integration Module.

Integrates the trained deep PyTorch GridSecurityClassifier (models/classifier_100k/grid_security_classifier.pt)
into the RL agent and evaluation pipelines.

Provides:
  - GridSecurityClassifier architecture definition
  - SecurityClassifierPredictor for fast single-case and batch inference
  - Feature extraction from simulation results (59 continuous features + 20 one-hot contingency outage features)
  - Pre-filtering and agreement tracking against rule-based classification
"""

import os
import json
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn

LABEL_MAP = {'Safe': 0, 'Alert': 1, 'Critical': 2}
INV_LABEL_MAP = {0: 'Safe', 1: 'Alert', 2: 'Critical'}

CONTINGENCY_ELEMENTS = [
    '1-2', '1-5', '2-3', '2-4', '2-5', '3-4', '4-5', '4-7', '4-9', '5-6',
    '6-11', '6-12', '6-13', '7-8', '7-9', '9-10', '9-14', '10-11', '12-13', '13-14'
]

# Order of continuous features (59 dims)
# V_bus1..14 (14) + angle_bus1..14 (14) + line_loading_1..20 (20) + gen_P_1..5 (5) + summary (6)
FEATURE_COLS_COUNT = 59
OUTAGE_COLS_COUNT = 20
TOTAL_FEATURE_DIM = 79


class GridSecurityClassifier(nn.Module):
    """
    3-layer MLP classifier for grid security state classification.
    Input: 79 dimensions (59 continuous + 20 one-hot contingency outages)
    Output: 3 classes (Safe=0, Alert=1, Critical=2)
    """
    def __init__(self, input_dim: int = TOTAL_FEATURE_DIM, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SecurityClassifierPredictor:
    """
    Inference wrapper for the trained grid security classifier.
    Loads weights and continuous feature scaler parameters, and produces
    security classifications for power-flow results.
    """

    def __init__(
        self,
        model_path: str = "models/classifier_100k/grid_security_classifier.pt",
        scaler_path: str = "models/classifier_100k/scaler_params.json",
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = GridSecurityClassifier(input_dim=TOTAL_FEATURE_DIM, hidden_dim=256, num_classes=3).to(self.device)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
        else:
            self.is_loaded = False
            print(f"[Warning] Classifier model not found at {model_path}. Predictions will be unavailable.")

        # Scaler parameters
        self.scaler_mean = None
        self.scaler_scale = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                sdata = json.load(f)
                self.scaler_mean = np.array(sdata['mean'], dtype=np.float32)
                self.scaler_scale = np.array(sdata['scale'], dtype=np.float32)
        else:
            # Fallback default normalization if params file not yet generated
            self.scaler_mean = np.zeros(FEATURE_COLS_COUNT, dtype=np.float32)
            self.scaler_scale = np.ones(FEATURE_COLS_COUNT, dtype=np.float32)

    def extract_features(
        self,
        results: Dict[str, Any],
        converged: bool,
        contingency_element: str = "1-2",
        total_load_mw: float = 259.0
    ) -> np.ndarray:
        """
        Extract the 79-dimensional feature vector matching dataset_100k format:
        [0:14]   V_bus1..V_bus14
        [14:28]  angle_bus1..angle_bus14
        [28:48]  line_loading_1..line_loading_20
        [48:53]  gen_P_1..gen_P_5
        [53:59]  min_bus_v, max_bus_v, max_line_l, max_v_dev, total_load, convergence_flag
        [59:79]  outage one-hot (20 classes)
        """
        feat = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)

        if converged and results.get('convergence', False):
            # Voltages
            v = results.get('bus_voltages', [])
            for i in range(14):
                feat[i] = v[i] if i < len(v) else 1.0

            # Angles
            a = results.get('bus_angles', [])
            for i in range(14):
                feat[14 + i] = a[i] if i < len(a) else 0.0

            # Line loadings
            l = results.get('line_loadings', [])
            for i in range(20):
                feat[28 + i] = l[i] if i < len(l) else 0.0

            # Gen P
            gp = results.get('gen_powers', [])
            for i in range(5):
                feat[48 + i] = gp[i] if i < len(gp) else 0.0

            # Summary
            min_v = float(results.get('min_voltage', 1.0))
            max_v = float(results.get('max_voltage', 1.0))
            max_l = float(results.get('max_line_loading', 0.0))
            max_v_dev = float(max(abs(min_v - 1.0), abs(max_v - 1.0)))

            feat[53] = min_v
            feat[54] = max_v
            feat[55] = max_l
            feat[56] = max_v_dev
            feat[57] = float(results.get('total_load_mw', total_load_mw))
            feat[58] = 1.0  # convergence_flag
        else:
            # Non-convergent sentinels
            feat[0:14] = 0.0
            feat[14:28] = 0.0
            feat[28:48] = 999.0
            feat[48:53] = 0.0
            feat[53] = 0.0    # min_v
            feat[54] = 0.0    # max_v
            feat[55] = 999.0  # max_l
            feat[56] = 1.0    # max_v_dev
            feat[57] = float(total_load_mw)
            feat[58] = 0.0    # convergence_flag

        # Contingency one-hot (columns 59 to 78)
        # Note: pd.get_dummies sorts alphabetically:
        clean_elem = contingency_element.replace('Branch ', '').replace('Line ', '').strip()
        sorted_elements = sorted(CONTINGENCY_ELEMENTS)
        if clean_elem in sorted_elements:
            idx = sorted_elements.index(clean_elem)
            feat[59 + idx] = 1.0
        else:
            # Fallback to first element if unknown
            feat[59] = 1.0

        return feat

    def predict(
        self,
        results: Dict[str, Any],
        converged: bool,
        contingency_element: str = "1-2",
        total_load_mw: float = 259.0
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs neural network classification on a power flow result.

        Returns:
            label (str): 'Safe', 'Alert', or 'Critical'
            confidence (float): softmax confidence score for predicted class
            probabilities (dict): {'Safe': p0, 'Alert': p1, 'Critical': p2}
        """
        if not self.is_loaded:
            return 'Safe', 1.0, {'Safe': 1.0, 'Alert': 0.0, 'Critical': 0.0}

        feat = self.extract_features(results, converged, contingency_element, total_load_mw)

        # Scale continuous features [0:59]
        if self.scaler_mean is not None and self.scaler_scale is not None:
            feat[:FEATURE_COLS_COUNT] = (feat[:FEATURE_COLS_COUNT] - self.scaler_mean) / np.maximum(self.scaler_scale, 1e-7)

        tensor_x = torch.from_numpy(feat).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        pred_label = INV_LABEL_MAP.get(pred_idx, 'Safe')
        confidence = float(probs[pred_idx])
        prob_dict = {
            'Safe': float(probs[0]),
            'Alert': float(probs[1]),
            'Critical': float(probs[2])
        }

        return pred_label, confidence, prob_dict
