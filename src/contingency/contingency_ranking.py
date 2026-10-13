"""
EEQ401 — Contingency Ranking and Screening Engine.

Implements classical and modernized Power System Performance Indices (PI):
  - Voltage Severity Index (PI_V): Penalizes bus voltages outside nominal operating bounds (0.95 - 1.10 p.u.).
  - Thermal Overload Index (PI_MW): Penalizes line overloads using higher-order exponents (loading / rating)^4.
  - Non-Convergence Penalty: Severe penalty for non-convergent / voltage-collapse cases.
  - Composite Severity Index (CSI): Combined weighted score for ranking contingencies from most critical (Rank 1) to safest (Rank 20).

Used for:
  1. Real-time Emergency Triage: Prioritizing which contingency the RL Agent remediates first.
  2. Fast Pre-Filtering: Discarding benign contingencies before running slower AC power flows.
  3. Curriculum Learning: Sampling harder training scenarios for Reinforcement Learning.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


# Default Weights and Exponents
DEFAULT_WV = 1.0          # Voltage severity weight
DEFAULT_WL = 1.0          # Thermal overload weight
DEFAULT_EXPONENT_L = 4    # Exponent for thermal loading (4 sharply penalizes lines > 100%)
DEFAULT_EXPONENT_V = 2    # Exponent for voltage deviation
DIVERGENCE_PENALTY = 10000.0  # Massive penalty for non-convergent power flow


def compute_voltage_severity_index(
    bus_voltages: List[float],
    v_min: float = 0.95,
    v_max: float = 1.10,
    exponent: int = DEFAULT_EXPONENT_V
) -> float:
    """
    Computes the Voltage Performance Index (PI_V).
    
    Penalizes voltage deviations outside the [v_min, v_max] safe window.
    Buses within [v_min, v_max] contribute 0 to the index.
    """
    if not bus_voltages:
        return 0.0

    pi_v = 0.0
    for v in bus_voltages:
        dev = 0.0
        if v < v_min:
            dev = (v_min - v) / 0.05
        elif v > v_max:
            dev = (v - v_max) / 0.05
        
        if dev > 0.0:
            pi_v += float(dev ** exponent)

    return float(pi_v)


def compute_thermal_severity_index(
    line_loadings: List[float],
    rating_limit: float = 100.0,
    exponent: int = DEFAULT_EXPONENT_L
) -> float:
    """
    Computes the Active Power / Thermal Performance Index (PI_MW).
    
    Penalizes line loading percentage normalized by continuous rating.
    Using an exponent of 4 heavily penalizes severe overloads:
      - 80% loading  -> (0.8)^4 = 0.40
      - 140% loading -> (1.4)^4 = 3.84
      - 240% loading -> (2.4)^4 = 33.18
    """
    if not line_loadings:
        return 0.0

    pi_mw = 0.0
    for loading in line_loadings:
        ratio = max(0.0, float(loading) / rating_limit)
        pi_mw += float(ratio ** exponent)

    return float(pi_mw)


def compute_contingency_severity_score(
    results: Dict[str, Any],
    converged: bool,
    cfg: Optional[Dict[str, Any]] = None,
    w_v: float = DEFAULT_WV,
    w_l: float = DEFAULT_WL
) -> Dict[str, float]:
    """
    Calculates PI_V, PI_MW, and the Composite Severity Index (CSI) for a single post-contingency result.
    """
    sec_cfg = (cfg or {}).get('security', {})
    v_safe_min = sec_cfg.get('v_safe_min', 0.95)
    v_safe_max = sec_cfg.get('v_safe_max', 1.10)
    line_safe_max = sec_cfg.get('line_safe_max', 100.0)

    if not converged or not results.get('convergence', False):
        return {
            'pi_v': DIVERGENCE_PENALTY / 2.0,
            'pi_mw': DIVERGENCE_PENALTY / 2.0,
            'csi': DIVERGENCE_PENALTY,
            'diverged': True,
        }

    bus_voltages = results.get('bus_voltages', [])
    line_loadings = results.get('line_loadings', [])

    pi_v = compute_voltage_severity_index(bus_voltages, v_min=v_safe_min, v_max=v_safe_max)
    pi_mw = compute_thermal_severity_index(line_loadings, rating_limit=line_safe_max)

    csi = (w_v * pi_v) + (w_l * pi_mw)

    return {
        'pi_v': round(pi_v, 4),
        'pi_mw': round(pi_mw, 4),
        'csi': round(csi, 4),
        'diverged': False,
    }


def rank_contingencies(
    scenario_cases: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Ranks all post-contingency cases within a scenario from most severe (Rank 1) to safest (lowest severity).

    Args:
        scenario_cases: List of dicts, each containing:
            - 'contingency_name': Name of outaged branch (e.g. '1-2')
            - 'element_idx': 0-based index of outaged branch
            - 'results': Extracted power flow results dict
            - 'converged': Boolean convergence status
            - 'label': Optional ground truth label ('Safe', 'Alert', 'Critical')
        cfg: System configuration dictionary.

    Returns:
        pd.DataFrame: Table sorted with Rank 1 as the most critical contingency.
    """
    scored = []

    for case in scenario_cases:
        res = case.get('results', {})
        conv = case.get('converged', False)
        cont_name = case.get('contingency_name', 'Unknown')
        cont_idx = case.get('element_idx', -1)
        label = case.get('label', 'Unknown')

        scores = compute_contingency_severity_score(res, conv, cfg)

        scored.append({
            'contingency': cont_name,
            'branch_idx': cont_idx,
            'csi': scores['csi'],
            'pi_v': scores['pi_v'],
            'pi_mw': scores['pi_mw'],
            'converged': conv,
            'label': label,
        })

    # Sort descending by CSI (Highest severity score = Rank 1)
    df = pd.DataFrame(scored)
    df = df.sort_values(by='csi', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    # Reorder columns for display
    cols = ['rank', 'contingency', 'csi', 'pi_v', 'pi_mw', 'label', 'converged', 'branch_idx']
    return df[cols]
