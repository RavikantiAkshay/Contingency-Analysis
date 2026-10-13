"""
Automated unit tests for N-1 contingency generation and application.
"""
from config.config import get_config
from src.network.load_network import load_network
from src.contingency.generate_contingencies import generate_contingencies
from src.contingency.apply_contingency import apply_contingency


def test_generate_line_contingencies():
    mpc, meta = load_network()
    contingencies = generate_contingencies(meta, mpc, contingency_type="line")
    assert len(contingencies) == 20, f"Expected 20 line contingencies, got {len(contingencies)}"
    first = contingencies[0]
    assert first['type'] == 'line'
    assert first['element_idx'] == 0
    assert 'bus1' in first and 'bus2' in first


def test_apply_line_contingency():
    mpc, meta = load_network()
    contingencies = generate_contingencies(meta, mpc, contingency_type="line")
    target = contingencies[0]  # First line contingency

    modified_mpc = apply_contingency(mpc, target)
    assert mpc['branch'][0, 10] == 1, "Base mpc was mutated!"
    assert modified_mpc['branch'][0, 10] == 0, "Target branch status was not set to 0!"


def test_random_load_generator():
    from src.contingency.random_load_generator import generate_random_load_pattern, find_valid_base_case
    cfg = get_config()
    mpc, meta = load_network()

    pattern, desc = generate_random_load_pattern(mpc, min_scale=0.80, max_scale=1.20)
    assert len(pattern) >= 2, "Expected at least 2 target buses in random load pattern."
    for bus_id, scale in pattern.items():
        assert 0.80 <= scale <= 1.20, f"Scale {scale} out of range [0.80, 1.20]"

    valid_pattern, base_res, desc = find_valid_base_case(mpc, cfg, min_scale=0.80, max_scale=1.20)
    assert valid_pattern is not None
    assert base_res['min_voltage'] > 0.80


def test_contingency_ranking():
    from src.contingency.contingency_ranking import (
        compute_voltage_severity_index,
        compute_thermal_severity_index,
        compute_contingency_severity_score,
        rank_contingencies
    )
    # 1. Nominal voltages should give 0.0 PI_V
    nominal_v = [1.0] * 14
    assert compute_voltage_severity_index(nominal_v) == 0.0

    # 2. Deviated voltages should give positive PI_V
    deviated_v = [1.0] * 13 + [0.85]  # Bus 14 at 0.85 (0.10 below 0.95)
    pi_v = compute_voltage_severity_index(deviated_v)
    assert pi_v > 0.0

    # 3. Line overloads should increase PI_MW
    nominal_loadings = [50.0] * 20
    overloaded = [50.0] * 19 + [180.0]  # One line at 180%
    assert compute_thermal_severity_index(overloaded) > compute_thermal_severity_index(nominal_loadings)

    # 4. Non-convergent case should receive massive penalty
    score_div = compute_contingency_severity_score({}, converged=False)
    assert score_div['csi'] >= 10000.0

    # 5. Ranking sorts highest severity to Rank 1
    cases = [
        {'contingency_name': 'SafeLine', 'element_idx': 0, 'results': {'convergence': True, 'bus_voltages': [1.0]*14, 'line_loadings': [40.0]*20}, 'converged': True, 'label': 'Safe'},
        {'contingency_name': 'BadLine', 'element_idx': 1, 'results': {'convergence': True, 'bus_voltages': [0.85]*14, 'line_loadings': [150.0]*20}, 'converged': True, 'label': 'Critical'},
    ]
    df_ranked = rank_contingencies(cases)
    assert df_ranked.iloc[0]['contingency'] == 'BadLine'
    assert df_ranked.iloc[0]['rank'] == 1

