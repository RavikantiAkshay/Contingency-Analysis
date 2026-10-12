"""
EEQ401 — END-TO-END DEMONSTRATION & VERIFICATION PIPELINE

Demonstrates the complete autonomous contingency remediation loop:
  1. Scenario & Network Ingestion
  2. N-1 Contingency Injection (Line/Generator outages)
  3. AC Newton-Raphson Power Flow Execution
  4. Dual Classification: Rule-Based Logic + Deep Neural Security Classifier
  5. Autonomous Decision: If insecure (Critical/Alert), Deep Q-Network Agent executes corrective control
  6. Post-Remediation AC Power Flow & Verification
  7. Detailed Trace Logging & Comprehensive Success Metrics

Usage:
  python run_full_pipeline.py --num-scenarios 10 --verbose
  python run_full_pipeline.py --num-scenarios 50 --output-csv pipeline_results.csv
"""

import os
import sys
import copy
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# Ensure standard UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure local imports work cleanly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config import get_config
from src.network.load_network import load_network
from src.dataset.scenario_pool import load_scenario_pool
from src.contingency.apply_contingency import apply_contingency
from src.simulation.run_load_flow import run_load_flow
from src.results.extract_results import extract_results
from src.labeling.classify_security import classify_security
from src.rl.classifier_integration import SecurityClassifierPredictor
from src.rl.dqn_agent import DQNAgent
from src.rl.action_space import NUM_ACTIONS, get_action_name, apply_action
from src.rl.state_builder import build_state


def parse_args():
    parser = argparse.ArgumentParser(description="Run End-to-End Contingency Analysis & RL Remediation Pipeline")
    parser.add_argument("--pool-path", type=str, default="data/tabular/dataset_100k/dataset_100k_pool.json",
                        help="Path to scenario pool JSON")
    parser.add_argument("--classifier-path", type=str, default="models/classifier_100k/grid_security_classifier.pt",
                        help="Path to trained security classifier weights")
    parser.add_argument("--scaler-path", type=str, default="models/classifier_100k/scaler_params.json",
                        help="Path to classifier scaler parameters JSON")
    parser.add_argument("--agent-path", type=str, default="models/new_samples_dqn/dqn_agent_new_samples.pt",
                        help="Path to trained DQN Agent weights")
    parser.add_argument("--num-scenarios", type=int, default=10,
                        help="Number of scenarios to test through the complete pipeline (default: 10 = 200 cases)")
    parser.add_argument("--start-scenario", type=int, default=4000,
                        help="Starting scenario index (default: 4000 to test on held-out test scenarios)")
    parser.add_argument("--output-csv", type=str, default="pipeline_demonstration_results.csv",
                        help="Path to save detailed execution traces CSV")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed step-by-step logs for each contingency case")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for neural models (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    print("\n" + "=" * 78)
    print("  EEQ401: AUTONOMOUS GRID CONTINGENCY REMEDIATION - END-TO-END PIPELINE")
    print("=" * 78)
    print(f"  [+] Device:            {args.device}")
    print(f"  [+] Scenario Pool:     {args.pool_path}")
    print(f"  [+] Classifier Model:  {args.classifier_path}")
    print(f"  [+] DQN Agent Model:   {args.agent_path}")
    print(f"  [+] Test Scenarios:    {args.num_scenarios} (starting at #{args.start_scenario})")
    print("=" * 78)

    # 1. Load Network
    base_mpc, meta = load_network()

    # 2. Load Scenario Pool
    t0 = time.time()
    if not os.path.exists(args.pool_path):
        raise FileNotFoundError(f"Scenario pool not found: {args.pool_path}")
    pool = load_scenario_pool(args.pool_path)
    print(f"  [+] Loaded scenario pool with {len(pool):,} total scenarios ({time.time() - t0:.2f}s)")

    # 3. Load Neural Security Classifier
    print(f"  [+] Loading Neural Security Classifier...")
    classifier = SecurityClassifierPredictor(
        model_path=args.classifier_path,
        scaler_path=args.scaler_path,
        device=args.device
    )
    if not classifier.is_loaded:
        print(f"  [!] Warning: Neural classifier weights not found at {args.classifier_path}. Running with rule-based fallback.")

    # 4. Load DQN Agent
    print(f"  [+] Loading Deep Q-Network Agent...")
    agent = DQNAgent(state_dim=77, num_actions=NUM_ACTIONS, device=args.device)
    if os.path.exists(args.agent_path):
        agent.load(args.agent_path)
        print(f"  [+] Successfully loaded DQN Agent weights.")
    else:
        print(f"  [!] Warning: DQN agent weights not found at {args.agent_path}. Agent will use random policy.")

    # Select test scenario indices
    max_scenarios = len(pool)
    end_sc = min(args.start_scenario + args.num_scenarios, max_scenarios)
    selected_indices = list(range(args.start_scenario, end_sc))
    cases_per_sc = len(pool[0]['contingencies'])
    total_expected_cases = len(selected_indices) * cases_per_sc
    print(f"  [+] Running complete pipeline across {len(selected_indices)} scenarios ({total_expected_cases:,} cases)...\n")

    traces = []
    cases_processed = 0
    insecure_count = 0
    pre_safe_count = 0
    post_safe_count = 0
    remediated_count = 0
    classifier_matches = 0

    crit_to_safe = 0
    crit_to_alert = 0
    alert_to_safe = 0

    pbar = tqdm(selected_indices, desc="Pipeline Execution", unit="scenario")

    for sc_idx in pbar:
        scenario = pool[sc_idx]
        sc_id = scenario['scenario_id']
        load_pattern = scenario['load_pattern']

        # Apply base scenario load scaling
        loaded_mpc = copy.deepcopy(base_mpc)
        for bus_id, scale in load_pattern.items():
            mask = loaded_mpc['bus'][:, 0] == int(bus_id)
            loaded_mpc['bus'][mask, 2] *= float(scale)
            loaded_mpc['bus'][mask, 3] *= float(scale)

        total_load_mw = float(loaded_mpc['bus'][:, 2].sum())

        for cont in scenario['contingencies']:
            cases_processed += 1
            cont_name = cont['element_name']
            outage_idx = cont['element_idx']

            # Step 1: Apply N-1 contingency
            cont_mpc = apply_contingency(loaded_mpc, cont)

            # Step 2: Post-contingency AC Power Flow
            conv_pre, raw_pre = run_load_flow(mpc=cont_mpc)
            res_pre = extract_results(raw_pre, conv_pre, meta=meta)

            # Step 3: Dual Security Classification
            rule_label_pre, _ = classify_security(res_pre, conv_pre, cfg)
            nn_label_pre, nn_conf, _ = classifier.predict(
                res_pre, conv_pre, contingency_element=cont_name, total_load_mw=total_load_mw
            )

            if nn_label_pre == rule_label_pre:
                classifier_matches += 1

            is_safe_initially = (rule_label_pre == 'Safe')
            if is_safe_initially:
                pre_safe_count += 1
            else:
                insecure_count += 1

            # Step 4: Corrective Control Decision
            obs_pre = build_state(res_pre, conv_pre, outaged_line_idx=outage_idx, total_load_mw=total_load_mw)
            info_pre = {'scenario_id': sc_id, 'contingency': cont_name, 'pre_label': rule_label_pre, 'pre_converged': conv_pre}

            # Agent chooses action based on policy
            action_idx = agent.select_action(obs_pre, info_pre, training=False)
            action_name = get_action_name(action_idx)

            # Step 5: Execute corrective action on network
            post_mpc, action_cost = apply_action(cont_mpc, action_idx)

            # Step 6: Post-action AC Power Flow & Verification
            conv_post, raw_post = run_load_flow(mpc=post_mpc)
            res_post = extract_results(raw_post, conv_post, meta=meta)
            rule_label_post, _ = classify_security(res_post, conv_post, cfg)

            is_safe_after = (rule_label_post == 'Safe')
            if is_safe_after:
                post_safe_count += 1

            # Remediation Tracking
            remediated = False
            if not is_safe_initially and is_safe_after:
                remediated = True
                remediated_count += 1

            if rule_label_pre == 'Critical' and rule_label_post == 'Safe':
                crit_to_safe += 1
            elif rule_label_pre == 'Critical' and rule_label_post == 'Alert':
                crit_to_alert += 1
            elif rule_label_pre == 'Alert' and rule_label_post == 'Safe':
                alert_to_safe += 1

            # Log trace record
            trace = {
                'scenario_id': sc_id,
                'contingency': cont_name,
                'pre_rule_label': rule_label_pre,
                'pre_nn_label': nn_label_pre,
                'nn_confidence': round(float(nn_conf), 3),
                'action_taken': action_name,
                'action_cost': action_cost,
                'post_rule_label': rule_label_post,
                'remediated_to_safe': remediated,
                'post_converged': conv_post
            }
            traces.append(trace)

            if args.verbose:
                status_icon = "[SAFE]" if is_safe_after else "[UNSAFE]"
                rem_tag = " -> REMEDIATED TO SAFE!" if remediated else ""
                print(f"[{sc_id} | {cont_name:<10}] Pre: {rule_label_pre:<8} (NN: {nn_label_pre:<8} {nn_conf*100:.1f}%) | Action: {action_name:<28} | Post: {status_icon}{rem_tag}")

    # Save CSV
    df_traces = pd.DataFrame(traces)
    df_traces.to_csv(args.output_csv, index=False)

    # Summary Report
    print("\n" + "=" * 78)
    print("  EEQ401: END-TO-END PIPELINE SUMMARY REPORT")
    print("=" * 78)
    print(f"  Total Cases Processed:              {cases_processed:,}")
    print(f"  Initial Safe Cases:                 {pre_safe_count:,} ({100.0 * pre_safe_count / max(1, cases_processed):.1f}%)")
    print(f"  Insecure Contingencies:             {insecure_count:,} ({100.0 * insecure_count / max(1, cases_processed):.1f}%)")
    print("  ----------------------------------------------------------------------------")
    print(f"  Neural Classifier Agreement:        {classifier_matches:,}/{cases_processed:,} ({100.0 * classifier_matches / max(1, cases_processed):.2f}%)")
    print("  ----------------------------------------------------------------------------")
    print(f"  Final Safe Cases (Post-Control):    {post_safe_count:,} ({100.0 * post_safe_count / max(1, cases_processed):.1f}%)")
    print(f"  Direct Remediations (Insecure->Safe): {remediated_count:,} / {insecure_count:,} ({100.0 * remediated_count / max(1, insecure_count):.1f}%)")
    print(f"    - Critical -> Safe:               {crit_to_safe:,}")
    print(f"    - Critical -> Alert:              {crit_to_alert:,}")
    print(f"    - Alert -> Safe:                  {alert_to_safe:,}")
    print("=" * 78)
    print(f"  [+] Complete per-case trace saved to: {args.output_csv}\n")


if __name__ == '__main__':
    main()
