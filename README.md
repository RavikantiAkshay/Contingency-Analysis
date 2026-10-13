# Autonomous Grid Contingency Remediation & AI Security Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Power System](https://img.shields.io/badge/Grid-IEEE%2014--Bus-success.svg)](#)
[![Dataset](https://img.shields.io/badge/Cases-600%2C000%20(3%20Tiers)-orange.svg)](#4-large-scale-contingency-datasets-600k-cases)
[![Classifier Accuracy](https://img.shields.io/badge/Classifier%20Accuracy-98.3%25-brightgreen.svg)](#6-deep-neural-grid-security-classifier)
[![RL Agent Safety](https://img.shields.io/badge/RL%20Post--Control%20Safety-50.2%25-blueviolet.svg)](#7-deep-q-network-dqn-remediation-engine)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(18%2F18)-brightgreen.svg)](#12-test-suite--validation)

An end-to-end autonomous artificial intelligence framework for high-voltage power transmission grids. The platform integrates AC Newton-Raphson power flow physics, systematic N-1 contingency analysis, 600,000 synthetic simulation cases, a deep neural grid security classifier, a Deep Q-Network (DQN) autonomous corrective remediation agent, and classical contingency ranking indices on the standard IEEE 14-bus test system.

---

## Table of Contents
1. [Project Overview & Key Objectives](#1-project-overview--key-objectives)
2. [End-to-End System Pipeline & Workflow](#2-end-to-end-system-pipeline--workflow)
3. [AC Power Flow Physics & Ground Truth Security Rules](#3-ac-power-flow-physics--ground-truth-security-rules)
4. [Large-Scale Contingency Datasets (600,000 Cases)](#4-large-scale-contingency-datasets-600000-cases)
5. [Contingency Ranking & Power System Severity Indices](#5-contingency-ranking--power-system-severity-indices)
6. [Deep Neural Grid Security Classifier (79-D Input)](#6-deep-neural-grid-security-classifier-79-d-input)
7. [Deep Q-Network (DQN) Remediation Engine (77-D State)](#7-deep-q-network-dqn-remediation-engine-77-d-state)
8. [Neural Network Layer Breakdown: What Happens Inside?](#8-neural-network-layer-breakdown-what-happens-inside)
9. [Execution Flow: Training Time vs. Testing Time](#9-execution-flow-training-time-vs-testing-time)
10. [Comprehensive Benchmark Results & Evaluation](#10-comprehensive-benchmark-results--evaluation)
11. [Codebase Architecture & Directory Structure](#11-codebase-architecture--directory-structure)
12. [Installation, CLI Usage & Execution Commands](#12-installation-cli-usage--execution-commands)
13. [Test Suite & Validation](#13-test-suite--validation)

---

## 1. Project Overview & Key Objectives

In modern power grids, unexpected transmission line outages (tripped by lightning, storm damage, or short-circuits) trigger immediate shifts in power flow. These sudden power reroutings can overload neighboring transmission lines or cause bus voltages to collapse, risking catastrophic cascading blackouts.

Traditional grid operations rely on human operators who may take several minutes to interpret alarms and issue redispatch orders. **This project delivers a closed-loop, autonomous AI system that:**
1. **Detects & Simulates** N-1 line outages across thousands of fluctuating load scenarios.
2. **Classifies Grid Security** in under 1 millisecond using a Deep Neural Network (Safe, Alert, Critical).
3. **Ranks Contingencies** by severity using classical power system performance indices (PI_V, PI_MW, CSI).
4. **Autonomously Remediates Violations** using a trained Deep Q-Network (DQN) agent that adjusts generator outputs or sheds minimal load to restore the grid to a secure operating state.

---

## 2. End-to-End System Pipeline & Workflow

The entire platform operates as a synchronized engineering pipeline:

```
[ Step 1: Base Grid Model ] ──> IEEE 14-bus system (14 buses, 20 branches, 5 generators)
           │
           ▼
[ Step 2: Load Perturbation ] ──> Random load scaling (0.8x to 1.2x across 11 load buses)
           │
           ▼
[ Step 3: N-1 Contingency ] ──> Systematic branch trip (all 20 branches evaluated individually)
           │
           ▼
[ Step 4: AC Newton-Raphson ] ──> Solves exact nonlinear polar power flow (V, theta, line flows)
           │
           ▼
[ Step 5: Feature Extraction ] ──> Assembles 63-column tabular telemetry row
           │
           ▼
[ Step 6: Dual Security Assessment ]
     ├── Physical Ground Truth Rules (Voltage bounds [0.95, 1.10], Line ratings <= 100%)
     └── Deep Neural Classifier (79-D Input ──> Safe / Alert / Critical in under 1 ms)
           │
           ▼ (If Alert or Critical)
[ Step 7: DQN Agent Remediation ] ──> 77-D Normalized State ──> Selects best of 20 physical actions
           │
           ▼
[ Step 8: Action Execution & Re-Solve ] ──> Physical grid updated; AC flow re-solved
           │
           ▼
[ Step 9: Post-Remediation Verification ] ──> Evaluates recovery to Safe state & computes reward
```

### What Happens at Each Stage:
- **Stage 1 (Grid Ingestion):** Imports the standard IEEE 14-bus test feeder (Base Power = 100 MVA, Base Voltage = 144 kV). Bus 1 is the slack reference; Buses 2, 3, 6, 8 have controllable voltage/generation; Buses 4, 5, 7, 9, 10, 11, 12, 13, 14 are customer load buses.
- **Stage 2 (Scenario Generation):** Electricity demand fluctuates continuously. The scenario generator perturbs active and reactive loads across all 11 customer substations between 80% and 120% of nominal demand, filtering out physically unsolvable base states.
- **Stage 3 (Contingency Injection):** One branch at a time is set out of service (status = 0) across all 20 lines and transformers.
- **Stage 4 (AC Power Flow Solving):** Solves nonlinear AC power balance equations via Newton-Raphson. If the power flow reaches a mathematical bifurcation or collapse, the solver flags "Non-convergence".
- **Stage 5 (Feature Extraction):** Records 14 bus voltages, 14 phase angles, 20 branch loadings, 5 generator powers, and 5 summary metrics.
- **Stage 6 (Dual Classification):** The state is checked against standard engineering thresholds and simultaneously evaluated by the deep neural classifier.
- **Stage 7 (DQN Remediation):** If the state is insecure, the 77-dimensional observation is passed to the DQN agent, which selects the optimal corrective action.
- **Stage 8 (Physical Execution):** The network model applies the generator redispatch or load curtailment and re-runs the AC power flow.
- **Stage 9 (Verification & Logging):** Verifies if the grid returned to Safe, logs the remediation trajectory, and stores full per-case telemetry.

---

## 3. AC Power Flow Physics & Ground Truth Security Rules

### 3.1 AC Newton-Raphson Load Flow Formulation
Power balance at each bus $i \in \{1, \dots, 14\}$ follows the full nonlinear AC equations in polar form:

$$P_i(V, \theta) = V_i \sum_{j=1}^{14} V_j \left( G_{ij} \cos(\theta_i - \theta_j) + B_{ij} \sin(\theta_i - \theta_j) \right)$$

$$Q_i(V, \theta) = V_i \sum_{j=1}^{14} V_j \left( G_{ij} \sin(\theta_i - \theta_j) - B_{ij} \cos(\theta_i - \theta_j) \right)$$

where:
- $V_i, \theta_i$: Voltage magnitude and phase angle at bus $i$.
- $Y_{ij} = G_{ij} + j B_{ij}$: Bus admittance matrix elements (rebuilt dynamically upon line trips).

The iterative Newton-Raphson solver corrects voltage magnitudes and angles via the Jacobian matrix:

$$\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} J_{11} & J_{12} \\ J_{21} & J_{22} \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta V / V \end{bmatrix}$$

Convergence is achieved when maximum power mismatch is under $10^{-4}$ per-unit.

### 3.2 Ground Truth Security Classification Rules
Each post-contingency operating point is classified according to IEEE operational standards:

| Security Class | Bus Voltage Magnitude ($V_i$) | Branch Thermal Loading ($S_{ij}$) | Grid Condition |
| :--- | :--- | :--- | :--- |
| **Safe** | $0.95 \le V_i \le 1.10$ per-unit | Loading $\le 100.0\%$ continuous rating | Fully compliant; normal operating state. |
| **Alert** | $0.90 \le V_i < 0.95$ or $1.10 < V_i \le 1.15$ | $100.0\% < \text{Loading} \le 120.0\%$ | Emergency rating exceeded; requires prompt corrective control. |
| **Critical** | $V_i < 0.90$ or $V_i > 1.15$ per-unit | Loading $> 120.0\%$ short-term limit | Severe overload, voltage collapse, or solver non-convergence. |

---

## 4. Large-Scale Contingency Datasets (600,000 Cases)

The repository houses **3 distinct, non-overlapping dataset tiers** stored in `data/tabular/`:

| Dataset Tier | Path | Base Scenarios | Outages / Scenario | Total Post-Contingency Rows | CSV Size | JSON Pool Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`dataset_100k`** | `data/tabular/dataset_100k/` | 5,000 | 20 | **100,000** | 91.0 MB | 20.5 MB |
| **`dataset_200k`** | `data/tabular/dataset_200k/` | 10,000 | 20 | **200,000** | 182.1 MB | 41.0 MB |
| **`dataset_300k`** | `data/tabular/dataset_300k/` | 15,000 | 20 | **300,000** | 273.1 MB | 61.4 MB |
| **Grand Total** | — | **30,000** | **20** | **600,000** | **546.2 MB** | **122.9 MB** |

### Rigorous Dataset Guarantees:
1. **Zero Repetition:** All 30,000 scenario load patterns are strictly unique across all three tiers (verified by `data/tabular/sets_manifest.json`).
2. **Schema Standardization:** Every CSV strictly conforms to the 63-column specification (4 metadata, 14 voltages, 14 angles, 20 loadings, 5 generator MW, 5 summary indices, 1 security label).
3. **Realistic Convergence Profile:** Across all 600,000 cases, 95.0% converge normally and 5.0% represent severe non-convergent voltage collapses, mirroring real-world stressed transmission conditions.

---

## 5. Contingency Ranking & Power System Severity Indices

In real transmission control rooms (Energy Management Systems), operators cannot run slow detailed simulations on every line simultaneously. Instead, they perform **contingency ranking / triage** to identify the most dangerous outages first.

Implemented in `src/contingency/contingency_ranking.py`:

### 5.1 Voltage Severity Index (PI_V)
Quantifies bus voltage deviations outside the safe operating window ($0.95$ to $1.10$ per-unit):

$$\text{PI}_V = \sum_{i=1}^{14} \left( \frac{\text{Deviation}_i}{0.05} \right)^2$$

Buses operating within the safe band contribute 0. Severe under-voltages (e.g. 0.80 per-unit) incur a heavy quadratic penalty.

### 5.2 Thermal Overload Severity Index (PI_MW)
Quantifies line loadings beyond 100% continuous MVA capacity using a fourth-power penalty:

$$\text{PI}_{MW} = \sum_{k=1}^{20} \left( \frac{\text{Loading}_k}{100.0} \right)^4$$

- 80% loading gives $(0.8)^4 = 0.40$ (low impact).
- 140% loading gives $(1.4)^4 = 3.84$ (moderate impact).
- 240% loading gives $(2.4)^4 = 33.17$ (extreme severity).

### 5.3 Composite Severity Index (CSI)
Combines voltage violations, thermal overloads, and non-convergence penalties:

$$\text{CSI} = w_V \cdot \text{PI}_V + w_L \cdot \text{PI}_{MW} + \text{Divergence Penalty}$$

Non-convergent collapses automatically receive a 10,000 penalty, shooting them straight to **Rank 1 (Most Critical)**.

### 5.4 Two-Stage Triage & Autonomous AI Remediation
Contingency Ranking and Deep Reinforcement Learning work together as a complementary two-stage operational framework:
1. **Stage 1 (Analytical Screening & Triage):** When an event occurs, the system evaluates all 20 potential branch trips and computes their Composite Severity Index (CSI). It immediately sorts the outages from **Rank 1 (severe emergencies)** down to **Rank 20 (benign)**.
2. **Stage 2 (Targeted AI Remediation):** The Deep Q-Network Agent is dispatched to resolve the top-ranked critical contingencies, while leaving benign outages alone (`Action 0: Do Nothing`).
3. **No Retraining Required:** Contingency Ranking operates as a diagnostic triage layer on post-contingency power flow results. It does not alter the underlying 77-D state space or action set, preserving all trained model checkpoints and benchmark evaluation metrics.

```python
from src.contingency.contingency_ranking import rank_contingencies

# Rank all 20 contingencies in a scenario from highest severity (Rank 1) to safest
ranking_df = rank_contingencies(scenario_cases, cfg)
print(ranking_df[['rank', 'contingency', 'csi', 'pi_v', 'pi_mw', 'label']])
```

---

## 6. Deep Neural Grid Security Classifier (79-D Input)

Implemented in `train_classifier_100k.py` and `src/rl/classifier_integration.py`.

- **Input Dimension (79 features):**
  - **59 continuous features:** 14 bus voltages, 14 bus angles, 20 line loadings, 5 generator MW, 6 summary metrics (`min_bus_voltage`, `max_bus_voltage`, `max_line_loading`, `max_voltage_deviation`, `total_load`, `convergence_flag`). Standardized via `StandardScaler`.
  - **20 categorical features:** One-hot outage indicator specifying which of the 20 branches failed.
- **Model Architecture:**
  $$\text{Input}(79) \xrightarrow{\text{Linear}} \text{Hidden}_1(256) \xrightarrow{\text{LayerNorm}} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Dropout}(0.1)} \xrightarrow{\text{Linear}} \text{Hidden}_2(256) \xrightarrow{\text{LayerNorm}} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Dropout}(0.1)} \xrightarrow{\text{Linear}} \text{Output}(3)$$
- **Output:** 3 Logits (`Safe`, `Alert`, `Critical`).
- **Performance:** **98.30% Test Accuracy**, **98.21% Macro F1**.
- **Cross-Dataset Validation:** 98.26% on `dataset_200k`, 98.17% on `dataset_300k`.

---

## 7. Deep Q-Network (DQN) Remediation Engine (77-D State)

Implemented in `src/rl/dqn_agent.py` and `src/rl/action_space.py`.

### 7.1 Observation Space (77 Dimensions)
The agent receives a continuous normalized vector:
- `[0:14]`: 14 Bus voltages (centered near 1.0 per-unit).
- `[14:28]`: 14 Bus angles (scaled by $/ 30^\circ$).
- `[28:48]`: 20 Line loadings (scaled by $/ 100\%$, clipped at 3.0).
- `[48:53]`: 5 Generator active powers (scaled by $/ 100 \text{ MW}$).
- `[53:73]`: 20-dimensional one-hot line outage mask.
- `[73:77]`: 4 Summary features ($V_{\min}, V_{\max}, L_{\max}/100, P_{\text{load}}/300$).

### 7.2 Action Space (20 Discrete Physical Actions)
The agent selects from 20 discrete operational actions:
- **Action 0:** `Do Nothing` (Maintain status quo, cost = 0.0).
- **Actions 1–8:** `Generator Redispatch` ($\pm 10 \text{ MW}$ adjustments on 4 controllable generators at Buses 2, 3, 6, 8, clamped to generator limits, cost = 2.0 flat).
- **Actions 9–19:** `Controlled Load Shedding` (10% active and reactive load curtailment at each of the 11 candidate load buses: 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, cost = $1.0 \times \Delta P_{\text{shed}}$).

### 7.3 Multi-Objective Reward Function
Evaluates corrective impact while penalizing unnecessary load cuts:

$$R = R_{\text{security}} - P_{\text{voltage}} - P_{\text{thermal}} - P_{\text{non\_conv}} - C_{\text{action}}$$

- $R_{\text{security}}$: $+50$ for reaching Safe, $+10$ for Alert, $+20$ bonus for Critical $\to$ Alert, $-30$ penalty if degraded into Critical.
- $P_{\text{voltage}}$: $-5.0 \times (\text{number of bus voltage violations})$.
- $P_{\text{thermal}}$: $-0.5 \times (\text{percentage points over } 100\% \text{ thermal rating})$.
- $P_{\text{non\_conv}}$: $-100.0$ if the action triggers solver non-convergence.
- $C_{\text{action}}$: Action operational cost (encourages redispatch over shedding customer load).

---

## 8. Neural Network Layer Breakdown: What Happens Inside?

### The Architecture: 77 inputs → 256 neurons → 256 neurons → 20 outputs

```
[ 77 Inputs ]  ──>  [ 256 Neurons ]  ──>  [ 256 Neurons ]  ──>  [ 20 Outputs ]
Telemetry Data      Feature Detectors     Strategic Context      Q-Value Scores
```

1. **Layer 1 (77 → 256):**
   - Contains 256 artificial neurons with $(77 \times 256) + 256 = \mathbf{19,968}$ trainable weights.
   - Each neuron acts as an **electrical pattern detector** (e.g. detecting when line 1-2 is tripped while bus 4 voltage drops).
   - **LayerNorm** stabilizes numerical variance; **ReLU** provides the non-linear gating required to model complex AC grid physics.
2. **Layer 2 (256 → 256):**
   - Contains 256 higher-level neurons with $(256 \times 256) + 256 = \mathbf{65,792}$ trainable weights.
   - Combines lower-level patterns into strategic assessments (e.g. identifying that line 2-4 is overloaded but generator 3 has spare capacity to relieve it without cutting load).
3. **Output Layer (256 → 20):**
   - Contains $(256 \times 20) + 20 = \mathbf{5,140}$ trainable weights.
   - Outputs 20 **Q-values** (Quality scores). Each value represents the expected cumulative reward for executing that specific action. The agent picks the action with the highest Q-value (`argmax`).

---

## 9. Execution Flow: Training Time vs. Testing Time

Here is the exact step-by-step chronological flow of what occurs inside the system during the **Training Phase** versus the **Testing / Deployment Phase**:

### 9.1 Training Phase Flow (Learning Across 80,000 Episodes)

During training, the agent acts like a pilot in a flight simulator—exploring actions, seeing consequences, making mistakes, and updating its internal brain to maximize rewards.

```
[ Step 1: Pick Scenario & Trip Line ]
                 │
                 ▼
[ Step 2: Pre-Action AC Power Flow & 77-D State Builder ]
                 │
                 ▼
[ Step 3: Decision: Explore Randomly OR Exploit Brain (Epsilon-Greedy) ]
                 │
                 ▼
[ Step 4: Apply Action (Redispatch / Load Shed) & Re-Solve Grid ]
                 │
                 ▼
[ Step 5: Reward Calculation (Safety Bonus vs. Overload/Cost Penalties) ]
                 │
                 ▼
[ Step 6: Store (State, Action, Reward, Next State) in 100k Replay Memory ]
                 │
                 ▼
[ Step 7: Sample 128 Random Memories on GPU ──> Backpropagation & Adam Update ]
                 │
                 ▼
[ Step 8: Decay Epsilon & Sync Target Network (every 200 episodes) ]
```

1. **Step 1 — Scenario & Outage Ingestion:** The environment selects a scenario from the 4,000 training scenarios, applies random customer load scaling, and trips one of the 20 transmission lines.
2. **Step 2 — Sensing & State Construction:** The AC Newton-Raphson solver computes the resulting power flow. The environment extracts 14 voltages, 14 angles, 20 line loadings, and generator outputs, packaging them into the **77-dimensional normalized state vector**.
3. **Step 3 — Action Selection (Exploration vs. Exploitation):**
   - The agent rolls a random number.
   - If below **Epsilon** (which starts at 100% and decays down to 5%), it chooses a **completely random action** among the 20 choices to discover new strategies.
   - Otherwise, it passes the 77-D state through the neural network and chooses the action with the highest predicted Q-value.
4. **Step 4 — Execution & Power Flow Re-Solve:** The chosen action (e.g. +10 MW at Generator 2, or 10% load cut at Bus 4) is applied to the grid model, and the AC Newton-Raphson solver is re-executed to find the new physical state.
5. **Step 5 — Reward Scoring:** The environment calculates a scalar reward: +50 if Safe, +20 if improved from Critical to Alert, -100 if collapsed, minus penalties for any remaining voltage or thermal violations and action operational costs.
6. **Step 6 — Experience Storage:** The transition tuple `(State, Action, Reward, Next State, Done)` is appended into the **100,000-capacity Experience Replay Buffer**.
7. **Step 7 — GPU Batch Learning & Weight Update:**
   - The agent samples a random mini-batch of **128 past experiences** from the replay buffer.
   - The **Policy Network** predicts Q-values for the taken actions.
   - The **Target Network** calculates target values using the Bellman equation: `Target = Reward + 0.95 * max(Target_Network(Next State))`.
   - The Mean Squared Error loss between predictions and targets is computed, and **backpropagation updates all neural network weights via the Adam optimizer**.
8. **Step 8 — Synchronization & Next Episode:** Epsilon is decayed slightly so the agent relies more on learned knowledge. Every 200 episodes, weights from the Policy Network are copied into the Target Network.

---

### 9.2 Testing / Inference Phase Flow (Evaluating 20,000 Unseen Test Cases)

During testing (and real-world control room deployment), the system operates in frozen execution mode—zero random exploration, zero weight updates, and instantaneous millisecond response.

```
[ Step 1: Ingest Unseen Test Scenario & Trip Line ]
                 │
                 ▼
[ Step 2: Pre-Action AC Power Flow & Dual Security Classification ]
                 │
         Is Grid Safe?
        ┌────────┴────────┐
       YES                NO (Alert / Critical)
        │                 │
        ▼                 ▼
 [ Remain Safe;   [ Step 3: 77-D State ──> Frozen Policy Network ]
   No Action ]            │
                          ▼
                  [ Step 4: Pick Single Highest Q-Value (argmax) ]
                          │
                          ▼
                  [ Step 5: Execute Action & Re-Solve AC Power Flow ]
                          │
                          ▼
                  [ Step 6: 3-Way Benchmark Verification ]
                    (Compare: No Action vs. RL Agent vs. Exhaustive Oracle)
                          │
                          ▼
                  [ Step 7: Record Detailed Case Telemetry & CSV Trace ]
```

1. **Step 1 — Unseen Outage Injection:** The pipeline loads a scenario from the 1,000 held-out test scenarios (completely unseen during training) and trips a transmission line.
2. **Step 2 — Sensing & Dual Classification:**
   - The AC power flow solver runs to measure the immediate post-trip electrical state.
   - The **Deep Security Classifier (79-D)** predicts whether the grid is Safe, Alert, or Critical in under 1 millisecond.
   - If the state is already **Safe**, the system logs that no intervention is required and skips control.
3. **Step 3 — Observation Pass into Frozen Policy Network:**
   - If Alert or Critical, the 77-D state vector is constructed.
   - **Exploration is turned off (Epsilon = 0):** No random guesses are ever allowed.
   - **Memory buffer and backpropagation are disabled:** Network weights are frozen.
   - The 77-D vector undergoes a single forward pass through the Policy Network in **under 1 millisecond**.
4. **Step 4 — Greedy Action Selection:** The agent selects the single action with the absolute highest Q-value (`Selected Action = argmax(Q)`).
5. **Step 5 — Physical Execution & Re-Solve:** The chosen action is applied to the grid, and the AC Newton-Raphson solver re-runs to evaluate the remediated grid state.
6. **Step 6 — 3-Way Benchmark Verification:**
   - To scientifically assess the agent's performance on this exact case, the pipeline evaluates:
     1. **No-Action Baseline:** What would happen if the grid was left unassisted?
     2. **RL Agent Policy:** Did the agent restore the grid from Critical/Alert to Safe?
     3. **Exhaustive Oracle (Offline Best):** All 20 possible actions are simulated one by one with 20 independent AC power flows to find the theoretical optimal action. Did the agent match the Oracle (Top-1) or choose one of the top 3 best actions (Top-3)?
7. **Step 7 — Telemetry Logging:** The full trace (pre-state, classifier prediction, action taken, action cost, post-state, and oracle match) is appended to `evaluation_cases.csv`.

---

## 10. Comprehensive Benchmark Results & Evaluation

Evaluated across **1,000 held-out test scenarios (20,000 unseen contingency cases)** in a 3-way benchmark:

```
Initial No-Action Baseline :  35.9% Safe  (7,171 safe cases out of 20,000)
DQN Agent Remediated       :  50.2% Safe  (10,040 safe cases out of 20,000)
Offline Best (Oracle Bound):  51.0% Safe  (10,200 theoretical upper limit)
```

### Detailed Evaluation Breakdown:
- **Total Test Cases Evaluated:** 20,000
- **Insecure Contingencies Requiring Help:** 12,829
- **DQN Mitigation Success Rate:** **22.4%** of insecure cases successfully brought to Safe
- **Critical → Alert Remediations:** **1,572 cases**
- **Alert → Safe Remediations:** **2,869 cases**
- **Safe States Preserved:** **7,171 cases**
- **Safe States Broken:** **0 cases (100% safety preservation)**
- **Top-1 Policy Agreement with Oracle:** 43.8%
- **Top-3 Policy Agreement with Oracle:** **82.0%**
- **Neural Classifier Accuracy on Test Set:** **98.47%**

---

## 11. Codebase Architecture & Directory Structure

```text
Contingency-Analysis/
├── config/
│   └── config.py                     # Centralized grid limits, solver tolerances & paths
│
├── src/
│   ├── network/
│   │   ├── load_network.py           # IEEE 14-bus importer and dynamic metadata parser
│   │   └── validate_network.py       # Topology and bus/branch validator
│   ├── simulation/
│   │   ├── custom_nr_solver.py       # Standalone vectorized AC Newton-Raphson solver
│   │   ├── run_load_flow.py          # PyPOWER reference solver wrapper
│   │   └── run_simulation_campaign.py# Parallel simulation campaign runner
│   ├── contingency/
│   │   ├── generate_contingencies.py # N-1 branch outage generator
│   │   ├── apply_contingency.py      # Outage matrix modifier
│   │   ├── random_load_generator.py  # Rejection-sampled load perturbation engine
│   │   └── contingency_ranking.py    # Power system severity indices (PI_V, PI_MW, CSI)
│   ├── labeling/
│   │   └── classify_security.py      # Rule-based security classifier (Safe, Alert, Critical)
│   ├── dataset/
│   │   ├── scenario_pool.py          # Scenario pool manager & JSON serialization
│   │   ├── build_dataset.py          # Tabular dataset builder (63 columns)
│   │   └── validate_dataset.py       # Dataset schema and completeness validator
│   ├── rl/
│   │   ├── action_space.py           # 20 discrete actions (redispatch + load shedding)
│   │   ├── state_builder.py          # 77-dimensional normalized observation vector
│   │   ├── reward.py                 # Multi-objective security reward function
│   │   ├── dqn_agent.py              # PyTorch DQN Agent, ReplayBuffer & Target Network
│   │   ├── environment.py            # Contingency remediation RL environment
│   │   └── classifier_integration.py # 79-D classifier inference wrapper
│   └── results/
│       ├── extract_results.py        # Telemetry & violation extractor
│       └── save_results.py           # CSV and JSON output exporter
│
├── data/
│   └── tabular/                      # Multi-tier datasets (Tracked via Git LFS)
│       ├── dataset_100k/             # 100,000 cases (5,000 scenarios x 20 lines)
│       ├── dataset_200k/             # 200,000 cases (10,000 scenarios x 20 lines)
│       ├── dataset_300k/             # 300,000 cases (15,000 scenarios x 20 lines)
│       └── sets_manifest.json        # Duplicate-free verification manifest
│
├── models/
│   ├── classifier_100k/              # Trained Deep Classifier weights, scalers & metrics
│   └── new_samples_dqn/              # Trained DQN weights, checkpoints & evaluation summary
│
├── tests/                            # Automated test suite (18 unit tests, 100% passing)
│   ├── test_network.py
│   ├── test_load_flow.py
│   ├── test_contingency.py
│   ├── test_dataset.py
│   ├── test_dataset_sets.py
│   └── test_rl.py
│
├── train_classifier_100k.py          # Trains the 79-D neural security classifier
├── train_on_new_samples.py           # Master DQN training & 3-way evaluation benchmark
├── run_full_pipeline.py              # End-to-end autonomous demonstration runner
├── run_generate_dataset_sets.py      # Parallel multi-dataset campaign generator
├── pipeline_and_architecture_guide.html # Visual human-readable browser report
└── requirements.txt                  # Python dependencies
```

---

## 12. Installation, CLI Usage & Execution Commands

### Prerequisites
- Python 3.10 or higher
- Git with Git LFS installed

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/RavikantiAkshay/Contingency-Analysis.git
cd Contingency-Analysis

# Pull large dataset files
git lfs pull

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Key Execution Workflows

1. **Run End-to-End Demonstration:**
   Demonstrates the complete closed-loop pipeline (outage → classification → DQN remediation → verification):
   ```bash
   python run_full_pipeline.py --num-scenarios 10 --verbose
   ```

2. **Train Deep Security Classifier (100k Samples):**
   Trains the 79-dimensional PyTorch classifier:
   ```bash
   python train_classifier_100k.py
   ```

3. **Train DQN Agent & Run 3-Way Benchmark:**
   Trains the DQN agent on 4,000 scenarios (80,000 episodes) and benchmarks over 1,000 test scenarios (20,000 cases):
   ```bash
   python train_on_new_samples.py --episodes 80000 --batch-size 128
   ```

4. **Generate Large-Scale Multi-Tier Datasets:**
   Generates 100k, 200k, and 300k datasets across parallel CPU workers:
   ```bash
   python run_generate_dataset_sets.py --workers 12
   ```

---

## 13. Test Suite & Validation

The test suite is managed via `pytest` and covers every core system module:

```bash
python -m pytest tests/ -v
```

### Test Coverage (18 / 18 Tests Passing):
- **`test_network.py`**: Verifies IEEE 14-bus topology, bus admittance matrix, and generator limits.
- **`test_load_flow.py`**: Validates convergence of both the custom Newton-Raphson solver and PyPOWER on base cases.
- **`test_contingency.py`**: Validates N-1 line outages, random load perturbations, and contingency ranking severity indices (PI_V, PI_MW, CSI).
- **`test_dataset.py`**: Validates schema compliance (63 columns) and feature types.
- **`test_dataset_sets.py`**: Confirms strict zero-repetition across all 30,000 generated scenarios.
- **`test_rl.py`**: Validates the 20-action discrete space, 77-D state builder, multi-objective reward function, DQNAgent initialization/step, and classifier integration.
