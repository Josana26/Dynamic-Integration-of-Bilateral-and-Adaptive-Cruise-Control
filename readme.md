# Dynamic Integration of Bilateral and Adaptive Cruise Control with Multi-Agent Reinforcement Learning

A high-fidelity traffic simulation and control engineering framework comparing classical **Adaptive Cruise Control (ACC)**, **Bilateral Cruise Control (BCC)**, **Dynamic ACC+BCC Integration**, and a decentralized **Twin Delayed Deep Deterministic Policy Gradient (TD3)** neural network controller for autonomous vehicle platoons.

---

## 1. Project Directory Structure

The codebase is organized into modular packages:

```text
Dynamic-Integration-of-Bilateral-and-Adaptive-Cruise-Control/
│
├── simulation/             # Physics engine, circular track geometry, and vehicle kinematics
│   ├── __init__.py
│   ├── car.py             # Physical vehicle dynamics, rolling resistance & aerodynamic drag
│   ├── road.py            # Circular highway boundary definitions
│   ├── city.py            # Simulation manager for ACC, BCC, and ACC+BCC classical controllers
│   ├── rl_city.py         # Subclass integrating direct RL acceleration actuation (zero heuristics)
│   └── config.py          # Centralized path discovery and default simulation constants
│
├── env/                    # Reinforcement Learning Gym environment
│   ├── __init__.py
│   └── accel_traffic_env.py # Decentralized multi-agent VecEnv with 24 reward function terms
│
├── gui/                    # Graphical user interfaces & live visualization
│   ├── __init__.py
│   ├── control_window.py  # Base Tkinter control dashboard
│   ├── rl_control_window.py # Full 4-model engineering comparison dashboard with live telemetry
│   ├── transportation_painter.py # Multi-lane canvas renderer
│   └── video_recorder.py  # Headless and GUI canvas video recorder
│
├── training/               # Policy training pipelines & callbacks
│   ├── __init__.py
│   └── train_td3_accel.py # TD3 training script with analytical equilibrium calibration
│
├── evaluation/             # Benchmarks, headless simulations, and analytics
│   ├── __init__.py
│   ├── evaluate_all_scenarios.py # Official 7 operational benchmark scenarios suite
│   ├── evaluate_analytics.py     # Trajectory rollout metrics and analytics plotting
│   ├── run_headless.py           # Multi-model 4-way headless comparison runner
│   └── get_energy.py             # Quantitative kWh traction energy consumption analysis
│
├── tests/                  # Verification and diagnostic test suites
│   ├── __init__.py
│   ├── test_stop_and_resume.py   # 15-vehicle stop-and-resume zero-heuristic test
│   ├── test_leader_stop_plot.py  # Emergency braking and shockwave propagation test
│   └── test_diagnostics.py       # Rapid scenario sanity checks across speed presets
│
├── tools/                  # Developer utilities and telemetry loggers
│   ├── __init__.py
│   └── log_reward_breakdown.py   # Generates detailed 24-reward runtime logs & CSV telemetry
│
├── data/                   # Real-world and synthetic driving cycle profiles
│   ├── data.csv            # Urban stop-and-go speed disturbance profile
│   ├── data1.csv           # Highway speed drop and recovery profile
│   └── data2.csv           # Follower disturbance cycle profile
│
├── docs/                   # Formal technical specifications and experiment logs
│   ├── l_training_log.md         # Comprehensive history of all 16 training phases and metrics
│   ├── reward_functions_log.md   # Mathematical specification of all 24 reward components
│   └── INSTRUCTIONS.md           # Operational instructions and setup guide
│
├── logs/                   # Execution logs and telemetry output datasets
│   ├── reward_breakdown_log.txt  # Human-readable step-by-step reward runtime log
│   └── reward_telemetry.csv      # Complete 3,000+ sample CSV telemetry dataset
│
├── plots/                  # Generated figures, plots, and trajectory charts
│   ├── analytics_plot.png
│   ├── energy_consumption.png
│   ├── leader_stop_restart_comparison.png
│   └── simulation_results.png
│
├── models/                 # Saved policy checkpoints and neural network weights
│   ├── td3_accel_agent.zip       # Active TD3 policy weights (TD3_49)
│   ├── td3_accel_agent_vecnormalize.pkl # Running normalization statistics
│   └── checkpoints/              # Periodic intermediate training checkpoints
│
├── main.py                 # Single unified CLI / GUI entrypoint
├── requirements.txt        # Pinned Python package dependencies
└── README.md               # Project documentation
```

---

## 2. Installation & Dependencies

Ensure you have Python 3.10+ installed. Install all verified dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
* `stable-baselines3>=2.3.0` & `gymnasium>=1.0.0` (Reinforcement Learning)
* `torch>=2.0.0` (Neural Network Backend)
* `numpy>=1.24.0` & `pandas>=2.0.0` (Kinematic calculations and telemetry data)
* `matplotlib>=3.7.0` (Engineering dashboard plots)
* `opencv-python>=4.8.0` (Simulation video recording)

---

## 3. Quick Start: Unified CLI (`main.py`)

All project tasks can be executed via the unified `main.py` entrypoint from the project root:

| Command | Action |
| :--- | :--- |
| `python main.py` or `python main.py --gui` | Launch interactive Tkinter 4-model engineering dashboard |
| `python main.py --eval` | Run official 7 operational scenario benchmarks |
| `python main.py --train` | Train decentralized TD3 direct acceleration policy |
| `python main.py --headless` | Run headless multi-model comparison simulation |
| `python main.py --energy` | Run quantitative kWh energy consumption analysis |
| `python main.py --analytics` | Run policy rollout analytics and generate diagnostic plots |
| `python main.py --test` | Run automated platoon stop-and-resume verification suite |
| `python main.py --rewards` | Generate 24-component reward function audit log and CSV |

---

## 4. Key Architectural Features

### 1. Zero-Heuristic RL Actuation
* All follower vehicles (`indices 1..N-1`) actuate direct continuous longitudinal acceleration commands (`target_acc = float(car.rl_action)`).
* All legacy programmatic heuristics (`launch_acc`, `catchup_acc`, `leader_stopping`) were eliminated from `simulation/rl_city.py`.
* The policy natively masters complete queue stops, stationary auto-hold at 6.0 m queue spacing, decisive drive-off launches, and emergency collision avoidance directly from reinforcement learning.

### 2. Physically-Convex 24-Component Reward Function
The reward function in `env/accel_traffic_env.py` contains 24 distinct terms structured into 8 functional groups:
1. **Safety Boundary**: Progressive barrier potential (`r_danger`) and emergency stopping reward (`r_safety_brake`).
2. **Headway Tracking**: Gaussian sweet-spot reward (`r_headway`), anti-lag penalty (`r_lag`), and catch-up reward (`r_catchup`).
3. **Speed Regulation**: Continuous blended target speed matching (`r_speed`) and linear cruise pull (`r_speed_linear`).
4. **String Stability**: Downstream deceleration wave damping (`r_string`) and anti-panic braking penalty (`r_panic_brake`).
5. **Standstill Precision**: Queue gap spacing (`r_standstill_gap`), stationary auto-hold brake (`r_standstill_hold`), and native launch reward (`r_launch`).
6. **Bilateral Consensus (BCC)**: Rear tailgating protection (`r_bcc_tailgate`) and lattice midpoint speed consensus (`r_bcc_consensus`).
7. **Passenger Comfort**: Acceleration centering (`r_accel`), jerk minimization (`r_jerk`), action rate penalty (`r_act_delta`), and zero-drift cruising lock (`r_cruise_lock`).
8. **Traction Efficiency**: Traction power penalty (`r_energy`).

### 3. Comprehensive Benchmark Verification
The active policy (`TD3_49`) demonstrates **0 collisions** across all 7 operational benchmarks:
* **Scenario 1 (15 m/s Cruise)**: 0 collisions, 1.322 kWh energy.
* **Scenario 2 (25 m/s High Speed)**: 0 collisions, 2.650 kWh energy.
* **Scenario 3 (Urban Stop-and-Go)**: 0 collisions, 0.93× shockwave attenuation, 1.951 kWh.
* **Scenario 4 (Speed Drop & Recovery)**: 0 collisions, 8.27 m min gap.
* **Scenario 5 (Full Stop & Standstill Hold)**: 0 collisions, 5.78 m min queue gap, 0.89× shockwave.
* **Scenario 6 (Emergency Stop -4.5 m/s²)**: 0 collisions, 4.15 m min gap, 0.98× shockwave.
* **Scenario 7 (Sinusoidal String Stability)**: 0 collisions, 0.95× shockwave attenuation vs ACC's 1.85×, 1.735 kWh vs ACC's 5.183 kWh (66.5% energy reduction).
