import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import TD3

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.accel_traffic_env import AccelTrafficEnv
from simulation.config import PLOTS_DIR, resolve_model_path, DEFAULT_SIM_PARAMS

def evaluate(model_path="td3_accel_agent.zip", output_plot=None):
    """
    Evaluates policy model in environment rollout and generates
    speed, acceleration, gap, and reward analytics plots.
    """
    resolved_model = resolve_model_path(model_path)
    if not os.path.exists(resolved_model):
        print(f"Model not found at {resolved_model}!")
        return
        
    env = AccelTrafficEnv()
    model = TD3.load(resolved_model)
    
    # Run evaluation
    obs = env.reset()
    
    # Override with GUI defaults for this evaluation
    gui_params = {
        'car_number': 15, 'kd': 0.9, 'kv': 0.5, 'kc': 0.4,
        'v_des': 15.0, 'max_v': 30.0, 'min_v': 0.0,
        'min_dis': 6.0, 'reaction_time': 0.8, 'max_a': 3.0, 'min_a': -5.0
    }
    env.sim_params.update(gui_params)
    
    init_args = [
        env.num_cars, env.sim_params['kd'], env.sim_params['kv'], env.sim_params['kc'],
        env.sim_params['v_des'], env.sim_params['max_v'], env.sim_params['min_v'],
        env.sim_params['min_dis'], env.sim_params['reaction_time'], 1.0,
        env.sim_params['max_a'], env.sim_params['min_a'], 2.0
    ]
    env.city.init(*init_args, dt=env.dt, model='ACC')
    for c in env.city.cars[1:]:
        c.is_rl = True
        c.rl_action = 0.0
    obs = env._get_obs()
    
    steps = 500
    speeds = []
    accels = []
    gaps = []
    jerks = []
    rewards = []
    
    dt = env.dt
    prev_accel = np.zeros(env.num_cars)
    
    for _ in range(steps):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards_step, dones, _ = env.step(actions)
        
        total_step_reward = float(np.sum(rewards_step))
        rewards.append(total_step_reward)
        
        current_speeds = []
        current_accels = []
        current_gaps = []
        current_jerks = []
        
        cars = env.city.cars
        road_len = env.city.roads[0].length if env.city.roads else 1000
        
        for i, car in enumerate(cars[1:], 1):
            current_speeds.append(car.velocity)
            current_accels.append(car.acceleration)
            
            j = (car.acceleration - prev_accel[i]) / dt
            current_jerks.append(j)
            
            cars_same_road = [c for c in cars if c.current_road == car.current_road and c != car]
            def gap_to(other):
                g = (car.pos - other.pos) % road_len
                return g if g > 0 else float('inf')
            front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None
            
            gap = (car.pos - front_car.pos - front_car.length) % road_len if front_car else road_len
            current_gaps.append(gap)
            
        speeds.append(current_speeds)
        accels.append(current_accels)
        gaps.append(current_gaps)
        jerks.append(current_jerks)
        
        prev_accel = np.array([c.acceleration for c in cars])
        if any(dones):
            break
            
    speeds = np.array(speeds)
    accels = np.array(accels)
    gaps = np.array(gaps)
    jerks = np.array(jerks)
    
    print("=== Analytics Report ===")
    print(f"Steps survived: {len(speeds)}")
    if len(speeds) > 0:
        print(f"Mean Speed: {np.mean(speeds):.2f} m/s (Target: {env.sim_params['v_des']})")
        print(f"Mean Acceleration: {np.mean(accels):.2f} m/s^2")
        print(f"Mean Gap: {np.mean(gaps):.2f} m")
        print(f"Mean Jerk: {np.mean(np.abs(jerks)):.2f} m/s^3")
        print(f"Total Reward: {np.sum(rewards):.2f}")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for i in range(min(5, speeds.shape[1])):
        plt.plot(speeds[:, i], label=f'Car {i}')
    plt.axhline(env.sim_params['v_des'], color='r', linestyle='--', label='Target Speed')
    plt.title("Vehicle Speeds Over Time")
    plt.xlabel("Step")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for i in range(min(5, accels.shape[1])):
        plt.plot(accels[:, i], label=f'Car {i}')
    plt.title("Vehicle Accelerations Over Time")
    plt.xlabel("Step")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for i in range(min(5, gaps.shape[1])):
        plt.plot(gaps[:, i], label=f'Car {i}')
    plt.title("Front Gap Over Time")
    plt.xlabel("Step")
    plt.ylabel("Gap (m)")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(rewards)
    plt.title("Step Reward Over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    
    plt.tight_layout()
    
    plot_target = Path(output_plot) if output_plot else PLOTS_DIR / "analytics_plot.png"
    plt.savefig(plot_target)
    print(f"Plot saved as {plot_target}")

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else "td3_accel_agent.zip"
    evaluate(target)
