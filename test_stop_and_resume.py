import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import TD3

from simulation.rl_city import RLCity
from env.accel_traffic_env import AccelTrafficEnv

def test_stop_and_resume(model_path="td3_accel_agent.zip", output_png="leader_stop_restart_comparison.png"):
    print(f"Loading model from {model_path}...")
    model = TD3.load(model_path)
    dt = 0.1
    params = [15, 0.9, 0.5, 0.4, 15.0, 30.0, 0.0, 6.0, 0.8, 1.0, 3.0, -5.0, 2.0]
    
    city = RLCity()
    city.init(*params, dt=dt, model='ACC')
    for i, car in enumerate(city.cars):
        if i > 0:
            car.is_rl = True
            car.rl_action = 0.0
        else:
            car.is_rl = False
            car.mode = 'ACC'
            
    road_len = city.roads[0].length

    print("\n--- Phase 1: Bringing all vehicles to complete standstill (leader_stop=True) ---")
    city.set_leader_stop(True)
    t_records = []
    
    for step in range(130):
        t = round(step * dt, 2)
        obs = AccelTrafficEnv.get_city_obs(city, 15.0)
        act, _ = model.predict(obs, deterministic=True)
        for i, car in enumerate(city.cars[1:]):
            car.rl_action = float(act[i][0])
        city.run(dt)
        t_records.append(t)
        
        if step in [0, 25, 50, 80, 129]:
            v0 = city.cars[0].velocity
            v1 = city.cars[1].velocity
            v2 = city.cars[2].velocity
            g1 = (city.cars[1].pos - city.cars[0].pos - city.cars[0].length) % road_len
            g2 = (city.cars[2].pos - city.cars[1].pos - city.cars[1].length) % road_len
            print(f"t={t:4.1f}s | Lead v={v0:4.1f} | C1: v={v1:4.1f}, gap={g1:4.1f}m, act={city.cars[1].rl_action:+5.2f} | C2: v={v2:4.1f}, gap={g2:4.1f}m")

    standstill_vels = [round(c.velocity, 2) for c in city.cars[:6]]
    standstill_gaps = [round((city.cars[i].pos - city.cars[i-1].pos - city.cars[i-1].length) % road_len, 2) for i in range(1, 6)]
    print(f"\nStandstill reached! Velocities: {standstill_vels} | Gaps: {standstill_gaps}")
    
    print("\n--- Phase 2: Resuming leader (leader_stop=False) - Native Model Launch ---")
    city.set_leader_stop(False)
    
    for step in range(250):
        t = round((130 + step) * dt, 2)
        t_post = round(step * dt, 1)
        obs = AccelTrafficEnv.get_city_obs(city, 15.0)
        act, _ = model.predict(obs, deterministic=True)
        for i, car in enumerate(city.cars[1:]):
            car.rl_action = float(act[i][0])
        city.run(dt)
        t_records.append(t)
        
        if t_post in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 24.9]:
            v0 = city.cars[0].velocity
            v1 = city.cars[1].velocity
            v2 = city.cars[2].velocity
            v7 = city.cars[7].velocity
            v14 = city.cars[14].velocity
            g1 = (city.cars[1].pos - city.cars[0].pos - city.cars[0].length) % road_len
            g2 = (city.cars[2].pos - city.cars[1].pos - city.cars[1].length) % road_len
            print(f"+{t_post:4.1f}s | Lead v={v0:4.1f} | C1: v={v1:4.1f}, gap={g1:4.1f}m, act={city.cars[1].rl_action:+5.2f} | C2: v={v2:4.1f}, gap={g2:4.1f}m | C7: v={v7:4.1f} | C14: v={v14:4.1f}")

    # Plot velocity, acceleration, and inter-vehicle gaps
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    time_axis = [i * dt for i in range(len(city.cars[0].vel_history))]
    
    # 1. Velocity
    for idx, car in enumerate(city.cars):
        color = "red" if idx == 0 else "green" if idx in (1, 2) else "blue" if idx == 14 else "grey"
        lw = 2.0 if idx in (0, 1, 2) else 1.0
        label = f"Car {idx} ({'Lead' if idx==0 else 'Ego' if idx==1 else 'Follower'})" if idx in (0, 1, 2, 14) else None
        axes[0].plot(time_axis, car.vel_history, color=color, linewidth=lw, label=label)
    axes[0].axvline(x=13.0, color="black", linestyle="--", alpha=0.7, label="Leader Resumes (t=13s)")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Platoon Stop-and-Resume Profile (Native Neural Network Policy, Zero Heuristics)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")
    
    # 2. Acceleration
    for idx, car in enumerate(city.cars):
        color = "red" if idx == 0 else "green" if idx in (1, 2) else "blue" if idx == 14 else "grey"
        lw = 2.0 if idx in (0, 1, 2) else 1.0
        axes[1].plot(time_axis, car.acc_history, color=color, linewidth=lw)
    axes[1].axvline(x=13.0, color="black", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    # 3. Inter-Vehicle Gap
    for idx in range(1, len(city.cars)):
        gaps = []
        for step_idx in range(len(city.cars[0].pos_history)):
            p_self = city.cars[idx].pos_history[step_idx]
            p_front = city.cars[idx-1].pos_history[step_idx]
            gap = (p_self - p_front - city.cars[idx-1].length) % road_len
            gaps.append(gap)
        color = "green" if idx in (1, 2) else "grey"
        lw = 2.0 if idx in (1, 2) else 0.8
        label = f"Gap {idx}->{idx-1}" if idx in (1, 2) else None
        axes[2].plot(time_axis, gaps, color=color, linewidth=lw, label=label)
    axes[2].axhline(y=6.0, color="orange", linestyle=":", linewidth=1.5, label="min_dis (6.0m)")
    axes[2].axvline(x=13.0, color="black", linestyle="--", alpha=0.7)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Gap (m)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"\nSaved evaluation plot to {output_png}")

if __name__ == '__main__':
    test_stop_and_resume()
