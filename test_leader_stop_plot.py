import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import TD3

from simulation.city import City
from simulation.rl_city import RLCity
from env.accel_traffic_env import AccelTrafficEnv

def simulate_and_plot(model_path="td3_accel_agent.zip", output_png="scratch_leader_stop_restart_comparison.png"):
    model = TD3.load(model_path)
    
    dt = 0.1
    car_number = 15
    kd = 0.9
    kv = 0.5
    kc = 0.4
    v_des = 15.0
    max_v = 30.0
    min_v = 0.0
    min_dis = 6.0
    reaction_time = 0.8
    headway_time = 1.0
    max_a = 3.0
    min_a = -5.0
    min_gap = 2.0
    
    init_params = [car_number, kd, kv, kc, v_des, max_v, min_v, min_dis, reaction_time, headway_time, max_a, min_a, min_gap]
    
    city_acc = City()
    city_acc.init(*init_params, dt=dt, model='ACC')
    city_bcc = City()
    city_bcc.init(*init_params, dt=dt, model='BCC')
    city_accbcc = City()
    city_accbcc.init(*init_params, dt=dt, model='ACC+BCC')
    city_rl = RLCity()
    city_rl.init(*init_params, dt=dt, model='ACC')
    
    for i, car in enumerate(city_rl.cars):
        if i > 0:
            car.is_rl = True
            car.rl_action = 0.0
        else:
            car.is_rl = False
            car.mode = 'ACC'
            
    total_time = 42.0
    steps = int(total_time / dt)
    t_stop = 4.0
    t_restart = 18.0
    
    # Track inter-vehicle gaps for each model
    cities = [
        ("ACC", city_acc),
        ("BCC", city_bcc),
        ("ACC+BCC Integration", city_accbcc),
        ("RL Direct Acceleration", city_rl)
    ]
    gap_histories = {name: [[] for _ in range(car_number - 1)] for name, _ in cities}
    
    for step in range(steps):
        t = step * dt
        leader_stop = (t_stop <= t < t_restart)
        
        city_acc.set_leader_stop(leader_stop)
        city_bcc.set_leader_stop(leader_stop)
        city_accbcc.set_leader_stop(leader_stop)
        city_rl.set_leader_stop(leader_stop)
        
        city_acc.run(dt)
        city_bcc.run(dt)
        city_accbcc.run(dt)
        
        obs = AccelTrafficEnv.get_city_obs(city_rl, v_des)
        act, _ = model.predict(obs, deterministic=True)
        for i, car in enumerate(city_rl.cars[1:]):
            car.rl_action = float(act[i][0])
            
        city_rl.run(dt)
        
        # Record gaps
        for name, city in cities:
            road_len = city.roads[0].length
            for i in range(1, car_number):
                car = city.cars[i]
                front = city.cars[i - 1]
                g = (car.pos - front.pos - front.length) % road_len
                gap_histories[name][i - 1].append(g)
        
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex='col')
    
    for row, (name, city) in enumerate(cities):
        # 1. Velocity Profile
        for idx, car in enumerate(city.cars):
            time_axis = [dt * i for i in range(len(car.vel_history))]
            color = "red" if idx == 0 else "green" if idx == 2 else "dimgrey"
            linewidth = 1.8 if idx in (0, 2) else 0.6
            axes[row, 0].plot(time_axis, car.vel_history, color=color, linewidth=linewidth, alpha=0.9)
        axes[row, 0].axvline(t_stop, color='darkorange', linestyle='--', alpha=0.7, label='Stop Lead' if row==0 else None)
        axes[row, 0].axvline(t_restart, color='blue', linestyle='--', alpha=0.7, label='Restart Lead' if row==0 else None)
        axes[row, 0].set_title(f"{name} Velocity")
        axes[row, 0].set_ylabel("Velocity (m/s)")
        axes[row, 0].grid(True, alpha=0.3)
        if row == 0: axes[row, 0].legend(loc='upper right', fontsize=8)
        if row == 3: axes[row, 0].set_xlabel("Time (s)")
        
        # 2. Acceleration Profile
        for idx, car in enumerate(city.cars):
            time_axis = [dt * i for i in range(len(car.acc_history))]
            color = "red" if idx == 0 else "green" if idx == 2 else "dimgrey"
            linewidth = 1.8 if idx in (0, 2) else 0.6
            axes[row, 1].plot(time_axis, car.acc_history, color=color, linewidth=linewidth, alpha=0.9)
        axes[row, 1].axvline(t_stop, color='darkorange', linestyle='--', alpha=0.7)
        axes[row, 1].axvline(t_restart, color='blue', linestyle='--', alpha=0.7)
        axes[row, 1].set_title(f"{name} Acceleration")
        axes[row, 1].set_ylabel("Accel (m/s²)")
        axes[row, 1].grid(True, alpha=0.3)
        if row == 3: axes[row, 1].set_xlabel("Time (s)")

        # 3. Headway Gap Profile
        time_axis = [dt * i for i in range(steps)]
        for i_follower, g_hist in enumerate(gap_histories[name]):
            color = "green" if i_follower == 1 else "dimgrey"
            linewidth = 1.8 if i_follower == 1 else 0.6
            axes[row, 2].plot(time_axis, g_hist, color=color, linewidth=linewidth, alpha=0.9)
        axes[row, 2].axhline(min_dis, color='crimson', linestyle=':', label='d_min (6m)' if row==0 else None)
        axes[row, 2].axvline(t_stop, color='darkorange', linestyle='--', alpha=0.7)
        axes[row, 2].axvline(t_restart, color='blue', linestyle='--', alpha=0.7)
        axes[row, 2].set_title(f"{name} Follower Gaps")
        axes[row, 2].set_ylabel("Gap (m)")
        axes[row, 2].grid(True, alpha=0.3)
        if row == 0: axes[row, 2].legend(loc='upper right', fontsize=8)
        if row == 3: axes[row, 2].set_xlabel("Time (s)")
        
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Saved plot to {output_png}")

if __name__ == '__main__':
    simulate_and_plot()
