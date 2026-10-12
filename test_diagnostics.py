import os
import sys
import csv
import numpy as np
from stable_baselines3 import TD3
from simulation.city import City
from simulation.rl_city import RLCity
from env.accel_traffic_env import AccelTrafficEnv

model = TD3.load("td3_accel_agent.zip")

scenarios = [
    {"name": "GUI Defaults (v_des=15, no profile)", "v_des": 15.0, "profile": False, "dur": 60},
    {"name": "GUI Defaults (v_des=15, with data.csv)", "v_des": 15.0, "profile": True, "dur": 60},
    {"name": "High Speed (v_des=25, no profile)", "v_des": 25.0, "profile": False, "dur": 35},
    {"name": "High Speed (v_des=25, with data.csv)", "v_des": 25.0, "profile": True, "dur": 60},
]

profile_data = []
if os.path.exists("data/data.csv"):
    with open("data/data.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profile_data.append((float(row['time']), float(row['velocity'])))

for sc in scenarios:
    print(f"\n=== Scenario: {sc['name']} ===")
    params = {
        'car_number': 15, 'kd': 0.9, 'kv': 0.5, 'kc': 0.4,
        'v_des': sc['v_des'], 'max_v': 30.0, 'min_v': 0.0,
        'min_dis': 6.0, 'reaction_time': 0.8, 'headway_time': 1.0,
        'max_a': 3.0, 'min_a': -5.0, 'min_gap': 2.0, 'dt': 0.1
    }
    init_args = [params[k] for k in ['car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v', 'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap']]
    city_rl = RLCity()
    city_rl.init(*init_args, dt=params['dt'], model='ACC')
    city_acc = City()
    city_acc.init(*init_args, dt=params['dt'], model='ACC')
    city_bcc = City()
    city_bcc.init(*init_args, dt=params['dt'], model='BCC')
    city_accbcc = City()
    city_accbcc.init(*init_args, dt=params['dt'], model='ACC+BCC')

    for i, car in enumerate(city_rl.cars):
        if i > 0: car.is_rl = True; car.rl_action = 0.0
        else: car.is_rl = False

    if sc['profile']:
        for c in [city_acc, city_bcc, city_accbcc, city_rl]:
            c.lead_velocity_profile = profile_data

    steps = int(sc['dur'] / params['dt'])
    collisions = 0
    min_gaps_seen = []
    rl_speeds = []
    rl_accels = []
    rl_jerks = []
    prev_acc = np.zeros(14)
    road_len = city_rl.roads[0].length

    for step in range(steps):
        city_acc.run(params['dt'])
        city_bcc.run(params['dt'])
        city_accbcc.run(params['dt'])
        
        obs_arr = AccelTrafficEnv.get_city_obs(city_rl, params['v_des'])
        act, _ = model.predict(obs_arr, deterministic=True)
        for i, car in enumerate(city_rl.cars[1:]): car.rl_action = float(act[i][0])
        city_rl.run(params['dt'])

        step_gaps = []
        cur_acc = []
        for car in city_rl.cars[1:]:
            cars_same = [c for c in city_rl.cars if c.current_road == car.current_road and c != car]
            def gap_to(other): g = (car.pos - other.pos) % road_len; return g if g > 0 else float('inf')
            front = min(cars_same, key=gap_to, default=None)
            fg = (car.pos - front.pos - front.length) % road_len if front else road_len
            step_gaps.append(fg)
            rl_speeds.append(car.velocity)
            cur_acc.append(car.acceleration)
            
        cur_acc = np.array(cur_acc)
        if step > 0:
            rl_jerks.extend(np.abs(cur_acc - prev_acc) / params['dt'])
        prev_acc = cur_acc
        rl_accels.extend(cur_acc)
        
        min_g = min(step_gaps)
        min_gaps_seen.append(min_g)
        if min_g < 0.5: collisions += 1

    print(f"Energy: ACC={sum(c.energy_used for c in city_acc.cars):.3f} kWh | "
          f"BCC={sum(c.energy_used for c in city_bcc.cars):.3f} kWh | "
          f"ACC+BCC={sum(c.energy_used for c in city_accbcc.cars):.3f} kWh | "
          f"RL={sum(c.energy_used for c in city_rl.cars):.3f} kWh")
    print(f"Follower Min Gap: {min(min_gaps_seen):.3f} m | Mean Min Gap: {np.mean(min_gaps_seen):.3f} m")
    print(f"RL Collision steps: {collisions}")
    print(f"RL Follower Mean Speed: {np.mean(rl_speeds):.2f} m/s (Target: {sc['v_des']:.1f}) | Mean Jerk: {np.mean(rl_jerks):.2f} m/s^3")
    print(f"Lead Car final speed: {city_rl.cars[0].velocity:.2f} m/s | Follower final avg speed: {np.mean([c.velocity for c in city_rl.cars[1:]]):.2f} m/s")
