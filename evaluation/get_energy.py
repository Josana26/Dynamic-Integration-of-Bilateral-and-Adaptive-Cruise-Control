import os
import sys
from pathlib import Path
from stable_baselines3 import TD3
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.city import City
from simulation.rl_city import RLCity
from simulation.config import DEFAULT_SIM_PARAMS, resolve_model_path
from evaluation.run_headless import load_velocity_profiles
from env.accel_traffic_env import AccelTrafficEnv

def calculate_energy(model_path="td3_accel_agent.zip", duration=60.0):
    """Calculates total energy consumption across ACC, BCC, ACC+BCC, and RL models."""
    params = dict(DEFAULT_SIM_PARAMS)
    init_args = [params[k] for k in ['car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v', 'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap']]

    city_acc = City()
    city_acc.init(*init_args, dt=params['dt'], model='ACC')
    city_bcc = City()
    city_bcc.init(*init_args, dt=params['dt'], model='BCC')
    city_accbcc = City()
    city_accbcc.init(*init_args, dt=params['dt'], model='ACC+BCC')
    city_rl = RLCity()
    city_rl.init(*init_args, dt=params['dt'], model='ACC')

    resolved_model = resolve_model_path(model_path)
    model = TD3.load(resolved_model)

    for i, car in enumerate(city_rl.cars):
        if i > 0:
            car.is_rl = True
            car.rl_action = 0.0
        else:
            car.is_rl = False

    load_velocity_profiles(city_acc, city_bcc, city_accbcc, city_rl)
    city_rl.lead_velocity_profile = [
        (0.0, 25.0), (10.0, 25.0), (13.0, 18.0), (20.0, 18.0), 
        (23.0, 28.0), (30.0, 28.0), (33.0, 15.0), (40.0, 15.0), 
        (44.0, 25.0), (50.0, 25.0), (53.0, 30.0), (60.0, 30.0)
    ]

    num_steps = int(duration / params['dt'])
    for _ in range(num_steps):
        city_acc.run(params['dt'])
        city_bcc.run(params['dt'])
        city_accbcc.run(params['dt'])
        
        obs = AccelTrafficEnv.get_city_obs(city_rl, v_des=params['v_des'])
        action, _ = model.predict(obs, deterministic=True)
        for i, car in enumerate(city_rl.cars[1:]):
            car.rl_action = float(action[i][0])
        city_rl.run(params['dt'])

    energies = {
        'ACC': sum(car.energy_used for car in city_acc.cars),
        'BCC': sum(car.energy_used for car in city_bcc.cars),
        'ACC+BCC': sum(car.energy_used for car in city_accbcc.cars),
        'RL': sum(car.energy_used for car in city_rl.cars)
    }

    print('ACC Energy:    ', f"{energies['ACC']:.4f} kWh")
    print('BCC Energy:    ', f"{energies['BCC']:.4f} kWh")
    print('ACC+BCC Energy:', f"{energies['ACC+BCC']:.4f} kWh")
    print('RL Energy:     ', f"{energies['RL']:.4f} kWh")
    return energies

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "td3_accel_agent.zip"
    calculate_energy(target)
