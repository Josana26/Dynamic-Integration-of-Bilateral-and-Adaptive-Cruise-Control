import sys
import os
sys.path.append('.')
from simulation.city import City
from simulation.rl_city import RLCity
from run_headless import load_velocity_profiles
from stable_baselines3 import TD3
import numpy as np
from env.accel_traffic_env import AccelTrafficEnv

params = {
    'car_number': 15, 'kd': 0.9, 'kv': 0.5, 'kc': 0.4,
    'v_des': 15.0, 'max_v': 30.0, 'min_v': 0.0,
    'min_dis': 6.0, 'reaction_time': 0.8, 'headway_time': 1.0,
    'max_a': 3.0, 'min_a': -5.0, 'min_gap': 2.0, 'dt': 0.1
}
init_args = [params[k] for k in ['car_number', 'kd', 'kv', 'kc', 'v_des', 'max_v', 'min_v', 'min_dis', 'reaction_time', 'headway_time', 'max_a', 'min_a', 'min_gap']]

city_acc = City()
city_acc.init(*init_args, dt=params['dt'], model='ACC')
city_bcc = City()
city_bcc.init(*init_args, dt=params['dt'], model='BCC')
city_accbcc = City()
city_accbcc.init(*init_args, dt=params['dt'], model='ACC+BCC')
city_rl = RLCity()
city_rl.init(*init_args, dt=params['dt'], model='ACC')

model = TD3.load('td3_accel_agent.zip')

for i, car in enumerate(city_rl.cars):
    if i > 0: car.is_rl = True; car.rl_action = 0.0
    else: car.is_rl = False

load_velocity_profiles(city_acc, city_bcc, city_accbcc, city_rl)
city_rl.lead_velocity_profile = [
    (0.0, 25.0), (10.0, 25.0), (13.0, 18.0), (20.0, 18.0), 
    (23.0, 28.0), (30.0, 28.0), (33.0, 15.0), (40.0, 15.0), 
    (44.0, 25.0), (50.0, 25.0), (53.0, 30.0), (60.0, 30.0)
]

def _get_rl_observation(city, v_des=15.0):
    return AccelTrafficEnv.get_city_obs(city, v_des)



num_steps = int(60 / params['dt'])
for _ in range(num_steps):
    city_acc.run(params['dt'])
    city_bcc.run(params['dt'])
    city_accbcc.run(params['dt'])
    
    obs = _get_rl_observation(city_rl, v_des=params['v_des'])
    action, _ = model.predict(obs, deterministic=True)
    for i, car in enumerate(city_rl.cars[1:]):
        car.rl_action = float(action[i][0])
    city_rl.run(params['dt'])

print('ACC Energy:', sum(car.energy_used for car in city_acc.cars))
print('BCC Energy:', sum(car.energy_used for car in city_bcc.cars))
print('ACC+BCC Energy:', sum(car.energy_used for car in city_accbcc.cars))
print('RL Energy:', sum(car.energy_used for car in city_rl.cars))
