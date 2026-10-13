import os
import sys
import csv
import numpy as np
from stable_baselines3 import TD3
from simulation.city import City
from simulation.rl_city import RLCity
from env.accel_traffic_env import AccelTrafficEnv

def load_csv_profile(filename):
    profile = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile.append((float(row['time']), float(row['velocity'])))
    return profile

def get_scenarios():
    scenarios = [
        {
            "id": "steady_15",
            "name": "1. Steady Cruise (v_des=15 m/s)",
            "v_des": 15.0,
            "type": "none",
            "dur": 60.0
        },
        {
            "id": "steady_25",
            "name": "2. High-Speed Cruise (v_des=25 m/s)",
            "v_des": 25.0,
            "type": "none",
            "dur": 50.0
        },
        {
            "id": "stop_and_go_data",
            "name": "3. Urban Stop-and-Go (data.csv)",
            "v_des": 15.0,
            "type": "csv",
            "csv_file": "data/data.csv",
            "dur": 60.0
        },
        {
            "id": "speed_drop_data1",
            "name": "4. Speed Drop & Recovery (data1.csv)",
            "v_des": 15.0,
            "type": "csv",
            "csv_file": "data/data1.csv",
            "dur": 60.0
        },
        {
            "id": "full_stop_hold",
            "name": "5. Platoon Full Stop & Standstill Hold",
            "v_des": 15.0,
            "type": "synthetic",
            "profile": [(0.0, 15.0), (12.0, 15.0), (16.0, 0.0), (45.0, 0.0), (50.0, 12.0), (60.0, 15.0)],
            "dur": 60.0
        },
        {
            "id": "emergency_brake",
            "name": "6. Emergency Braking (-4.5 m/s^2 to 0)",
            "v_des": 20.0,
            "type": "synthetic",
            "profile": [(0.0, 20.0), (10.0, 20.0), (13.5, 0.0), (40.0, 0.0)],
            "dur": 40.0
        },
        {
            "id": "string_stability_wave",
            "name": "7. Sinusoidal Wave String Stability (v=15+-4m/s)",
            "v_des": 15.0,
            "type": "wave",
            "dur": 60.0
        }
    ]
    return scenarios

def evaluate_model_on_scenarios(model_path, scenarios=None):
    if scenarios is None:
        scenarios = get_scenarios()
        
    print(f"\n=======================================================")
    print(f"EVALUATING MODEL: {model_path}")
    print(f"=======================================================")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist!")
        return None
        
    model = TD3.load(model_path)
    results = {}
    
    for sc in scenarios:
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
            if i > 0:
                car.is_rl = True
                car.rl_action = 0.0
            else:
                car.is_rl = False
                
        # Assign lead vehicle profile
        lead_profile = []
        if sc['type'] == 'csv':
            lead_profile = load_csv_profile(sc['csv_file'])
        elif sc['type'] == 'synthetic':
            lead_profile = sc['profile']
        elif sc['type'] == 'wave':
            # Generate sine wave: v(t) = 15.0 + 4.0 * sin(2*pi*t / 15)
            lead_profile = [(t, 15.0 + 4.0 * np.sin(2 * np.pi * t / 15.0)) for t in np.arange(0, sc['dur'] + 1.0, 0.5)]
            
        if lead_profile:
            for c in [city_acc, city_bcc, city_accbcc, city_rl]:
                c.lead_velocity_profile = lead_profile
                
        steps = int(sc['dur'] / params['dt'])
        collisions = 0
        min_gaps_seen = []
        all_gaps = []
        rl_speeds = []
        rl_accels = []
        rl_jerks = []
        prev_acc = np.zeros(14)
        road_len = city_rl.roads[0].length
        
        follower_max_decel_rl = np.zeros(14)
        follower_max_decel_acc = np.zeros(14)
        follower_max_decel_bcc = np.zeros(14)
        follower_max_decel_accbcc = np.zeros(14)
        leader_max_decel = 0.0
        
        prev_acc_rl = np.zeros(14)
        prev_acc_acc = np.zeros(14)
        prev_acc_bcc = np.zeros(14)
        prev_acc_accbcc = np.zeros(14)
        
        jerks_rl, jerks_acc, jerks_bcc, jerks_accbcc = [], [], [], []
        gaps_rl, gaps_acc, gaps_bcc, gaps_accbcc = [], [], [], []
        collisions_acc = 0
        collisions_bcc = 0
        collisions_accbcc = 0
        
        for step in range(steps):
            city_acc.run(params['dt'])
            city_bcc.run(params['dt'])
            city_accbcc.run(params['dt'])
            
            obs_arr = AccelTrafficEnv.get_city_obs(city_rl, params['v_des'])
            act, _ = model.predict(obs_arr, deterministic=True)
            for i, car in enumerate(city_rl.cars[1:]):
                car.rl_action = float(act[i][0])
            city_rl.run(params['dt'])
            
            leader = city_rl.cars[0]
            leader_max_decel = max(leader_max_decel, max(0.0, -leader.acceleration))
            
            # Helper to collect metrics per city
            def step_metrics(city, prev_acc, follower_max_decel, jerks_list, gaps_list):
                step_gaps = []
                cur_acc = []
                for i, car in enumerate(city.cars[1:]):
                    cars_same = [c for c in city.cars if c.current_road == car.current_road and c != car]
                    def gap_to(other):
                        g = (car.pos - other.pos) % road_len
                        return g if g > 0 else float('inf')
                    front = min(cars_same, key=gap_to, default=None)
                    fg = (car.pos - front.pos - front.length) % road_len if front else road_len
                    step_gaps.append(fg)
                    gaps_list.append(fg)
                    cur_acc.append(car.acceleration)
                    follower_max_decel[i] = max(follower_max_decel[i], max(0.0, -car.acceleration))
                cur_acc = np.array(cur_acc)
                if step > 0:
                    jerks_list.extend(np.abs(cur_acc - prev_acc) / params['dt'])
                min_g = min(step_gaps)
                collided = (min_g < 0.5) or getattr(city, 'collided_this_step', False)
                return cur_acc, min_g, collided
            
            cur_acc_rl, min_g_rl, col_rl = step_metrics(city_rl, prev_acc_rl, follower_max_decel_rl, jerks_rl, gaps_rl)
            cur_acc_acc, min_g_acc, col_acc = step_metrics(city_acc, prev_acc_acc, follower_max_decel_acc, jerks_acc, gaps_acc)
            cur_acc_bcc, min_g_bcc, col_bcc = step_metrics(city_bcc, prev_acc_bcc, follower_max_decel_bcc, jerks_bcc, gaps_bcc)
            cur_acc_accbcc, min_g_accbcc, col_accbcc = step_metrics(city_accbcc, prev_acc_accbcc, follower_max_decel_accbcc, jerks_accbcc, gaps_accbcc)
            
            prev_acc_rl = cur_acc_rl
            prev_acc_acc = cur_acc_acc
            prev_acc_bcc = cur_acc_bcc
            prev_acc_accbcc = cur_acc_accbcc
            
            min_gaps_seen.append(min_g_rl)
            if col_rl: collisions += 1
            if col_acc: collisions_acc += 1
            if col_bcc: collisions_bcc += 1
            if col_accbcc: collisions_accbcc += 1
            
            for car in city_rl.cars[1:]:
                rl_speeds.append(car.velocity)
                
        e_acc = sum(c.energy_used for c in city_acc.cars)
        e_bcc = sum(c.energy_used for c in city_bcc.cars)
        e_accbcc = sum(c.energy_used for c in city_accbcc.cars)
        e_rl = sum(c.energy_used for c in city_rl.cars)
        
        # String stability damping ratios
        l_decel = max(leader_max_decel, 0.1)
        damping_rl = float(np.max(follower_max_decel_rl) / l_decel)
        damping_acc = float(np.max(follower_max_decel_acc) / l_decel)
        damping_bcc = float(np.max(follower_max_decel_bcc) / l_decel)
        damping_accbcc = float(np.max(follower_max_decel_accbcc) / l_decel)
        
        sc_res = {
            "name": sc['name'],
            "collisions_rl": collisions,
            "min_gap_rl": float(min(min_gaps_seen)),
            "mean_gap_rl": float(np.mean(gaps_rl)),
            "mean_speed_rl": float(np.mean(rl_speeds)),
            "mean_jerk_rl": float(np.mean(jerks_rl)) if jerks_rl else 0.0,
            "mean_jerk_acc": float(np.mean(jerks_acc)) if jerks_acc else 0.0,
            "mean_jerk_bcc": float(np.mean(jerks_bcc)) if jerks_bcc else 0.0,
            "mean_jerk_accbcc": float(np.mean(jerks_accbcc)) if jerks_accbcc else 0.0,
            "energy_rl": float(e_rl),
            "energy_acc": float(e_acc),
            "energy_bcc": float(e_bcc),
            "energy_accbcc": float(e_accbcc),
            "damping_rl": damping_rl,
            "damping_acc": damping_acc,
            "damping_bcc": damping_bcc,
            "damping_accbcc": damping_accbcc,
        }
        results[sc['id']] = sc_res
        
        print(f"\n--- {sc['name']} ---")
        print(f"  Collisions: RL={collisions} | ACC={collisions_acc} | BCC={collisions_bcc} | ACC+BCC={collisions_accbcc}")
        print(f"  Min Gap:    RL={sc_res['min_gap_rl']:.2f}m | Mean Gap: RL={sc_res['mean_gap_rl']:.2f}m")
        print(f"  Mean Jerk:  RL={sc_res['mean_jerk_rl']:.2f} m/s^3 | ACC={sc_res['mean_jerk_acc']:.2f} | BCC={sc_res['mean_jerk_bcc']:.2f} | ACC+BCC={sc_res['mean_jerk_accbcc']:.2f}")
        print(f"  Energy:     RL={e_rl:.3f} kWh | ACC={e_acc:.3f} | BCC={e_bcc:.3f} | ACC+BCC={e_accbcc:.3f}")
        print(f"  Shockwave:  RL={damping_rl:.2f}x | ACC={damping_acc:.2f}x | BCC={damping_bcc:.2f}x | ACC+BCC={damping_accbcc:.2f}x")
        
    return results

if __name__ == '__main__':
    target_model = sys.argv[1] if len(sys.argv) > 1 else "td3_accel_agent.zip"
    evaluate_model_on_scenarios(target_model)
