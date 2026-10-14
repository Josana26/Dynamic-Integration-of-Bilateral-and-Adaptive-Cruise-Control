import os
import sys
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import TD3

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.city import City
from simulation.rl_city import RLCity
from simulation.config import DATA_DIR, PLOTS_DIR, resolve_model_path, DEFAULT_SIM_PARAMS
from env.accel_traffic_env import AccelTrafficEnv

def load_velocity_profiles(city_acc, city_bcc, city_accbcc, city_rl):
    """Loads velocity profiles from data files and assigns them to the cities."""
    ego_velocity_profile = []
    data_path = DATA_DIR / "data.csv"
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                time = float(row['time'])
                velocity = float(row['velocity'])
                ego_velocity_profile.append((time, velocity))

        print(f"Velocity profiles loaded from {data_path}")

        for c in (city_acc, city_bcc, city_accbcc, city_rl):
            c.lead_velocity_profile = ego_velocity_profile
            c.follower_velocity_profile = []

    except FileNotFoundError:
        print(f"Warning: {data_path} not found. Running without velocity profiles.")
        for c in (city_acc, city_bcc, city_accbcc, city_rl):
            c.lead_velocity_profile = []
            c.follower_velocity_profile = []

def plot_results(city_acc, city_bcc, city_accbcc, city_rl, dt, use_profiles, output_png=None):
    """Plots velocity and acceleration profiles for all four models."""
    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex='col')
    fig.suptitle('Simulation Results', fontsize=16)

    def plot_model(ax_vel, ax_acc, city, model_name):
        num_cars = len(city.cars)
        for idx, car in enumerate(city.cars):
            if idx == 0:
                continue
            time_axis = [dt * i for i in range(len(car.vel_history))]
            color = 'green' if (use_profiles and idx == 2) else 'gray'
            linewidth = 1.5 if (use_profiles and idx == 2) else 0.8
            ax_vel.plot(time_axis, car.vel_history, color=color, linewidth=linewidth)
            ax_acc.plot(time_axis, car.acc_history, color=color, linewidth=linewidth)

        if num_cars > 0:
            lead_car = city.cars[0]
            time_axis = [dt * i for i in range(len(lead_car.vel_history))]
            ax_vel.plot(time_axis, lead_car.vel_history, color='red', linewidth=1.5)
            ax_acc.plot(time_axis, lead_car.acc_history, color='red', linewidth=1.5)

        ax_vel.set_title(f"{model_name} Velocity")
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_vel.grid(True)

        ax_acc.set_title(f"{model_name} Acceleration")
        ax_acc.set_ylabel("Acceleration (m/s^2)")
        ax_acc.grid(True)

    plot_model(axes[0, 0], axes[0, 1], city_acc, "ACC")
    plot_model(axes[1, 0], axes[1, 1], city_bcc, "BCC")
    plot_model(axes[2, 0], axes[2, 1], city_accbcc, "ACC + BCC Integration")
    plot_model(axes[3, 0], axes[3, 1], city_rl, "RL Direct Acceleration")

    axes[3, 0].set_xlabel("Time (s)")
    axes[3, 1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])
    target = Path(output_png) if output_png else PLOTS_DIR / "simulation_results.png"
    plt.savefig(target)
    print(f"Saved velocity/acceleration plots to {target}")

def plot_energy_consumption(city_acc, city_bcc, city_accbcc, city_rl, output_png=None):
    """Plots the total energy consumption for each model as a bar graph."""
    total_energy_acc = sum(car.energy_used for car in city_acc.cars)
    total_energy_bcc = sum(car.energy_used for car in city_bcc.cars)
    total_energy_accbcc = sum(car.energy_used for car in city_accbcc.cars)
    total_energy_rl = sum(car.energy_used for car in city_rl.cars)

    models = ['ACC', 'BCC', 'ACC+BCC', 'RL Model']
    energy_values = [total_energy_acc, total_energy_bcc, total_energy_accbcc, total_energy_rl]

    plt.figure(figsize=(7, 6))
    plt.bar(models, energy_values, color=['lightblue'], width=0.5)
    plt.ylabel('Energy Consumption (kWh)')
    plt.title('Energy Consumption of Different Models')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    target = Path(output_png) if output_png else PLOTS_DIR / "energy_consumption.png"
    plt.savefig(target)
    print(f"Saved energy consumption plot to {target}")

def get_gap_statistics(gaps):
    """Calculates summary gap statistics."""
    if not gaps:
        print("No gap data available.")
        return
    min_gap = min(gaps)
    max_gap = max(gaps)
    mean_gap = sum(gaps) / len(gaps)
    print(f"Min gap: {min_gap:.2f} m, Max gap: {max_gap:.2f} m, Mean gap: {mean_gap:.2f} m")

def run_headless(model_path="td3_accel_agent.zip", duration=60.0):
    """Executes a 4-way comparison simulation headlessly and produces diagnostic plots."""
    USE_VELOCITY_PROFILES = True
    params = dict(DEFAULT_SIM_PARAMS)
    params['v_des'] = 25.0
    dt = params['dt']
    num_steps = int(duration / dt)

    city_acc = City()
    city_bcc = City()
    city_accbcc = City()
    city_rl = RLCity()

    init_args = [params[k] for k in ["car_number", "kd", "kv", "kc", "v_des", "max_v", "min_v", "min_dis", "reaction_time", "headway_time", "max_a", "min_a", "min_gap"]]

    city_acc.init(*init_args, dt=dt, model='ACC')
    city_bcc.init(*init_args, dt=dt, model='BCC')
    city_accbcc.init(*init_args, dt=dt, model='ACC+BCC')
    city_rl.init(*init_args, dt=dt, model='ACC')

    resolved_model = resolve_model_path(model_path)
    model = None
    if os.path.exists(resolved_model):
        model = TD3.load(resolved_model)
        print(f"Successfully loaded RL model from {resolved_model}")
    else:
        print(f"Warning: RL model not found at {resolved_model}")

    for i, car in enumerate(city_rl.cars):
        if i > 0:
            car.is_rl = True
            car.rl_action = 0.0
        else:
            car.is_rl = False

    if USE_VELOCITY_PROFILES:
        load_velocity_profiles(city_acc, city_bcc, city_accbcc, city_rl)

    rl_lead_profile = [
        (0.0,  25.0), (10.0, 25.0), (13.0, 18.0), (20.0, 18.0),
        (23.0, 28.0), (30.0, 28.0), (33.0, 15.0), (40.0, 15.0),
        (44.0, 25.0), (50.0, 25.0), (53.0, 30.0), (60.0, 30.0),
    ]
    city_rl.lead_velocity_profile = rl_lead_profile

    print(f"Running simulation for {duration} seconds ({num_steps} steps)...")
    for step in range(num_steps):
        city_acc.run(dt)
        city_bcc.run(dt)
        city_accbcc.run(dt)

        if model and city_rl.cars:
            obs = AccelTrafficEnv.get_city_obs(city_rl, v_des=params["v_des"])
            action, _ = model.predict(obs, deterministic=True)
            for i, car in enumerate(city_rl.cars[1:]):
                if i < len(action):
                    car.rl_action = float(action[i][0])
        city_rl.run(dt)

    print("Simulation complete. Generating plots...")
    plot_results(city_acc, city_bcc, city_accbcc, city_rl, dt, USE_VELOCITY_PROFILES)
    plot_energy_consumption(city_acc, city_bcc, city_accbcc, city_rl)

    print("ACC Stats:")
    get_gap_statistics(city_acc.all_gaps)
    print("\nBCC Stats:")
    get_gap_statistics(city_bcc.all_gaps)
    print("\nIntegrated Stats:")
    get_gap_statistics(city_accbcc.all_gaps)
    print("\nRL Stats:")
    get_gap_statistics(city_rl.all_gaps)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "td3_accel_agent.zip"
    run_headless(target)
