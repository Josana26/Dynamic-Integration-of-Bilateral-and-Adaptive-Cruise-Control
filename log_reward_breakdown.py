import os
import sys
import csv
import numpy as np
from stable_baselines3 import TD3

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from simulation.rl_city import RLCity
from env.accel_traffic_env import AccelTrafficEnv

def generate_reward_breakdown_log(
    model_path="td3_accel_agent.zip",
    output_log="reward_breakdown_log.txt",
    output_csv="reward_telemetry.csv"
):
    """
    Executes a high-fidelity 5-phase platoon simulation and records every single
    input feature and evaluated reward value for all 24 reward function terms.
    """
    print(f"Loading policy model from {model_path}...")
    model = TD3.load(model_path)
    env = AccelTrafficEnv()
    
    city = RLCity()
    # 15 cars, v_des=15.0, max_v=30.0, min_v=0.0, min_dis=6.0, reaction_time=0.8, max_a=3.0, min_a=-5.0
    params = [15, 0.9, 0.5, 0.4, 15.0, 30.0, 0.0, 6.0, 0.8, 1.0, 3.0, -5.0, 2.0]
    dt = 0.1
    city.init(*params, dt=dt, model='ACC')
    for i, c in enumerate(city.cars):
        c.is_rl = (i > 0)
        c.rl_action = 0.0
    
    # Attach city to env
    env.city = city
    env.sim_params['v_des'] = 15.0
    env.prev_accel = np.zeros(15)
    env.prev_action = np.zeros(14)
    env.action_deltas = np.zeros(14)
    env.in_collision = np.zeros(14, dtype=bool)
    
    # Text Log Lines
    lines = []
    lines.append("=" * 170)
    lines.append("        REINFORCEMENT LEARNING REWARD FUNCTION INPUTS & OUTPUTS RUNTIME LOG (WITH EXACT VALUES)")
    lines.append(f"Model: {model_path} | Environment: AccelTrafficEnv | Followers: 14 RL Agents | Timestep dt = {dt}s")
    lines.append("=" * 170)
    lines.append("")
    lines.append("SUMMARY OF ALL 24 REWARD COMPONENTS LOGGED:")
    lines.append("  Safety & Collision : r_col (Collision), r_dng (Danger Boundary), r_safe (Emergency Braking)")
    lines.append("  Headway Regulation : r_head (Safe Headway), r_lag (Anti-Lag), r_catch (Open Catch-Up), r_open_h (Open Hesitation)")
    lines.append("  Speed & Progress   : r_v_mtch (Relative Vel Match), r_cls_dmp (Closing Damp), r_spd (Speed Gaussian), r_spd_lin (Speed Linear), r_stall (Anti-Stall)")
    lines.append("  String Stability   : r_strg (Shockwave Damping), r_panic (Anti-Panic Brake)")
    lines.append("  Standstill Control : r_std_gap (Queue Buffer), r_pre_stp (Premature Stop), r_hold (Standstill Auto-Hold), r_laun (Drive-Off Launch)")
    lines.append("  BCC Lattice        : r_bcc_tg (Rear Tailgate), r_bcc_dmp (Rear Damp), r_bcc_cns (Midpoint Consensus), r_bcc (Total BCC)")
    lines.append("  Comfort & Power    : r_acc (Cruise a=0 Bias), r_jerk (Jerk Minimization), r_act_d (Action Delta Chattering), r_c_lock (Cruising Attractor Lock), r_enrg (Traction Power)")
    lines.append("=" * 170)
    
    # Setup CSV Writer
    csv_rows = []
    csv_headers = [
        "phase", "step", "time_sec", "car_idx",
        # Inputs
        "v_ego", "a_cmd", "front_gap", "desired_gap", "effective_desired_gap", "v_rel_f",
        "v_lead", "a_lead", "back_gap", "v_rear", "e_f", "target_v", "v_err", "jerk", "act_delta",
        # Reward values
        "r_collision", "r_danger", "r_safety_brake",
        "r_headway", "r_lag", "r_catchup", "r_open_road_hesitation",
        "r_v_match", "r_closing_damp", "r_speed", "r_speed_linear", "r_stall",
        "r_string", "r_panic_brake",
        "r_standstill_gap", "r_premature_stop", "r_standstill_hold", "r_launch",
        "r_bcc_tailgate", "r_bcc_damp", "r_bcc_consensus", "r_bcc",
        "r_accel", "r_jerk", "r_act_delta", "r_comf", "r_cruise_lock", "r_energy",
        "total_reward"
    ]
    csv_rows.append(csv_headers)
    
    phases = [
        {
            "id": 1,
            "name": "PHASE 1: Steady Highway Cruise Equilibrium (v_des = 15.0 m/s, target headway = 18.0 m)",
            "steps": 15,
            "setup": "cruise",
            "snapshot_step": 5,
            "desc": "Testing flatline equilibrium lock (a=0.0 m/s²), zero jerk, and headway sweet-spot reward."
        },
        {
            "id": 2,
            "name": "PHASE 2: Platoon Deceleration & Stop to Complete Rest (Leader smoothly brakes to 0.0 m/s)",
            "steps": 110,
            "setup": "stop",
            "snapshot_step": 8,
            "desc": "Testing string stability damping (r_string), safety braking (r_safety_brake), and queue packing (r_standstill_gap)."
        },
        {
            "id": 3,
            "name": "PHASE 3: Complete Standstill Queue Auto-Hold (All vehicles stopped at 0.00 m/s, gap ~6.0 m)",
            "steps": 20,
            "setup": "hold",
            "snapshot_step": 10,
            "desc": "Testing stationary holding brake reward (r_standstill_hold), creeping prevention, and standstill queue spacing."
        },
        {
            "id": 4,
            "name": "PHASE 4: Platoon Drive-Off Launch (Leader resumes forward and departs)",
            "steps": 35,
            "setup": "resume",
            "snapshot_step": 12,
            "desc": "Testing native drive-off launch acceleration reward (r_launch) and deadband hesitation penalty."
        },
        {
            "id": 5,
            "name": "PHASE 5: Highway Cruising Recovery & Re-Convergence to Platoon Equilibrium",
            "steps": 35,
            "setup": "recovery",
            "snapshot_step": 20,
            "desc": "Testing gap catch-up acceleration (r_catchup), speed matching (r_speed), and cruise attractor re-lock."
        }
    ]
    
    global_step = 0
    
    for ph in phases:
        lines.append("")
        lines.append("-" * 170)
        lines.append(f"[{ph['name']}]")
        lines.append(f"Objective: {ph['desc']}")
        lines.append("-" * 170)
        
        if ph['setup'] == "stop":
            city.set_leader_stop(True)
        elif ph['setup'] == "resume":
            city.set_leader_stop(False)
            
        # Table Header with ALL 24 individual reward values + inputs
        header_line1 = (
            f"{'Step':4s} | {'Time':5s} | {'v_ego':6s} {'a_cmd':6s} {'s_front':7s} {'s_des':6s} {'v_rel':6s} {'a_lead':6s} | "
            f"{'r_col':5s} {'r_dng':5s} {'r_safe':6s} {'r_head':6s} {'r_lag':5s} {'r_ctch':6s} {'r_oph':5s} "
            f"{'r_vmt':5s} {'r_cld':5s} {'r_spd':5s} {'r_spl':5s} {'r_stl':5s} {'r_str':6s} {'r_pnc':5s} "
            f"{'r_sgp':5s} {'r_pst':5s} {'r_hld':6s} {'r_lau':6s} {'r_bcc':6s} {'r_acc':5s} {'r_jrk':6s} "
            f"{'r_act':6s} {'r_lck':5s} {'r_eng':5s} | {'TOTAL_R':8s}"
        )
        lines.append(header_line1)
        lines.append("-" * 170)
        
        snapshot_step = ph.get('snapshot_step', 5)
        snapshot_data = None
        
        for step in range(ph['steps']):
            obs = AccelTrafficEnv.get_city_obs(city, 15.0)
            act, _ = model.predict(obs, deterministic=True)
            acts = [float(a[0]) for a in act]
            
            env.actions = acts
            env.action_deltas = np.abs(np.array(acts) - env.prev_action)
            env.prev_action = np.array(acts).copy()
            env.prev_accel = np.array([c.acceleration for c in city.cars])
            
            for i, c in enumerate(city.cars[1:]):
                c.is_rl = True
                c.rl_action = acts[i]
                
            city.run(dt)
            
            # Use canonical breakdown method from environment
            all_rewards, breakdowns = env._calculate_rewards(return_breakdown=True)
            
            # Focus on Car 1 (first follower directly behind leader) for table display
            b1 = breakdowns[0]
            inp = b1['inputs']
            rew = b1['rewards']
            tot_r = b1['total_reward']
            
            if step == snapshot_step:
                snapshot_data = b1
                
            t_sec = global_step * dt
            
            # Format row with EVERY individual reward value
            row = (
                f"{step:4d} | {t_sec:4.1f}s | {inp['v_ego']:6.2f} {inp['a_cmd']:+6.2f} {inp['front_gap']:7.2f} {inp['desired_gap']:6.2f} {inp['v_rel_f']:+6.2f} {inp['a_lead']:+6.2f} | "
                f"{rew['r_collision']:+5.1f} {rew['r_danger']:+5.1f} {rew['r_safety_brake']:+6.2f} {rew['r_headway']:+6.2f} {rew['r_lag']:+5.1f} {rew['r_catchup']:+6.2f} {rew['r_open_road_hesitation']:+5.1f} "
                f"{rew['r_v_match']:+5.2f} {rew['r_closing_damp']:+5.1f} {rew['r_speed']:+5.2f} {rew['r_speed_linear']:+5.2f} {rew['r_stall']:+5.1f} {rew['r_string']:+6.2f} {rew['r_panic_brake']:+5.1f} "
                f"{rew['r_standstill_gap']:+5.1f} {rew['r_premature_stop']:+5.1f} {rew['r_standstill_hold']:+6.2f} {rew['r_launch']:+6.2f} {rew['r_bcc']:+6.2f} {rew['r_accel']:+5.2f} {rew['r_jerk']:+6.2f} "
                f"{rew['r_act_delta']:+6.2f} {rew['r_cruise_lock']:+5.2f} {rew['r_energy']:+5.2f} | {tot_r:+8.3f}"
            )
            lines.append(row)
            
            # Add to CSV for all followers
            for b in breakdowns:
                ci = b['car_idx']
                binp = b['inputs']
                brew = b['rewards']
                csv_rows.append([
                    ph['id'], step, f"{t_sec:.1f}", ci,
                    f"{binp['v_ego']:.3f}", f"{binp['a_cmd']:.3f}", f"{binp['front_gap']:.3f}",
                    f"{binp['desired_gap']:.3f}", f"{binp['effective_desired_gap']:.3f}", f"{binp['v_rel_f']:.3f}",
                    f"{binp['v_lead']:.3f}", f"{binp['a_lead']:.3f}", f"{binp['back_gap']:.3f}", f"{binp['v_rear']:.3f}",
                    f"{binp['e_f']:.4f}", f"{binp['target_v']:.3f}", f"{binp['v_err']:.4f}",
                    f"{binp['jerk']:.3f}", f"{binp['act_delta']:.4f}",
                    # Rewards
                    f"{brew['r_collision']:.3f}", f"{brew['r_danger']:.3f}", f"{brew['r_safety_brake']:.3f}",
                    f"{brew['r_headway']:.3f}", f"{brew['r_lag']:.3f}", f"{brew['r_catchup']:.3f}", f"{brew['r_open_road_hesitation']:.3f}",
                    f"{brew['r_v_match']:.3f}", f"{brew['r_closing_damp']:.3f}", f"{brew['r_speed']:.3f}", f"{brew['r_speed_linear']:.3f}", f"{brew['r_stall']:.3f}",
                    f"{brew['r_string']:.3f}", f"{brew['r_panic_brake']:.3f}",
                    f"{brew['r_standstill_gap']:.3f}", f"{brew['r_premature_stop']:.3f}", f"{brew['r_standstill_hold']:.3f}", f"{brew['r_launch']:.3f}",
                    f"{brew['r_bcc_tailgate']:.3f}", f"{brew['r_bcc_damp']:.3f}", f"{brew['r_bcc_consensus']:.3f}", f"{brew['r_bcc']:.3f}",
                    f"{brew['r_accel']:.3f}", f"{brew['r_jerk']:.3f}", f"{brew['r_act_delta']:.3f}", f"{brew['r_comf']:.3f}", f"{brew['r_cruise_lock']:.3f}", f"{brew['r_energy']:.3f}",
                    f"{b['total_reward']:.4f}"
                ])
                
            global_step += 1
            
        # Append In-Depth Audit Snapshot for this Phase
        if snapshot_data:
            lines.append("")
            lines.append(f"  >>> MATHEMATICAL REWARD AUDIT SNAPSHOT ({ph['name']} @ step {snapshot_step}):")
            sinp = snapshot_data['inputs']
            srew = snapshot_data['rewards']
            lines.append(f"      Inputs: v_ego={sinp['v_ego']:.2f} m/s, a_cmd={sinp['a_cmd']:+.2f} m/s², s_front={sinp['front_gap']:.2f} m, s_des={sinp['desired_gap']:.2f} m, v_rel={sinp['v_rel_f']:+.2f} m/s, a_lead={sinp['a_lead']:+.2f} m/s², Jerk={sinp['jerk']:.2f} m/s³")
            lines.append(f"      Calculated Reward Component Values:")
            lines.append(f"        • Safety & Boundary : r_col = {srew['r_collision']:+6.2f} | r_dng = {srew['r_danger']:+6.2f} | r_safe = {srew['r_safety_brake']:+6.2f}")
            lines.append(f"        • Headway Control   : r_head = {srew['r_headway']:+6.2f} | r_lag = {srew['r_lag']:+6.2f} | r_ctch = {srew['r_catchup']:+6.2f} | r_oph = {srew['r_open_road_hesitation']:+6.2f}")
            lines.append(f"        • Velocity Matching : r_vmt = {srew['r_v_match']:+6.2f} | r_cld = {srew['r_closing_damp']:+6.2f} | r_spd = {srew['r_speed']:+6.2f} | r_spl = {srew['r_speed_linear']:+6.2f} | r_stl = {srew['r_stall']:+6.2f}")
            lines.append(f"        • String Stability  : r_str = {srew['r_string']:+6.2f} | r_pnc = {srew['r_panic_brake']:+6.2f}")
            lines.append(f"        • Standstill & Que  : r_sgp = {srew['r_standstill_gap']:+6.2f} | r_pst = {srew['r_premature_stop']:+6.2f} | r_hld = {srew['r_standstill_hold']:+6.2f} | r_lau = {srew['r_launch']:+6.2f}")
            lines.append(f"        • Bilateral BCC     : r_b_tg = {srew['r_bcc_tailgate']:+6.2f} | r_b_dmp = {srew['r_bcc_damp']:+6.2f} | r_b_cns = {srew['r_bcc_consensus']:+6.2f} (r_bcc = {srew['r_bcc']:+6.2f})")
            lines.append(f"        • Comfort & Power   : r_acc = {srew['r_accel']:+6.2f} | r_jrk = {srew['r_jerk']:+6.2f} | r_act = {srew['r_act_delta']:+6.2f} | r_lck = {srew['r_cruise_lock']:+6.2f} | r_eng = {srew['r_energy']:+6.2f}")
            lines.append(f"        ----------------------------------------------------------------------------------------------------")
            lines.append(f"        TOTAL STEP REWARD   : {snapshot_data['total_reward']:+8.4f}")
            lines.append("")
            
    # Write Text Log
    with open(output_log, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated text runtime reward log at: {output_log}")
    
    # Write CSV Log
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Generated CSV telemetry reward log at: {output_csv}")
    
    return output_log, output_csv

if __name__ == "__main__":
    generate_reward_breakdown_log()
