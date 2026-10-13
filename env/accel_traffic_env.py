from stable_baselines3.common.vec_env import VecEnv
import numpy as np
from gymnasium import spaces
from simulation.rl_city import RLCity

class AccelTrafficEnv(VecEnv):
    """
    Decentralized Multi-Agent Environment wrapping RLCity.
    Uses VecEnv interface so Stable-Baselines3 naturally trains
    a single parameter-shared policy across all cars.
    
    Key design choices:
    - Lead car (index 0) is NOT RL-controlled. It follows a randomized
      velocity profile each episode so follower cars learn to react.
    - Only follower cars (indices 1..N-1) are RL-controlled.
    - Simulation parameters are domain-randomized each episode for robustness.
    - Observations are normalized to roughly [-1, 1] for stable learning.
    """
    # Normalization constants for Error-State Dynamic Coordinates
    # Features: [vel_err, ego_accel, front_gap_err, front_rel_vel, back_gap_err, back_rel_vel, rel_gap_ratio_err]
    # At cruise equilibrium (v=v_des, gap=desired_gap, a=0), ALL 7 features are IDENTICALLY 0.0.
    OBS_MEANS = np.zeros(7, dtype=np.float32)
    OBS_STDS  = np.array([5.0, 2.0, 10.0, 4.0, 10.0, 4.0, 0.5], dtype=np.float32)

    def __init__(self):
        self.city = RLCity()
        self.dt = 0.1
        self.max_steps = 1000  # 100 seconds at dt=0.1
        self.current_step = 0
        self.num_cars = 15
        
        # Base simulation parameters (aligned with GUI defaults)
        self.base_params = {
            'kd': 0.9, 'kv': 0.5, 'kc': 0.4,
            'v_des': 15.0,
            'min_dis': 6.0,
            'reaction_time': 0.8,
            'max_a': 3.0,
            'min_a': -5.0,
            'max_v': 30.0,
            'min_v': 0.0,
        }
        
        # Current episode parameters (set during reset)
        self.sim_params = dict(self.base_params)
        
        # Lead car velocity profile for current episode
        self.lead_profile = []
        self.lead_profile_time = 0.0
        
        # Spaces for a SINGLE car (action = raw acceleration)
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.float32(-5.0), 
            high=np.float32(5.0), 
            shape=(1,), 
            dtype=np.float32
        )
        
        # VecEnv treats each car as a separate "environment"
        # But we only train follower cars (num_cars - 1)
        self.num_rl_cars = self.num_cars - 1  # Exclude lead car
        super().__init__(self.num_rl_cars, obs_space, act_space)
        self.prev_accel = np.zeros(self.num_cars)
        self.prev_action = np.zeros(self.num_rl_cars, dtype=np.float32)
        self.action_deltas = np.zeros(self.num_rl_cars, dtype=np.float32)
        # Track collision state per car so penalty is one-time, not per-step
        self.in_collision = np.zeros(self.num_rl_cars, dtype=bool)
        
        # Load ALL custom lead profiles from CSV files for training
        self.custom_profiles = []
        import csv
        import os
        env_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(env_dir)
        csv_files = ["data/data.csv", "data/data1.csv", "data/data2.csv"]
        for csv_file in csv_files:
            csv_path = os.path.join(project_dir, csv_file)
            if os.path.exists(csv_path):
                try:
                    profile = []
                    with open(csv_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            t = float(row['time'])
                            v = float(row['velocity'])
                            profile.append((t, v))
                    if profile:
                        self.custom_profiles.append(profile)
                        print(f"Loaded custom profile from {csv_file} ({len(profile)} points, "
                              f"{profile[0][1]:.1f}-{max(v for _,v in profile):.1f} m/s)")
                except Exception as e:
                    print(f"Could not load {csv_file}: {e}")
        print(f"Total custom profiles loaded: {len(self.custom_profiles)}")
        
    def _randomize_params(self):
        """Domain randomization: vary parameters around GUI defaults each episode."""
        rng = np.random.default_rng()
        self.sim_params = {
            'kd': rng.uniform(0.7, 1.1),
            'kv': rng.uniform(0.4, 0.7),
            'kc': rng.uniform(0.3, 0.5),
            'v_des': rng.uniform(12.0, 28.0),
            'min_dis': rng.uniform(5.0, 7.0),
            'reaction_time': rng.uniform(0.7, 1.0),
            'max_a': 3.0,
            'min_a': -5.0,
            'max_v': 35.0,
            'min_v': 0.0,
        }
    
    def _generate_lead_profile(self):
        """
        Generate a random velocity profile for the lead car.
        This creates realistic disturbances: cruise, brake, accelerate.
        """
        rng = np.random.default_rng()
        profile = []
        t = 0.0
        v = self.sim_params['v_des']
        episode_duration = self.max_steps * self.dt  # 100 seconds
        
        while t < episode_duration:
            # Hold current velocity for a random duration
            hold_time = rng.uniform(5.0, 20.0)
            profile.append((t, v))
            t += hold_time
            
            if t >= episode_duration:
                break
            
            # Choose a new target velocity (realistic perturbation)
            event = rng.choice(['brake', 'accel', 'cruise'], p=[0.35, 0.30, 0.35])
            if event == 'brake':
                v_new = max(0.0, v - rng.uniform(3.0, 15.0))  # Allow complete stop
            elif event == 'accel':
                v_new = min(self.sim_params['max_v'] - 5, v + rng.uniform(3.0, 10.0))
            else:
                v_new = v  # Maintain speed with slight variation
                v_new += rng.uniform(-2.0, 2.0)
                v_new = max(0.0, min(self.sim_params['max_v'] - 5, v_new))  # Allow complete stop
            
            # Transition to new velocity over a short ramp
            ramp_time = rng.uniform(2.0, 5.0)
            profile.append((t, v))
            t += ramp_time
            v = v_new
            profile.append((t, v))
        
        # Ensure we end at the episode boundary
        profile.append((episode_duration, v))
        return profile

    def _generate_full_stop_profile(self):
        """
        Generate a realistic platoon full-stop & standstill queue profile.
        Lead car cruises at v_des, executes a smooth deceleration ramp to 0.0 m/s,
        holds at complete standstill (0.0 m/s) for 25-45 seconds to force all
        follower vehicles to decelerate smoothly and pack cleanly at min_dis.
        """
        rng = np.random.default_rng()
        v_init = self.sim_params['v_des']
        episode_duration = self.max_steps * self.dt  # 100 seconds
        
        # 1. Initial cruise phase: 8 - 18 seconds
        t_cruise = rng.uniform(8.0, 18.0)
        profile = [(0.0, v_init), (t_cruise, v_init)]
        
        # 2. Smooth braking deceleration ramp down to 0.0 m/s over 3.0 - 5.0 seconds
        t_brake = rng.uniform(3.0, 5.0)
        profile.append((t_cruise + t_brake, 0.0))
        
        # 3. Standstill hold duration: 25 - 45 seconds at 0.0 m/s
        t_hold = rng.uniform(25.0, 45.0)
        t_stopped_until = min(episode_duration, t_cruise + t_brake + t_hold)
        profile.append((t_stopped_until, 0.0))
        
        # 4. In 40% of cases, resume back up to speed to test stop-and-go launch;
        # in 60% remain at complete standstill until episode end.
        if t_stopped_until < episode_duration and rng.random() < 0.4:
            t_accel = rng.uniform(4.0, 6.0)
            v_resume = rng.uniform(10.0, v_init)
            profile.append((t_stopped_until + t_accel, v_resume))
            profile.append((episode_duration, v_resume))
        else:
            profile.append((episode_duration, 0.0))
            
        return profile

    def _generate_wave_profile(self):
        """
        Generate sinusoidal / harmonic wave profile to train String Stability.
        v(t) = v_base + amplitude * sin(2 * pi * t / period)
        Forces follower vehicles to learn smooth damping of upstream speed waves.
        """
        rng = np.random.default_rng()
        v_base = rng.uniform(12.0, 22.0)
        amp = rng.uniform(2.5, 4.5)
        period = rng.uniform(12.0, 22.0)
        episode_duration = self.max_steps * self.dt
        
        profile = []
        for t in np.arange(0.0, episode_duration + 0.5, 0.5):
            v_t = max(2.0, min(self.sim_params['max_v'] - 3.0, v_base + amp * np.sin(2.0 * np.pi * t / period)))
            profile.append((float(t), float(v_t)))
        self.sim_params['v_des'] = v_base
        return profile

    def _generate_speed_drop_profile(self):
        """
        Generate highway speed drop & recovery profile (e.g. 20 -> 6 -> 20 m/s).
        Tests deceleration shockwave dissipation and string-stable re-acceleration.
        """
        rng = np.random.default_rng()
        v_high = rng.uniform(15.0, 25.0)
        v_low = rng.uniform(4.0, 8.0)
        t_cruise1 = rng.uniform(10.0, 18.0)
        t_ramp_down = rng.uniform(3.0, 5.0)
        t_hold_low = rng.uniform(15.0, 30.0)
        t_ramp_up = rng.uniform(4.0, 7.0)
        episode_duration = self.max_steps * self.dt
        
        t1 = t_cruise1
        t2 = t1 + t_ramp_down
        t3 = t2 + t_hold_low
        t4 = min(episode_duration, t3 + t_ramp_up)
        
        profile = [
            (0.0, v_high),
            (t1, v_high),
            (t2, v_low),
            (t3, v_low),
            (t4, v_high),
            (episode_duration, v_high)
        ]
        self.sim_params['v_des'] = v_high
        return profile
    
    def _apply_lead_profile(self):
        """Apply the velocity profile to the lead car via the city's built-in mechanism."""
        self.city.lead_velocity_profile = self.lead_profile
        
    def reset(self):
        self.current_step = 0
        
        # Randomize parameters for this episode
        self._randomize_params()
        
        # Mix training profiles equally across 5 operational domains:
        # 1. 20% steady-state cruise (zero-oscillation flatline at v_des)
        # 2. 20% full-stop queue hold (clean deceleration & standstill packing at min_dis)
        # 3. 20% harmonic wave oscillations (string-stability wave attenuation)
        # 4. 20% highway speed drop & recovery (shockwave damping without over-braking)
        # 5. 20% custom CSV profiles (data.csv urban stop-and-go disturbance)
        rng = np.random.default_rng()
        prob = rng.random()
        if prob < 0.20:
            self.lead_profile = [
                (0.0, self.sim_params['v_des']),
                (self.max_steps * self.dt, self.sim_params['v_des'])
            ]
        elif prob < 0.40:
            self.lead_profile = self._generate_full_stop_profile()
        elif prob < 0.60:
            self.lead_profile = self._generate_wave_profile()
        elif prob < 0.80:
            self.lead_profile = self._generate_speed_drop_profile()
        elif self.custom_profiles:
            idx = rng.integers(0, len(self.custom_profiles))
            self.lead_profile = self.custom_profiles[idx]
            max_v_in_profile = max(v for _, v in self.lead_profile)
            self.sim_params['v_des'] = max(max_v_in_profile, 5.0)
        else:
            self.lead_profile = self._generate_lead_profile()
        
        self.city.init(
            self.num_cars,
            self.sim_params['kd'],
            self.sim_params['kv'],
            self.sim_params['kc'],
            self.sim_params['v_des'],
            self.sim_params['max_v'],
            self.sim_params['min_v'],
            self.sim_params['min_dis'],
            self.sim_params['reaction_time'],
            2.0,  # headway_time
            self.sim_params['max_a'],
            self.sim_params['min_a'],
            2.0,  # min_gap
            self.dt,
            model='ACC'  # Base model for the lead car's heuristic fallback
        )
        
        # Lead car (index 0): NOT RL-controlled, follows velocity profile
        self.city.cars[0].is_rl = False
        
        # Follower cars (index 1..N-1): RL-controlled
        for car in self.city.cars[1:]:
            car.is_rl = True
            car.rl_action = 0.0
        
        # Apply the lead car's velocity profile
        self._apply_lead_profile()
            
        self.prev_accel = np.zeros(self.num_cars)
        self.prev_action = np.zeros(self.num_rl_cars, dtype=np.float32)
        self.action_deltas = np.zeros(self.num_rl_cars, dtype=np.float32)
        self.in_collision = np.zeros(self.num_rl_cars, dtype=bool)
        return self._get_obs()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        self.current_step += 1
        
        acts = np.array(self.actions).flatten()
        
        # Track action rate of change (delta) to heavily penalize chattering
        self.action_deltas = np.abs(acts[:self.num_rl_cars] - self.prev_action)
        self.prev_action = acts[:self.num_rl_cars].copy()
        
        # Save previous accelerations for jerk calculation in rewards
        self.prev_accel = np.array([c.acceleration for c in self.city.cars]) if self.city.cars else np.zeros(self.num_cars)
        
        # Apply RL actions to FOLLOWER cars only (indices 1..N-1)
        for i, car in enumerate(self.city.cars[1:]):
            if i < len(acts):
                car.is_rl = True
                car.rl_action = float(acts[i])
            
        self.city.run(self.dt)
        
        obs = self._get_obs()
        rewards = self._calculate_rewards()
        
        # We no longer terminate the episode early on a collision
        # to avoid punishing all cars for one car's mistake (Credit Assignment Problem).
        done = (self.current_step >= self.max_steps)
        
        global_collision = self._check_collision()
            
        dones = np.array([done] * self.num_rl_cars)
        infos = [{"collisions": 1 if global_collision else 0} for _ in range(self.num_rl_cars)]
        
        # Auto-reset logic required for VecEnv
        if done:
            terminal_obs = obs.copy()
            for i in range(self.num_rl_cars):
                infos[i]['terminal_observation'] = terminal_obs[i]
            obs = self.reset()
            
        return obs, rewards, dones, infos

    @classmethod
    def get_city_obs(cls, city, v_des=15.0):
        """
        Build normalized observations for FOLLOWER cars only (indices 1..N-1).
        Used by environment, GUI, and headless evaluation for 100% mathematical consistency.
        Features: [ego_vel, ego_accel, front_gap, front_rel_vel, back_gap, back_rel_vel, v_des]
        """
        cars = city.cars
        if not cars or len(cars) <= 1:
            return np.zeros((0, 7), dtype=np.float32)
            
        road_length = city.roads[0].length if city.roads else 1000
        num_followers = len(cars) - 1
        obs_array = np.zeros((num_followers, 7), dtype=np.float32)
        lead_car = cars[0]
        MAX_RADAR_RANGE = 70.0  # Max realistic radar sensing range in meters
        
        for obs_idx, car in enumerate(cars[1:]):
            cars_same_road = [c for c in cars if c.current_road == car.current_road and c != car]
            
            def gap_to(other):
                gap = (car.pos - other.pos) % road_length
                return gap if gap > 0 else float('inf')
                
            def gap_from(other):
                gap = (other.pos - car.pos) % road_length
                return gap if gap > 0 else float('inf')
                
            front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None
            back_car = min(cars_same_road, key=gap_from, default=None) if cars_same_road else None
            
            ego_vel = car.velocity
            ego_accel = car.acceleration
            
            min_dis = getattr(city, 'min_dis', 6.0)
            reaction_time = getattr(city, 'reaction_time', 0.8)
            desired_gap = min_dis + ego_vel * reaction_time

            if front_car:
                front_gap = (car.pos - front_car.pos - front_car.length) % road_length
                front_rel_vel = front_car.velocity - ego_vel
            else:
                front_gap = MAX_RADAR_RANGE
                front_rel_vel = 0.0
            front_gap = min(front_gap, MAX_RADAR_RANGE)
            
            # Dynamic Kinematic Cushion: expands desired headway when rapidly closing in
            d_kinematic = (front_rel_vel ** 2) / (2.0 * 3.5) if front_rel_vel < -0.5 else 0.0
            effective_desired_gap = desired_gap + d_kinematic
            
            # Headway error with bounded limits to prevent tanh saturation
            raw_gap_err = front_gap - effective_desired_gap
            front_gap_err = float(np.clip(raw_gap_err, -30.0, 30.0))
            rel_gap_ratio_err = float(np.clip(raw_gap_err / max(effective_desired_gap, 1.0), -2.0, 2.0))
            
            # Bilateral Rear Car Sensing:
            desired_back_gap = min_dis + (back_car.velocity if back_car else ego_vel) * reaction_time
            if back_car and back_car != lead_car:
                back_gap_raw = (back_car.pos - car.pos - car.length) % road_length
                if back_gap_raw <= MAX_RADAR_RANGE:
                    back_gap = back_gap_raw
                    back_rel_vel = back_car.velocity - ego_vel
                else:
                    back_gap = desired_back_gap
                    back_rel_vel = 0.0
            else:
                back_gap = desired_back_gap
                back_rel_vel = 0.0
                
            back_gap_err = float(np.clip(back_gap - desired_back_gap, -30.0, 30.0))
            
            # Adaptive Target Velocity: adapts to slower lead vehicle in radar range
            if front_car and front_gap < 45.0 and front_car.velocity < v_des:
                effective_v_des = front_car.velocity
            else:
                effective_v_des = v_des
            vel_err = float(np.clip(ego_vel - effective_v_des, -15.0, 15.0))
                
            raw = np.array([vel_err, ego_accel, front_gap_err, front_rel_vel, back_gap_err, back_rel_vel, rel_gap_ratio_err], dtype=np.float32)
            obs_array[obs_idx] = (raw - cls.OBS_MEANS) / (cls.OBS_STDS + 1e-8)
            
        return obs_array

    def _get_obs(self):
        """Build normalized observations using canonical get_city_obs."""
        return self.get_city_obs(self.city, self.sim_params['v_des'])

    def _calculate_rewards(self):
        """
        Smooth, physically-convex reward function for follower cars (indices 1..N-1).
        Prioritizes collision avoidance, sweet-spot headway tracking, string stability,
        bilateral awareness, and comfort without erratic cliff-edges.
        """
        cars = self.city.cars
        rewards = np.zeros(self.num_rl_cars, dtype=np.float32)
        if not cars: return rewards
        
        road_length = self.city.roads[0].length if self.city.roads else 1000
        v_des = self.sim_params['v_des']
        max_a = self.sim_params['max_a']
        min_dis = self.sim_params['min_dis']
        reaction_time = self.sim_params['reaction_time']
        lead_car = cars[0]
        MAX_RADAR_RANGE = 70.0
        
        for obs_idx, car in enumerate(cars[1:]):
            car_idx = obs_idx + 1  # Actual index in self.city.cars
            cars_same_road = [c for c in cars if c.current_road == car.current_road and c != car]
            
            def gap_to(other):
                gap = (car.pos - other.pos) % road_length
                return gap if gap > 0 else float('inf')
            def gap_from(other):
                gap = (other.pos - car.pos) % road_length
                return gap if gap > 0 else float('inf')
                
            front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None
            back_car = min(cars_same_road, key=gap_from, default=None) if cars_same_road else None
            
            front_gap = (car.pos - front_car.pos - front_car.length) % road_length if front_car else road_length
            
            desired_gap = min_dis + car.velocity * reaction_time
            v_rel_f = (front_car.velocity - car.velocity) if front_car else 0.0
            
            # Kinematic stopping buffer when closing in fast
            d_kinematic = (v_rel_f ** 2) / (2.0 * 3.5) if v_rel_f < -0.5 else 0.0
            effective_desired_gap = desired_gap + d_kinematic
            
            # === 1. COLLISION ===
            has_collided = (
                getattr(car, 'just_collided_front', False)
                or getattr(car, 'just_collided', False)
                or (getattr(car, 'collision_timer', 0) == 40)
                or (front_gap < 0.5)
            )
            r_collision = 0.0
            if has_collided:
                r_collision = -40.0
                if not self.in_collision[obs_idx]:
                    r_collision -= 40.0  # -80 total on collision start
            self.in_collision[obs_idx] = has_collided
            
            # === 2. SAFE HEADWAY TRACKING ===
            e_f = (front_gap - effective_desired_gap) / max(effective_desired_gap, 1.0)
            r_headway = 2.5 * np.exp(- (e_f / 0.25) ** 2)
            
            # Safety boundary: progressive danger penalty
            r_danger = 0.0
            if front_gap < 4.5:
                r_danger = -15.0 * (((4.5 - max(front_gap, 0.0)) / 4.5) ** 2) - 5.0
            elif front_gap < 0.65 * desired_gap:
                r_danger = -4.0 * (((0.65 * desired_gap - front_gap) / (0.65 * desired_gap)) ** 2)
                
            # === 3. RELATIVE VELOCITY & STRING STABILITY DAMPING ===
            r_v_match = 1.5 * np.exp(- (v_rel_f / 2.0) ** 2)
            
            # Asymmetric damping: if closing in fast on front vehicle (v_rel_f < -1.0)
            r_closing_damp = 0.0
            if v_rel_f < -1.0 and front_gap < 2.0 * effective_desired_gap:
                r_closing_damp = -2.5 * min(((-v_rel_f - 1.0) / 3.0) ** 2, 4.0)
                
            # String Stability: strictly penalize downstream acceleration/deceleration wave amplification
            r_string = 0.0
            if front_car:
                if front_car.acceleration < -0.1 and car.acceleration < front_car.acceleration:
                    # Over-braking relative to lead vehicle (amplifying deceleration wave)
                    r_string -= 3.5 * min((front_car.acceleration - car.acceleration) / 1.5, 3.5)
                elif front_car.acceleration > 0.1 and car.acceleration > front_car.acceleration:
                    # Over-accelerating relative to lead vehicle (amplifying acceleration wave)
                    r_string -= 2.0 * min((car.acceleration - front_car.acceleration) / 1.5, 2.0)
                
            # === 4. SPEED TRACKING & EQUILIBRIUM PROGRESS ===
            if front_car and front_car.velocity < v_des and front_gap < 45.0:
                target_v = front_car.velocity
            else:
                target_v = v_des
                
            v_err = (car.velocity - target_v) / max(target_v, 1.0)
            r_speed = 2.0 * np.exp(- (v_err / 0.15) ** 2)
            
            # Linear speed tracking penalty: active only when open road ahead (front_gap >= effective_desired_gap)
            r_speed_linear = 0.0
            if front_gap >= effective_desired_gap:
                r_speed_linear = -1.5 * abs(v_err)
            
            # Anti-lag penalty: prevents lazy falling back when open road ahead
            r_lag = 0.0
            if e_f > 0.35:
                r_lag = -1.5 * min(e_f - 0.35, 3.0)
            
            # Anti-stall: penalize sitting near 0 m/s when open road is available
            r_stall = 0.0
            if car.velocity < 3.0 and target_v > 8.0 and front_gap > effective_desired_gap:
                r_stall = -4.0 * (1.0 - car.velocity / 3.0)
                
            # === 5. STANDSTILL QUEUE-PACKING & HEADWAY PRECISION ===
            # Applies ONLY when front car is stationary/crawling AND ego is stopping/stopped!
            r_standstill_gap = 0.0
            r_premature_stop = 0.0
            if front_car and front_car.velocity <= 0.2 and car.velocity <= 1.0:
                if front_gap >= (min_dis - 0.5) and front_gap <= (min_dis + 1.2):
                    r_standstill_gap = 2.5
                elif front_gap > (min_dis + 1.2):
                    gap_excess = front_gap - (min_dis + 1.2)
                    r_standstill_gap = -2.0 * min((gap_excess / 5.0) ** 2, 2.5)
                    
                if car.velocity < 0.2 and front_gap > min_dis + 2.0:
                    r_premature_stop = -2.5 * min((front_gap - (min_dis + 2.0)) / min_dis, 2.0)

            # Tail car (boundary vehicle) compensation
            is_tail_car = (back_car is None or back_car == lead_car)
            if is_tail_car:
                r_headway *= 1.25

            # === 6. BCC BILATERAL REAR AWARENESS & SPRING-DAMPER LATTICE ===
            r_bcc = 0.0
            if back_car and back_car != lead_car:
                back_gap_raw = (back_car.pos - car.pos - car.length) % road_length
                if back_gap_raw <= MAX_RADAR_RANGE:
                    desired_back_gap = min_dis + back_car.velocity * reaction_time
                    e_b = (back_gap_raw - desired_back_gap) / max(desired_back_gap, 1.0)
                    # Symmetric bilateral spring: maintains bilateral tension
                    r_bcc_spring = -1.0 * min(e_b ** 2, 3.0)
                    # Bilateral velocity damping: synchronizes speed with rear vehicle
                    r_bcc_damp = -0.5 * min(((back_car.velocity - car.velocity) / 2.0) ** 2, 2.0)
                    r_bcc = r_bcc_spring + r_bcc_damp
                        
            # === 7. SMOOTHNESS, ACTION-RATE & ZERO-ACCELERATION EQUILIBRIUM BIAS ===
            r_accel = 1.0 * np.exp(- (car.acceleration / 0.35) ** 2) - 0.2 * ((car.acceleration / 2.0) ** 2)
            delta_a = car.acceleration - self.prev_accel[car_idx]
            r_jerk = -0.6 * ((delta_a / 1.0) ** 2)
            
            act_delta = self.action_deltas[obs_idx] if obs_idx < len(self.action_deltas) else 0.0
            r_act_delta = -1.5 * ((act_delta / 1.0) ** 2)
            
            # Equilibrium Cruising Lock: When within 15% of headway and speed matches, strongly lock a=0.0
            r_cruise_lock = 0.0
            if abs(e_f) < 0.15 and abs(v_rel_f) < 0.8 and abs(v_err) < 0.10:
                r_cruise_lock = 3.5 * np.exp(- (car.acceleration / 0.15) ** 2)

            # === 8. ENERGY EFFICIENCY (Traction power) ===
            p_pos = max(0.0, car.acceleration * car.velocity)
            r_energy = -0.2 * (p_pos / max(max_a * v_des, 1.0))
                
            rewards[obs_idx] = (
                r_collision + r_headway + r_danger + r_v_match + 
                r_closing_damp + r_string + r_speed + r_speed_linear + r_lag + r_stall + 
                r_standstill_gap + r_premature_stop +
                r_bcc + r_accel + r_jerk + r_act_delta + r_cruise_lock + r_energy
            )
            
        return rewards

    def _check_collision(self):
        return getattr(self.city, 'collided_this_step', False) or any(
            getattr(c, 'just_collided', False) or getattr(c, 'collision_timer', 0) == 40
            for c in self.city.cars
        )

    # Required VecEnv methods
    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [None]*self.num_rl_cars
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): return [None]*self.num_rl_cars
    def env_is_wrapped(self, wrapper_class, indices=None): return [False]*self.num_rl_cars
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
