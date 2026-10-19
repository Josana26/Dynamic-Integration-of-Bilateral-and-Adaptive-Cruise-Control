import math
import numpy as np
from simulation.city import City

class RLCity(City):
    """
    Extended City class to support direct acceleration control via RL.
    Cars placed in 'RL-A' mode will have their acceleration determined 
    by their `rl_action` attribute, bypassing the standard heuristics.

    Supports two action modes:
    1. Direct acceleration (default): policy outputs target acceleration,
       physically jerk-clamped to +/-5.0 m/s^3.
    2. Jerk-limited action (Stage 1): policy outputs change in acceleration (delta_a),
       clipped to +/- jerk_limit_rate * dt (default 3.0 m/s^3 * dt), then clamped
       to [min_a, max_a].
    """
    def __init__(self, jerk_limited_action=False, jerk_limit_rate=3.0):
        super().__init__()
        self.jerk_limited_action = jerk_limited_action
        self.jerk_limit_rate = jerk_limit_rate

    def driver_decision(self):
        # Save previous acceleration for RL-controlled cars to calculate jerk properly
        for car in self.cars:
            if getattr(car, 'is_rl', False):
                car._prev_acc = car.acceleration
                # First-order physical actuator filter (tau ≈ 0.2s, alpha = 0.35)
                # Eliminates high-frequency policy chattering and jagged acceleration spikes
                if hasattr(car, 'rl_action'):
                    raw_cmd = float(car.rl_action)
                    prev_cmd = getattr(car, '_filtered_cmd', raw_cmd)
                    if car.velocity <= 1e-4 and raw_cmd > 0.0:
                        prev_cmd = max(0.0, prev_cmd)
                    alpha = 0.30  # Balanced filter: good chattering suppression without C3 calmness regression
                    car.rl_action = alpha * raw_cmd + (1.0 - alpha) * prev_cmd
                    car._filtered_cmd = car.rl_action

        # Let the standard heuristic compute accelerations for all cars
        super().driver_decision()

        # Overwrite the acceleration for RL-controlled cars
        dt = self.dt
        for car in self.cars:
            if getattr(car, 'is_rl', False) and hasattr(car, 'rl_action'):
                raw_cmd = float(car.rl_action)
                last_acc = getattr(car, '_prev_acc', car.acceleration)

                is_jerk_limited = (
                    getattr(self, 'jerk_limited_action', False) or
                    getattr(car, 'jerk_limited_action', False)
                )

                if is_jerk_limited:
                    # Jerk-limited action mode (Stage 1):
                    # Policy outputs change in acceleration (delta_a),
                    # acc = previous_accel + clipped change (limited to 3.0 m/s^3 * dt)
                    jerk_rate = float(getattr(self, 'jerk_limit_rate', 3.0))
                    max_change = jerk_rate * dt
                    clipped_change = max(-max_change, min(max_change, raw_cmd))
                    target_acc = last_acc + clipped_change
                    acc = max(self.min_a, min(self.max_a, target_acc))
                else:
                    # Standard direct acceleration command from the RL model
                    target_acc = max(self.min_a, min(self.max_a, raw_cmd))
                    jerk = (target_acc - last_acc) / dt
                    max_jerk = 3.0  # Tightened from 5.0 for automotive comfort-grade smoothness
                    
                    if jerk > max_jerk:
                        acc = last_acc + max_jerk * dt
                    elif jerk < -max_jerk:
                        acc = last_acc - max_jerk * dt
                    else:
                        acc = target_acc
                
                # If car is at rest, physical holding brake gives zero net acceleration
                if car.velocity <= 1e-4 and acc <= 0.0:
                    acc = 0.0

                car.acceleration = acc

    def init(self, *args, **kwargs):
        if 'jerk_limited_action' in kwargs:
            self.jerk_limited_action = kwargs.pop('jerk_limited_action')
        if 'jerk_limit_rate' in kwargs:
            self.jerk_limit_rate = kwargs.pop('jerk_limit_rate')
        super().init(*args, **kwargs)
        self.collided_this_step = False
        for car in self.cars:
            car.just_collided = False
            car.just_collided_front = False
            car.just_collided_rear = False
            car._filtered_cmd = 0.0
            car._prev_acc = 0.0
            car.jerk_limited_action = getattr(self, 'jerk_limited_action', False)

    def handle_collisions(self):
        self.collided_this_step = False
        for car in self.cars:
            car.just_collided = False
            car.just_collided_front = False
            car.just_collided_rear = False

        road_length = self.roads[0].length if self.roads else 1000
        sorted_cars = sorted(self.cars, key=lambda c: c.pos)
        for i, car in enumerate(sorted_cars):
            next_car = sorted_cars[(i + 1) % len(sorted_cars)]
            if car == next_car:
                continue
            # car is ahead (smaller pos), next_car is behind (larger pos)
            gap_raw = (next_car.pos - car.pos) % road_length
            if gap_raw <= car.length:
                self.collided_this_step = True
                next_car.just_collided = True
                next_car.just_collided_front = True
                car.just_collided = True
                car.just_collided_rear = True

        super().handle_collisions()

