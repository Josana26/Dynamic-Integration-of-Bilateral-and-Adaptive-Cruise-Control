import math
import numpy as np
from simulation.city import City

class RLCity(City):
    """
    Extended City class to support direct acceleration control via RL.
    Cars placed in 'RL-A' mode will have their acceleration determined 
    by their `rl_action` attribute, bypassing the standard heuristics.
    """
    def driver_decision(self):
        # Save previous acceleration for RL-controlled cars to calculate jerk properly
        for car in self.cars:
            if getattr(car, 'is_rl', False):
                car._prev_acc = car.acceleration

        # Let the standard heuristic compute accelerations for all cars
        super().driver_decision()

        # Overwrite the acceleration for RL-controlled cars
        dt = self.dt
        road_len = self.roads[0].length if self.roads else 1000
        min_dis = getattr(self, 'min_dis', 6.0)

        for car in self.cars:
            if getattr(car, 'is_rl', False) and hasattr(car, 'rl_action'):
                # rl_action is already a raw acceleration value in [min_a, max_a]
                target_acc = float(car.rl_action)
                
                # Standstill Auto-Hold Coordination:
                # If vehicle has come to a stop or is at crawl (v <= 0.01 m/s):
                if car.velocity <= 0.01:
                    if target_acc <= 0.0:
                        target_acc = 0.0
                    else:
                        cars_same_road = [c for c in self.cars if c.current_road == car.current_road and c != car]
                        def gap_to(other):
                            g = (car.pos - other.pos) % road_len
                            return g if g > 0 else float('inf')
                        front_car = min(cars_same_road, key=gap_to, default=None) if cars_same_road else None
                        front_v = front_car.velocity if front_car else 0.0
                        front_g = (car.pos - front_car.pos - front_car.length) % road_len if front_car else road_len
                        # If lead vehicle is stopped or stopping and within buffer, hold standstill
                        if (front_v <= 0.1 or getattr(self, 'leader_stop', False)) and front_g <= min_dis * 1.15:
                            target_acc = 0.0

                # Enforce physical constraints: min/max acceleration
                target_acc = max(self.min_a, min(self.max_a, target_acc))
                
                # Enforce physical constraints: jerk (rate of change of acceleration)
                last_acc = getattr(car, '_prev_acc', car.acceleration)
                jerk = (target_acc - last_acc) / dt
                max_jerk = 5.0
                
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
        super().init(*args, **kwargs)
        self.collided_this_step = False
        for car in self.cars:
            car.just_collided = False
            car.just_collided_front = False
            car.just_collided_rear = False

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

