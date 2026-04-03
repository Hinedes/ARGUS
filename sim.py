import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from shapely.geometry import Point, LineString, MultiLineString
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DynamicEchosEnv(gym.Env):
    """
    Echos Live Simulation Environment.
    Randomized mazes. Command A (Scout) and Command B (Hunt) logic.
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Obs: [VectorX_to_waypoint, VectorY_to_waypoint, Covariance, Ray1...Ray5]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.COVARIANCE_THRESHOLD = 5.0
        self.SNR_HUNT_THRESHOLD = 4.5 # LOWERED: Echos will now hear the human at 4.5 meters.
        self.MAX_RANGE = 10.0 
        
        self.current_step = 0
        self.max_steps = 1500 # Increased battery life for the return trip

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.uav_pos = np.array([1.0, 0.0])
        self.uav_yaw = 0.0
        self.covariance = 0.1
        self.sensor_mode = "THROW"
        self.command_state = "COMMAND A: SCOUT"
        self.current_step = 0
        self.last_yaw_rate = 0.0
        self.rth_complete = False
        self.min_wall_dist = 10.0  # Safe default before first scan
        
        # 1. Randomize the Maze and Survivor
        self._generate_random_maze()
        
        # 2. Initial Commander Waypoint
        self.current_waypoint = np.array([8.0, 0.0]) # Initial scout waypoint (the junction)
        self.path_b_snr = self._calculate_path_b_snr()
        
        return self._get_obs(), self._get_info()

    def _generate_random_maze(self):
        """Randomly generates a Left-turn, Right-turn, or T-junction."""
        maze_type = np.random.choice(["LEFT", "RIGHT", "T_JUNCT"])
        
        base_walls = [
            ((0, 1), (8, 1)), ((0, -1), (8, -1)), ((0, 1), (0, -1)) # Sealed starting corridor
        ]
        
        if maze_type == "LEFT":
            self.walls = MultiLineString(base_walls + [
                ((8, -1), (12, -1)), ((12, -1), (12, 5)), ((8, 1), (8, 5)), ((8, 5), (12, 5))
            ])
            self.survivor_pos = np.array([10.0, 4.0])
        elif maze_type == "RIGHT":
            self.walls = MultiLineString(base_walls + [
                ((8, 1), (12, 1)), ((12, 1), (12, -5)), ((8, -1), (8, -5)), ((8, -5), (12, -5))
            ])
            self.survivor_pos = np.array([10.0, -4.0])
        else: # T_JUNCT
            self.walls = MultiLineString(base_walls + [
                ((8, 1), (8, 5)), ((8, -1), (8, -5)), ((8, 5), (12, 5)), 
                ((8, -5), (12, -5)), ((12, 5), (12, -5))
            ])
            # Randomly place survivor in left or right branch
            self.survivor_pos = np.array([10.0, np.random.choice([4.0, -4.0])])

    def step(self, action):
        self.current_step += 1
        self.previous_pos = np.copy(self.uav_pos)
        
        v_forward, v_rate = action
        self.last_v_forward = v_forward # Save for conditional yaw penalty
        self.last_yaw_rate = v_rate # Save for the reward function
        
        self.uav_yaw += v_rate * 0.1
        self.uav_pos[0] += v_forward * np.cos(self.uav_yaw) * 0.1
        self.uav_pos[1] += v_forward * np.sin(self.uav_yaw) * 0.1
        
        # 1. Take a radar reading FIRST
        self.last_scan = self._simulate_argus_scan()
        self.min_wall_dist = float(np.min(self.last_scan))  # Cache — reused by reward + crash check
        
        # 2. Algorithms process the state
        self.path_b_snr = self._calculate_path_b_snr()
        self._update_eskf_covariance()
        self._commander_sensor_toggle()
        self._commander_mission_logic()
        
        reward = self._calculate_reward()
        terminated = self._check_crash() or self.rth_complete
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _commander_mission_logic(self):
        """Dynamic Command Hierarchy: Scout -> Hunt -> RTH"""
        # STATE 3: Return to Home
        if self.command_state == "COMMAND C: RTH":
            self.current_waypoint = np.array([1.0, 0.0]) # Set waypoint back to start
            if np.linalg.norm(self.uav_pos - np.array([1.0, 0.0])) < 1.0:
                self.rth_complete = True # Mission Accomplished

        # STATE 2: Hunt (Target Acquired)
        elif self.path_b_snr > 150.0:
            print(f"\n[SYSTEM] Target Acquired at {self.uav_pos[0]:.2f}, {self.uav_pos[1]:.2f}. Dropping Beacon.")
            self.command_state = "COMMAND C: RTH"
            
        # STATE 2: Hunt (Signal Detected)
        elif self.path_b_snr > self.SNR_HUNT_THRESHOLD:
            self.command_state = "COMMAND B: HUNT"
            self.current_waypoint = self.survivor_pos 
            
        # STATE 1: Scout
        else:
            self.command_state = "COMMAND A: SCOUT"
            if np.linalg.norm(self.uav_pos - np.array([8.0, 0.0])) < 1.5:
                if Point(12.0, 5.0).distance(self.walls) < 0.1: 
                    self.current_waypoint = np.array([8.0, -6.0]) # Go bottom
                else:
                    self.current_waypoint = np.array([8.0, 6.0])  # Go top

    def _update_eskf_covariance(self):
        self.covariance += 0.2
        if self.sensor_mode == "FLOOD":
            self.covariance = max(0.1, self.covariance - 1.5)

    def _commander_sensor_toggle(self):
        """Intelligent ToF Trigger: Look around if an obstacle is near any ray."""
        # Check if the closest return across all 5 rays is dangerously near
        minimum_clearance = self.min_wall_dist
        
        if minimum_clearance < 1.5:
            # Hazard detected on the periphery. Open the beam.
            self.sensor_mode = "FLOOD"
            self.covariance = max(0.1, self.covariance - 1.5)
        elif self.covariance > self.COVARIANCE_THRESHOLD:
            self.sensor_mode = "FLOOD"
        else:
            self.sensor_mode = "THROW"

    def _calculate_path_b_snr(self):
        dist = float(np.linalg.norm(self.uav_pos - self.survivor_pos))
        return 100.0 / (dist**2 + 0.1)

    def _simulate_argus_scan(self):
        if self.sensor_mode == "THROW":
            angles = np.linspace(-math.radians(2.5), math.radians(2.5), 5)
        else:
            angles = np.linspace(-math.radians(30), math.radians(30), 5)

        returns = []
        uav_point = Point(self.uav_pos)
        self.current_rays = [] # For visualization

        for angle in angles:
            ray_angle = self.uav_yaw + angle
            end_x = self.uav_pos[0] + self.MAX_RANGE * math.cos(ray_angle)
            end_y = self.uav_pos[1] + self.MAX_RANGE * math.sin(ray_angle)
            ray = LineString([(self.uav_pos[0], self.uav_pos[1]), (end_x, end_y)])
            intersection = ray.intersection(self.walls)
            
            if intersection.is_empty:
                returns.append(self.MAX_RANGE)
                self.current_rays.append((end_x, end_y))
            else:
                dist = uav_point.distance(intersection)
                returns.append(dist)
                # Calculate exact hit point for the visualizer
                hit_x = self.uav_pos[0] + dist * math.cos(ray_angle)
                hit_y = self.uav_pos[1] + dist * math.sin(ray_angle)
                self.current_rays.append((hit_x, hit_y))
                    
        return np.array(returns, dtype=np.float32)

    def _calculate_reward(self):
        # Carrot: Move to waypoint
        dist_to_waypoint = np.linalg.norm(self.uav_pos - self.current_waypoint)
        prev_dist_to_waypoint = np.linalg.norm(self.previous_pos - self.current_waypoint)
        reward = (prev_dist_to_waypoint - dist_to_waypoint) * 10.0
        
        # Stick 1: Punish yawing only when moving fast AND clear of walls.
        if abs(self.last_v_forward) > 0.2 and self.min_wall_dist > 0.8:
            reward -= abs(self.last_yaw_rate) * 0.1
        
        # Stick 2: Proximity to walls
        if self.min_wall_dist < 0.5:
            reward -= (0.5 - self.min_wall_dist) * 5.0
            
        # Stick 3: Crash
        if self._check_crash():
            reward -= 100.0
            
        # Massive Bonus: Successfully completing RTH
        if self.rth_complete:
            reward += 500.0
            
        return float(reward)

    def _check_crash(self): return self.min_wall_dist <= 0.2
    def _check_target_reached(self): return self.path_b_snr > 150.0
    
    def _get_obs(self):
        scan_returns = self._simulate_argus_scan()
        vector_to_waypoint = self.current_waypoint - self.uav_pos
        dist_to_waypoint = float(np.linalg.norm(vector_to_waypoint))
        
        # Calculate the relative steering angle to the target
        angle_to_target = math.atan2(vector_to_waypoint[1], vector_to_waypoint[0]) - self.uav_yaw
        # Normalize the angle between -pi and pi
        angle_to_target = float((angle_to_target + np.pi) % (2 * np.pi) - np.pi)
        
        return np.array([
            dist_to_waypoint, 
            angle_to_target, 
            self.covariance,
            scan_returns[0], scan_returns[1], scan_returns[2], scan_returns[3], scan_returns[4]
        ], dtype=np.float32)

    def _get_info(self): return {"mode": self.sensor_mode, "state": self.command_state}

# --- THE LIVE VISUALIZER ---
def run_live_sim():
    print("Initializing ARGUS Live Tracking Matrix...")
    env = DynamicEchosEnv()
    
    # Try to load the newly trained v2 brain with RTH and dynamic maze capabilities
    try:
        model = PPO.load("echos_ppo_v2")
        print("[System] Neural weights (v2) loaded successfully.")
    except Exception as e:
        print(f"[Warning] Could not load v2 model (Error: {e}). Running uncalibrated agent.")
        model = PPO("MlpPolicy", env, verbose=0)
    
    obs, info = env.reset()
    
    # Setup Matplotlib Interactive Mode
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    terminated, truncated = False, False
    
    # NEW: Open the telemetry log file
    with open("echos_telemetry.txt", "w") as log_file:
        log_file.write("STEP | STATE | MODE | POS (X,Y) | TARGET (X,Y) | ACTION (V, YAW) | REWARD\n")
        log_file.write("-" * 85 + "\n")
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # NEW: Write current state to the log
            log_line = (f"{env.current_step:04d} | {info['state']} | {info['mode']} | "
                        f"({env.uav_pos[0]:.2f}, {env.uav_pos[1]:.2f}) | "
                        f"({env.current_waypoint[0]:.2f}, {env.current_waypoint[1]:.2f}) | "
                        f"Cmd: [{action[0]:.2f}, {action[1]:.2f}] | Rwd: {reward:.2f}\n")
            log_file.write(log_line)
            
            # Clear frame
            ax.clear()
            
            # 1. Draw Maze Walls
            for line in env.walls.geoms:
                x, y = line.xy
                ax.plot(x, y, color='black', linewidth=3)
                
            # 2. Draw Survivor Acoustic Gradient (Concentric Rings)
            for radius in [0.5, 1.0, 1.5, 2.5]:
                circle = patches.Circle(tuple(env.survivor_pos), radius, color='green', fill=False, alpha=0.3/radius, linewidth=2)
                ax.add_patch(circle)
            ax.plot(env.survivor_pos[0], env.survivor_pos[1], 'go', markersize=10, label='Survivor')
                
            # 3. Draw ARGUS Rays
            ray_color = 'cyan' if info['mode'] == 'THROW' else 'red'
            for ray_end in env.current_rays:
                ax.plot([env.uav_pos[0], ray_end[0]], [env.uav_pos[1], ray_end[1]], color=ray_color, alpha=0.6, linewidth=1.5)
                ax.plot(ray_end[0], ray_end[1], color=ray_color, marker='o', markersize=3, alpha=0.8)
                
            # 4. Draw Drone
            drone_circle = patches.Circle(tuple(env.uav_pos), 0.2, color='blue', zorder=5)
            ax.add_patch(drone_circle)
            # Direction indicator
            dx = 0.5 * math.cos(env.uav_yaw)
            dy = 0.5 * math.sin(env.uav_yaw)
            ax.arrow(env.uav_pos[0], env.uav_pos[1], dx, dy, head_width=0.2, color='white', zorder=6)
            
            # 5. UI Elements
            ax.set_title(f"ARGUS Live Feed | {info['state']} | Sensor: {info['mode']}", fontsize=14, fontweight='bold')
            ax.set_xlim(-2, 14)
            ax.set_ylim(-6, 6)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.pause(0.05) # 20 FPS lock
            
    plt.ioff()
    print("Simulation Terminated. Telemetry written to echos_telemetry.txt.")
    if env._check_crash(): print("Result: Chassis Loss (Crash).")
    elif env._check_target_reached(): print("Result: Target Acquired.")
    plt.show()

if __name__ == "__main__":
    run_live_sim()