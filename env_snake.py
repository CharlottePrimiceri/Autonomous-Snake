import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time
from make_maze import create_maze_layout, get_valid_spawn_points, astar
from mujoco_tools import make_maze_on_mujoco
from snake_cpg import PaperCPG

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()
        

        self.cpg_frequency = 50
        self.rl_frequency = 0.1
        self.sim_steps_per_rl_step = int(self.cpg_frequency / self.rl_frequency)
        self.max_episode_steps = 300 
        self.R = 1.0 # Target amplitude R
        self.omega = 1.0 # Base omega ω
        self.delta = 0.0 # Offset δ

        self.start_pos_maze = (1, 1)
        #self.init_qpos = np.zeros(12, dtype=np.float32)  
        self.base_xml_path = 'scenes/scene.xml'
        self.maze_xml_path = 'scenes/scene_maze_trial.xml'


        self.action_space = spaces.Box(low=np.array([0.5]), high=np.array([1.0]), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
        self.maze_layout = create_maze_layout(3, 7)
        make_maze_on_mujoco(
            load_file_path=self.base_xml_path,
            maze=np.array(self.maze_layout),
            start_pos=self.start_pos_maze,
            goal_pos=self.start_pos_maze, # Temporary goal(waypoint)
            save_file_path=self.maze_xml_path
        )

        self.cpg = PaperCPG(n_joints=12, dt=1.0/self.cpg_frequency)
        self.cpg.set_hyper_parameters(convergence_rate=63.0, coupling_strength=3.0)
        
        self.path_waypoints_world = []
        self.current_waypoint_index = 0
        self.waypoint_threshold = 0.7
        self.cumulative_waypoint_reward = 0.0

        self.reached_waypoints = []
        
        self.render_mode = render_mode
        self.viewer = None

    def _maze_to_world(self, maze_pos):
        """ maze coordinates in mujoco coordinates"""
        pos_x = 2 * 0.5 * (maze_pos[0] - self.start_pos_maze[0])
        pos_y = -2 * 0.5 * (maze_pos[1] - self.start_pos_maze[1])
        return np.array([pos_x, pos_y, 0.15])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.viewer is not None:
            self.close()
        self.maze_layout = create_maze_layout(3, 7)
        self.valid_spawn_points_maze = get_valid_spawn_points(self.maze_layout)
        # Random Goal 
        goal_idx = self.np_random.integers(0, len(self.valid_spawn_points_maze))
        goal_pos_maze = self.valid_spawn_points_maze[goal_idx]
        
        # A* path towards new goal
        path_maze = astar(self.maze_layout, self.start_pos_maze, goal_pos_maze)
        
        # New xml with new goal and waypoints  
        make_maze_on_mujoco(
            load_file_path=self.base_xml_path,
            maze=np.array(self.maze_layout),
            start_pos=self.start_pos_maze,
            goal_pos=goal_pos_maze,
            waypoints=path_maze,
            save_file_path=self.maze_xml_path
        )
        
        self.model = mujoco.MjModel.from_xml_path(self.maze_xml_path)
        self.data = mujoco.MjData(self.model)
        #self.data.qpos[-12:] = self.init_qpos
        #self.data.qpos[3:7] = np.array([1, 0, 0, 0])
        # Waypoint in world coordinate 
        self.path_waypoints_world = [self._maze_to_world(p) for p in path_maze]
        #print(f"Path waypoints (world coordinates): {self.path_waypoints_world}")
        self.current_waypoint_index = 0
        
        # Reset of reached waypoints 
        self.reached_waypoints = []
        self.cumulative_waypoint_reward = 0.0

        # Save goal position in world coordinate 
        self.goal_pos_world = self._maze_to_world(goal_pos_maze)
        if len(self.path_waypoints_world) > 0:
           self.path_waypoints_world[-1] = self.goal_pos_world.copy()
        # Reset state of snake and CPG 
        self.current_step = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        
        return self._get_obs(), {}
    def _get_snake_heading_angle(self, target_pos):

        # head position 
        head_pos = self.data.body('frame_0-1').xpos
        
        # rotation matrix of the head
        head_rotation_matrix = self.data.body('frame_0-1').xmat.reshape(3, 3)
        
        # principal axis of the snake in the world
        snake_axis = head_rotation_matrix[:, 0]  
        
        # vector from head to target
        vec_to_target = target_pos - head_pos
        vec_to_target_2d = vec_to_target[:2]  
        snake_axis_2d = snake_axis[:2]        
        
        # normalization
        vec_to_target_norm = vec_to_target_2d / (np.linalg.norm(vec_to_target_2d) + 1e-6)
        if vec_to_target_norm[0] == 0 and vec_to_target_norm[1] == 0:
            return np.pi/2
        snake_axis_norm = snake_axis_2d / (np.linalg.norm(snake_axis_2d) + 1e-6)
        
        # angle in rad
        dot_product = np.dot(snake_axis_norm, vec_to_target_norm)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # sign of the angle
        cross_product = np.cross(snake_axis_norm, vec_to_target_norm)
        if cross_product < 0:
            angle_rad = -angle_rad
    
        return angle_rad
    def _get_current_target(self):
        if self.path_waypoints_world:
            i = self.current_waypoint_index
            if i < len(self.path_waypoints_world) - 1:
                w1 = self.path_waypoints_world[i]
                w2 = self.path_waypoints_world[i + 1]
                return 0.5 * (w1 + w2)
            elif i < len(self.path_waypoints_world):
                return self.path_waypoints_world[i]
        return self.goal_pos_world
    def _get_obs(self):
        joint_pos = self.data.qpos[-12:]
        joint_vel = self.data.qvel[-12:]
        head_pos = self.data.body('frame_0-1').xpos
        current_target = self._get_current_target()
        head_to_target_vec = current_target - head_pos

        obs = np.concatenate([joint_pos, joint_vel, head_to_target_vec]).astype(np.float32)


        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            raise ValueError(f"[SnakeEnv] Invalid observation detected: {obs}")

        # clip observation to safe range
        obs = np.clip(obs, -1000.0, 1000.0)

        return obs
    def step(self, action):
        """Performs an environment step."""
        self.current_step += 1
        print(self.current_step)
        # Parse the action and set the CPG parameters
        theta = action
        self.cpg.set_parameters(R=self.R, omega = self.omega, theta=theta, delta=self.delta)

        # Retrieve the current waypoint or goal
        current_target = self._get_current_target()
        head_pos = self.data.body('frame_0-1').xpos
        distance_to_waypoint = np.linalg.norm(current_target - head_pos)

        goal_reward = 0.0

        joint_range = self.model.jnt_range[-12:]
        for _ in range(self.sim_steps_per_rl_step):
            target_positions = self.cpg.update()
            clipped_targets = np.clip(target_positions, joint_range[:, 0], joint_range[:, 1])
            self.data.ctrl[:] = clipped_targets

            mujoco.mj_step(self.model, self.data)

        new_head_pos = self.data.body('frame_0-1').xpos
        new_distance_to_waypoint = np.linalg.norm(current_target - new_head_pos)
        r2 = distance_to_waypoint - new_distance_to_waypoint

     
        if new_distance_to_waypoint < 0.7:  
            if self.current_waypoint_index < len(self.path_waypoints_world):
                reached_waypoint = self.path_waypoints_world[self.current_waypoint_index].copy()
                
                already_reached = any(np.allclose(reached_waypoint, existing_wp, atol=1e-3) 
                                    for existing_wp in self.reached_waypoints)
                
                if not already_reached:
                    self.reached_waypoints.append(reached_waypoint)
                    print(f"Waypoint {self.current_waypoint_index} reached! Position: {reached_waypoint}")
                    
                    self.current_waypoint_index += 1
                    waypoint_bonus = 300.0  
                    self.cumulative_waypoint_reward += waypoint_bonus
                    # update target
                    current_target = self._get_current_target()
                    print(f"New target: {current_target}")

        self_collision_penalty = 0
        geom_names = [self.model.geom(i).name for i in range(self.model.ngeom)]
        snake_geom_ids = {i for i, name in enumerate(geom_names) if 'frame' in name}

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            if geom1_id in snake_geom_ids and geom2_id in snake_geom_ids:
                self_collision_penalty -= 5
                break
        angle_to_target = self._get_snake_heading_angle(current_target)
        head_pos = self.data.body('frame_0-1').xpos
        tail_pos = self.data.body(f'frame{self.cpg.n_joints}_1').xpos  # Assicurati che il nome sia corretto!

        # Distanza euclidea tra testa e coda
        head_tail_dist = np.linalg.norm(head_pos - tail_pos)

        # Penalità se la distanza è troppo corta (ad esempio, < 0.5)
        min_dist = 0.5
        penalty = 0.0
        if head_tail_dist < min_dist:
            penalty = -100.0 * (min_dist - head_tail_dist)  # Penalità proporzionale
        # max reward when o angle=0, minimum when angle=π
        r_angle_alignment = np.cos(angle_to_target)
        r1 = 1.0 / (0.1 + new_distance_to_waypoint)
         
        r3 = -0.3 * np.linalg.norm(action - self.last_action)
        control_penalty = -0.01 * np.square(self.data.ctrl).mean()
        wall_penalty = 0
        if self._check_wall_collision():
            wall_penalty = -75

        # Check goal termination
        terminated = np.linalg.norm(new_head_pos - self.goal_pos_world) < 0.7
        
        if terminated: 
            print("Goal reached!")
            goal_reward = 500.0

        reward = (10.0 * r1) + (2000.0*r2) + (0.1) * r3 + self.cumulative_waypoint_reward + goal_reward + control_penalty + (100)*r_angle_alignment #+ 20)penalty #+ wall_penalty    
        
        truncated = self.current_step >= self.max_episode_steps
        self.last_action = action
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, bool(terminated), bool(truncated), {}

    def _check_wall_collision(self):
     for i in range(self.data.ncon):
        contact = self.data.contact[i]
        geom1 = self.model.geom(contact.geom1).name
        geom2 = self.model.geom(contact.geom2).name
        
        # Customize these names to match your XML
        if ("wall" in geom1 and "frame" in geom2) or ("wall" in geom2 and "frame" in geom1):
            return True
     return False
    def render(self):
        if self.viewer is None and self.data is not None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
            time.sleep(1.0 / self.metadata['render_fps'])
        elif self.viewer:
            self.close()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None






