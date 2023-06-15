#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:08:48 2021

@author: luca
"""
import numpy as np
import gym
import pygame
import matplotlib.pyplot as plt
import time
import cv2
import os
import glob

#import sys
#sys.path.insert(0, 'C:/Users/trunk/Documents/Python Scripts/car_pedestrian')

from util.agents.pedestrians import Pedestrian, PedestrianSGSFM
from util.agents.vehicles import Vehicle, Road
from util.geometry import Rotation, Angle
from collections import defaultdict
from stable_baselines3.common.env_checker import check_env
        

class Scene:
    def __init__(self, 
                 window_size=(800, 600),
                 **kwargs):
        self.width_m = 30.0
        self.height_m = 15.0
        self.window_size = window_size
        self.real_world_size = (self.width_m, self.height_m)

        self.road = Road(lane_center=self.height_m/2.0)

        self.vehicle = Vehicle(position=[3.0, self.road.vehicle_lane_center],
                               velocity=[4.0, 0],
                               goal=[self.width_m*0.9, self.road.vehicle_lane_center],
                               orientation=self.road.orientation)
        
        ####################################
        ### ADD PEDESTRIAN TO SIMULATION ###
        ####################################
        if 'pedestrian_model' in kwargs:
            if kwargs['pedestrian_model'] == 'SGSFM':
                self.pedestrian = PedestrianSGSFM(
                       position=(self.width_m/2.0, self.road.y_max + self.road.pavement_width/2),
                       goal = (self.width_m/2.0, self.road.y_min - self.road.pavement_width/2)
                       )
         
        else:
            self.pedestrian = Pedestrian(
                position=(self.width_m/2.0, self.road.y_max + self.road.pavement_width/2),
                goal = (self.width_m/2.0, self.road.y_min - self.road.pavement_width/2)
                )
        
        self.awareness_probability = kwargs['p_aware'] if 'p_aware' in kwargs else 1
        self.done_counter = defaultdict(int)
        
        print(f"Pedestrian awareness probability set to: {self.awareness_probability}")
        
        # Store if collision between car and pedestrian occurred
        self.collision_occurred = False
        
        
    def step(self, dt):
        self.vehicle.step(dt)
        self.pedestrian.step(dt, self)
        self.check_pedestrian_collision()
        
        
    def reset(self, spawn_randomly, pedestrian_info=None, vehicle_info=None):
        # Reset vehicle initial position and velocity
        if spawn_randomly:
            x_0v, y_0v, v_0v = self.get_random_vehicle_spawn_info()
            
            start_bottom = (np.random.randint(0,2) == 0)
            self.pedestrian.is_crossing = np.random.binomial(1, 0.9)
            x_0p, y_0p, x_g, y_g = self.get_random_pedestrian_spawn_info(self.pedestrian.is_crossing, start_bottom)
            is_aware = np.random.binomial(1, self.awareness_probability)
            
        else:
            start_bottom = pedestrian_info["is_bottom"]
            self.pedestrian.is_crossing = pedestrian_info["is_crossing"]
            x_0p = pedestrian_info["spawn_x"]
            y_0p = pedestrian_info["spawn_y"]
            x_g = pedestrian_info["goal_x"]
            y_g = pedestrian_info["goal_y"]
            is_aware = pedestrian_info["is_aware"]
            
            x_0v = vehicle_info["spawn_x"]
            y_0v = vehicle_info["spawn_y"]
            v_0v = vehicle_info["initial_velocity"]
        
        
        self.vehicle.reset(initial_position=(x_0v, y_0v), initial_velocity=(v_0v, 0))
        self.pedestrian.reset(initial_position=(x_0p, y_0p), 
                                goal=(x_g, y_g), 
                                is_aware=is_aware)
        self.collision_occurred = False
        
    
    def get_random_pedestrian_spawn_info(self, is_crossing, start_bottom):
        if start_bottom:
            y_0 = self.road.y_min - self.road.pavement_width/2.
            if is_crossing:
                y_g = self.road.y_max + self.road.pavement_width/2.0
            else:
                y_g = y_0
        else:
            y_0 = self.road.y_max + self.road.pavement_width/2.0
            if is_crossing:
                y_g = self.road.y_min - self.road.pavement_width/2.0
            else:
                y_g = y_0                
            
        if is_crossing:
            x_0 = self.width_m/2.0 + np.random.uniform(-3, 3)
            x_g = np.random.uniform(-2, 2) + x_0
        else:
            x_0 = self.width_m/2.0 + np.random.uniform(-10, 10)
            x_g = np.random.uniform(-20, 20) + x_0    
        
        return x_0, y_0, x_g, y_g
    
    
    def get_random_vehicle_spawn_info(self):
        x_0v = 1.0 + self.vehicle.shape.width/2.0
        y_0v = np.random.uniform(low=self.road.y_min + self.vehicle.shape.height/2.0,
                                high = self.road.lane_center - self.vehicle.shape.height/2.0)
        v_0v = np.random.uniform(low=-1.0, high=4.0)
        
        return x_0v, y_0v, v_0v
        
        
    def check_pedestrian_collision(self) -> bool:
        self.collision_occurred = self.vehicle.shape.intersect(self.pedestrian.shape)
    
    
    def check_out_of_lane(self) -> bool:
        # Check if any of the vehicle corners is outside of the road
        # Check upper border of the road
        num_corners_out = np.sum(self.vehicle.shape.corners[:,1] > self.road.y_max)
        if num_corners_out > 0:
            return True
        else:
            num_corners_out = np.sum(self.vehicle.shape.corners[:,1] < self.road.y_min)
            if num_corners_out > 0:
                return True
            else:
                return False
            
            
    def check_out_of_scene(self) -> bool:
        if  self.vehicle.position[0] >= self.width_m or self.vehicle.position[0] < 0  \
            or self.vehicle.position[1] >= self.height_m or self.vehicle.position[1] < 0:
            return True
        else:
            return False
        
        
    def check_lane_invasion(self) -> int:
        # Returns: number of vehicle corners invading the lane
        return np.sum(self.vehicle.shape.corners[:,1] > self.road.lane_center)
        
        
    # Map coordinate to pixels    
    def m2p(self, x, y, window_size):
        width, _ = window_size
        c = width / self.width_m
        return (int(x*c), int((self.height_m - y)*c))
    
    
    def draw_arrow(self, image, p1, angle, length, color, width):
        length = length / 2.0
        p2 = p1 + Rotation(angle)*np.array([length,0])
        pygame.draw.line(image, color, 
                         self.m2p(*p1, image.get_size()),
                         self.m2p(*p2, image.get_size()), 
                         width=width)
        
        arrow_points = np.array([[0.2, 0], 
                                 [-0.1, -0.1], 
                                 [0,0], 
                                 [-0.1, 0.1]])
        
        points = p1 + Rotation(angle)*(arrow_points*np.sqrt(length) + np.array([1,0])*length)
        pygame.draw.polygon(image, color, [self.m2p(*p, image.get_size()) for p in points])
       
        
    def draw_line_dashed(self, surface, color, 
                         start_pos, end_pos, width = 1, 
                         dash_length = 10, exclude_corners = True):

        # convert tuples to numpy arrays
        start_pos = np.array(start_pos)
        end_pos   = np.array(end_pos)
    
        # get euclidian distance between start_pos and end_pos
        length = np.linalg.norm(end_pos - start_pos)
    
        # get amount of pieces that line will be split up in (half of it are amount of dashes)
        dash_amount = int(length / dash_length)
    
        # x-y-value-pairs of where dashes start (and on next, will end)
        dash_knots = np.array([np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)]).transpose()
    
        return [pygame.draw.line(surface, color, tuple(dash_knots[n]), tuple(dash_knots[n+1]), width)
                for n in range(int(exclude_corners), dash_amount - int(exclude_corners), 2)]
        
    
    def render(self):
        """
        Parameters
        ----------
        window_size : 2-D Tuple, optional
            DESCRIPTION. The default is (800, 600).

        Returns
        -------
        image : pygame image of the scene
            DESCRIPTION.

        """
        width, height = self.window_size
        image = pygame.Surface(size=self.window_size)
        
        BLACK = (0,0,0)
        RED = (150, 0, 0)
        GREEN = (11, 102, 35)
        BLUE = (0, 0, 255)
        
        LIGHT_GREY = (190, 190, 190)
        DARK_GREY = (100, 100, 100)
        WHITE = (255, 255, 255)
        
        c = width / self.width_m
        image.fill(GREEN)
        
        LINEWIDTH = 2
        
        # Draw road structure
        road_rect = (*self.m2p(0, self.road.y_max, self.window_size), 
                     c*self.width_m, c*self.road.lane_width)
        image.fill(LIGHT_GREY, 
                    rect=road_rect)
        
        # Draw upper pavement
        pavement_rect_1 = (*self.m2p(0, self.road.y_max + self.road.pavement_width, self.window_size), 
                           c*self.width_m, c*self.road.pavement_width)
        image.fill(DARK_GREY, 
                    rect=pavement_rect_1)
        
        # Draw lower pavement
        pavement_rect_2 = (*self.m2p(0, self.road.y_min, self.window_size), 
                           c*self.width_m, c*self.road.pavement_width)
        image.fill(DARK_GREY, 
                    rect=pavement_rect_2)
        
        # Draw borders
        self.draw_line_dashed(image, WHITE, 
                              self.m2p(0, self.road.lane_center, self.window_size), 
                              self.m2p(self.width_m, self.road.lane_center, self.window_size),
                              width=LINEWIDTH*2,
                              dash_length=1.0*c)
        pygame.draw.line(image, BLACK, 
                          self.m2p(0, self.road.y_min, self.window_size), 
                          self.m2p(self.width_m, self.road.y_min, self.window_size),
                          width=LINEWIDTH*2)
        pygame.draw.line(image, BLACK, 
                          self.m2p(0, self.road.y_max, self.window_size), 
                          self.m2p(self.width_m, self.road.y_max, self.window_size),
                          width=LINEWIDTH*2)
        
        corners = [self.m2p(c[0], c[1], self.window_size) for c in self.vehicle.shape.corners]
        pygame.draw.polygon(image, RED, corners)
        self.draw_arrow(image, self.vehicle.position, 
                        self.vehicle.orientation,
                        length=np.linalg.norm(self.vehicle.velocity), 
                        color=BLUE, 
                        width=3)
        
        
        # Draw Pedestrian
        self.pedestrian.render(image, 
                               position_pixel=self.m2p(self.pedestrian.position[0], 
                                           self.pedestrian.position[1], self.window_size),
                               window_size=self.window_size,
                               real_world_size=self.real_world_size)
        
        
        self.draw_arrow(image, 
                        self.pedestrian.position, 
                        self.pedestrian.get_direction(), 
                        length=np.linalg.norm(self.pedestrian.velocity),
                        color=BLUE, 
                        width=3)
        
        
        # Draw Pedestrian Goal
        pygame.draw.circle(image, GREEN, 
                           center=self.m2p(self.pedestrian.goal[0], 
                                           self.pedestrian.goal[1], self.window_size),
                           radius=c*.2)
                
        return image
    
    
class CarEnv(gym.Env):
    def __init__(self, 
                 fps=10,
                 window_size=(1600, 1200),
                 render=True,
                 reward_fn=None,
                 training=True,
                 svo = 0,
                 **kwargs):
        super(CarEnv, self).__init__()
        
        self.training = training
        
        self.fps = fps
        self.dt = 1.0 / self.fps
        self.timestep = 0
        
        self.scene = Scene(window_size, **kwargs)
        self.window_size = window_size
        
        # Action and Observation space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), 
                                           dtype=np.float32)
        
        self.svo = Angle(svo, units='deg')
        
        # Observation consists of:
        # - vehicle distance from its path
        # - vehicle angular deviation from its path
        # - vehicle speed
        # - obstacle position relative to the vehicle
        # - obstacle velocity relative to the vehicle
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(7,), dtype=np.float32)
        
            
        self.reward_fn = reward_fn if reward_fn is not None else CarEnv.default_reward_fn
        
        # Manage kwargs
        self.track_states = kwargs['track_states'] if 'track_states' in kwargs else False
        self.visited_states = np.zeros(shape=self.window_size) if self.track_states else None
       
        # Rendering options
        self.show = render
        if self.show:
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
        
        # Recording options
        self.record_video = kwargs['record_video'] if 'record_video' in kwargs else False
        self.video_name = kwargs['video_name'] if 'video_name' in kwargs else None
        if self.record_video:
            self.frame_counter = 0
            if not os.path.isdir('./videos/'):
                os.mkdir('./videos/')
            if not os.path.isdir('./images/'):
                os.mkdir('./images/')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.recorder = cv2.VideoWriter("./videos/output.avi", fourcc, fps, window_size)
        

    def __getattr__(self, name):
        """Relay missing methods and members to underlying scene object"""
        return getattr(self.scene, name)
    
    
    def get_observation(self):
        n = self.scene.vehicle.goal - self.scene.vehicle.initial_position
        mu = np.array([-n[1], n[0]])/np.sqrt(n[0]**2 + n[1]**2)
        
        OP = self.scene.vehicle.position - self.scene.vehicle.initial_position
        distance_from_lane = np.dot(mu, OP)
        
        return np.array([distance_from_lane, 
                         self.scene.vehicle.orientation.to180(), 
                         np.linalg.norm(self.scene.vehicle.velocity),
                         *self.scene.vehicle.transform.toLocal(self.scene.pedestrian.position),
                         *self.scene.vehicle.relative_speed(self.scene.pedestrian.velocity)])
        
    
    def reset(self, pedestrian_info=None, vehicle_info=None):
        if self.training:
            self.scene.reset(spawn_randomly=True)
        else:
            # During testing, we want to pass the pedestrian spawn spawn point
            # and goal explicitly, so that they are consistent in the testing environment.
            # whereas during training the should be generated automatically
            self.scene.reset(spawn_randomly=False, pedestrian_info=pedestrian_info, vehicle_info=vehicle_info)
        
        self.done = False
        self.goal_reached = False
        self.timestep = 0
        self.total_reward = 0
        self.start_t = time.time()
        self.action = np.zeros(shape=self.action_space.shape)
        self.previous_action = np.zeros(shape=self.action_space.shape)
        
        return self.get_observation()
    
    
    def step(self, action):
        self.action = action
        # self.scene.vehicle.apply_control(*action) # XXX
        self.scene.vehicle.apply_control(0, *action)
        
        # Perform one time step integration
        self.timestep += 1
        self.scene.step(self.dt)
        
        reward, self.done = self.reward_fn(self) if callable(self.reward_fn) else (0, False)
        self.previous_action = action
        self.total_reward += reward
    
        if self.track_states and not self.done:
            self.visited_states[self.m2p(*self.vehicle.position, self.window_size)] = 255
            
        return self.get_observation(), reward, self.done, {}
    
    
    def close(self):
        pygame.quit()
        
        # Close video recorder and save video
        if self.record_video:
            i = 0
            while os.path.isfile(f"./videos/output_{i:03}.avi"):
                i += 1
            print(self.video_name)
            
            if self.video_name is not None:
                video_name = self.video_name
            else:
                video_name = f"./videos/output_{i:03}.avi"
            
            print(f"Writing video to: {video_name}")
            writer = cv2.VideoWriter(video_name,
                                     cv2.VideoWriter_fourcc(*'DIVX'), 
                                     self.fps, 
                                     self.window_size)
            
            for filename in sorted(glob.glob('./images/tmp*.png')):
                im = cv2.imread(filename)
                writer.write(im)
                
            for filename in glob.glob('./images/tmp*.png'):
                os.remove(filename)
            
            writer.release()
        
    def plot_visited_states(self):
        if self.track_states:
            plt.imshow(self.visited_states.T)
            
    def render(self, mode='human', speed=1):
        image = self.scene.render()
        
        if self.show:
            self.screen.blit(image, (0, 0))
            
            # Flip the display
            pygame.display.update()
            time.sleep(self.dt/speed)
            
        if self.record_video:
            self.frame_counter += 1
            pygame.image.save(image, f"./images/tmp_{self.frame_counter:05}.png")
            
    @staticmethod
    def default_reward_fn(env):
        reward = 0
        done = env.check_out_of_lane() \
                or env.check_out_of_scene() \
                or env.check_pedestrian_collision() \
                or (time.time() - env.start_t > 20.0)
        
        return reward, done
            
        
            
        
    
    
if __name__=="__main__":
    import time
    RUN_CONTINUOUSLY = True
    
    try:
        window_size = (1200, 600)
        env = CarEnv(fps=30, 
                     window_size=window_size, 
                     record_video=False,
                     render=True,
                     pedestrian_model='SGSFM',
                     p_aware=0.0)
        # env = CarEnv(fps=30, window_size=window_size, 
        #              render=True)
        check_env(env)
        
        # Run until the user asks to quit
        running = True
        env.reset()
        
        if not RUN_CONTINUOUSLY:
            while running:
                    
                # Did the user click the window close button?
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_RIGHT:
                                action = env.action_space.sample() 
                                obs, _, _, _ = env.step(action)
                                
                                if env.scene.check_out_of_scene() or env.scene.check_out_of_lane() or env.scene.check_pedestrian_collision():
                                    env.reset()
                        
                env.render()
        else:
            while running:
                env.render(speed=1.0)
                action = env.action_space.sample() 
                print(action)
                obs, reward, done, _ = env.step(action)
                
                if done:
                    env.reset()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        env.close()
                        
                                
                    
    finally:
        pygame.quit()