# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:54:35 2021

@author: trunk
"""
from environment import CarEnv
from train_mlp import reward_fn, social_reward_fn
from stable_baselines3 import SAC
import pygame
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    try:
        for x in range(9):
            env = CarEnv(window_size = (800, 400), 
                         record_video=False,
                         render=False, 
                         reward_fn=social_reward_fn,
                         pedestrian_model='SGSFM')
            model = SAC.load(f'./logs/svo_{x}0', env=env)
            
            NUM_EVAL_EPISODES = 100
            MAX_LEN = 200
            
            goal_reached = np.zeros(shape=(NUM_EVAL_EPISODES,), dtype=bool)
            time_to_reach_goal = np.zeros(shape=(NUM_EVAL_EPISODES,), dtype=np.float32)
            ep_lengths = np.zeros(shape=(NUM_EVAL_EPISODES,), dtype=np.uint8)
            collision_occurred = np.zeros(shape=(NUM_EVAL_EPISODES,), dtype=bool)
            collision_relative_speed = np.zeros(shape=(NUM_EVAL_EPISODES,), dtype=np.float32)
            # acceleration = np.zeros(shape=(NUM_EVAL_EPISODES, MAX_LEN), dtype=np.float32)
            
            for i in range(NUM_EVAL_EPISODES):
                obs = env.reset()
                counter = 0
                done = False
                while not done:
                    env.render()
                    action, _ = model.predict(obs, deterministic=True)
                    
                    obs, reward, done, _ = env.step(action)
                    
                    # if counter < MAX_LEN:
                    #     acceleration[i, counter] = env.vehicle.get_acceleration()
                    #     counter += 1
                    
                    if done:
                        goal_reached[i] = env.vehicle.goal_reached
                        collision_occurred[i] = env.collision_occurred
                        ep_lengths[i] = env.timestep
                        
                        if env.vehicle.goal_reached:
                            time_to_reach_goal[i] = env.timestep * env.dt
                            
                        if env.collision_occurred:
                            car_speed = env.vehicle.velocity
                            ped_speed = env.pedestrian.velocity
                            rel_speed = car_speed - ped_speed
                            
                            collision_relative_speed[i] = np.linalg.norm(rel_speed)
                        
                        env.reset()
            
            print(f"svo_{x}0")
            print(f"Number of collisions: {env.done_counter['collision']}")
            print(f"Number of times goal was reached: {env.done_counter['goal reached']}")
            print(f"Collision percentage: {100.0*env.done_counter['collision']/NUM_EVAL_EPISODES}")
            print(f"Average time to reach goal: {np.mean(time_to_reach_goal)}")
            print(f"Time to reach goal standard deviation: {np.std(time_to_reach_goal)}")
            print("\n\n")
        
        # a0 = acceleration[0, :ep_lengths[0]]
        # t0 = np.arange(0, ep_lengths[0]) * env.dt
        # plt.plot(t0, a0)
   
    finally:
        pygame.quit()
