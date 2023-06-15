#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:01:43 2021

@author: luca
"""
from environment import CarEnv
from train_mlp import reward_fn, social_reward_fn
from stable_baselines3 import SAC
import pygame
import os
import pandas as pd

if __name__=="__main__":
    try:
        training_instance = "TRAINING_20000_eval2"
        source_folder = os.path.join("./results/", training_instance)
        episode = 11
        svo = "svo_80"
        video_name = f"./videos/ep_{episode}_{svo}.avi"
        
        env = CarEnv(window_size = (800, 400), 
                     record_video=True,
                     render=True, 
                     reward_fn=social_reward_fn,
                     video_name=video_name,
                     training=False,
                     pedestrian_model='SGSFM')
        model = SAC.load(f'./logs/TRAINING_20000/{svo}/best_model', env=env)
        
        
        ped_info = pd.read_pickle(os.path.join(source_folder, "pedestrian_info.pkl")).loc[episode]
        veh_info = pd.read_pickle(os.path.join(source_folder, "vehicle_info.pkl")).loc[episode]
        
        obs = env.reset(pedestrian_info=ped_info, vehicle_info=veh_info)
        done = False
        while not done:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    env.close()
            
            if done:
                print(f"Total reward: {env.total_reward}")
                print(f"Total timesteps: {env.timestep}")
                running = False
                env.close()
                
    finally:
        pygame.quit()
