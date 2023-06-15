#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:22:28 2021

@author: luca
"""
from environment import CarEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import pygame
import time
from typing import Callable
import os
import json
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        # if progress_remaining < 0.1:
        #     return 0.1*initial_value
        # else:
        return progress_remaining * initial_value

    return func


def reward_fn(env):
    done = False
    
    if env.collision_occurred == True or env.check_out_of_scene():
        done = True
        env.done_counter['collision'] += 1
        current_reward = -30
    # elif np.linalg.norm(env.vehicle.position - env.vehicle.goal) < 1.0:
    elif env.vehicle.position[0] >= env.vehicle.goal[0]:
        done = True
        env.done_counter['goal reached'] += 1
        current_reward = 30 + 7*np.exp(-(env.vehicle.position[1] - env.road.vehicle_lane_center)**2/0.5)
    elif env.check_out_of_lane():
        done = True
        env.done_counter['out_of_lane'] += 1
        current_reward = -10
    elif time.time() - env.start_t > 10.0 and np.linalg.norm(env.vehicle.velocity) < 0.01:
        done = True
        env.done_counter['timeout'] += 1
        current_reward = -10
    else:
        num_corners_invading = env.check_lane_invasion()
        angular_deviation = np.abs(env.vehicle.orientation.to180())
        
        forward_direction = env.vehicle.goal - np.array([0, env.road.vehicle_lane_center])
        speed_reward = 0.1 * np.sign(forward_direction.dot(env.vehicle.velocity)) * np.linalg.norm(env.vehicle.velocity)
        
        current_reward = speed_reward   \
                        - 0.01*angular_deviation                     \
                        - 0.4*num_corners_invading 
    
    return current_reward, done


def social_reward_fn(env):
    done = False
    
    if env.collision_occurred == True or env.check_out_of_scene():
        done = True
        env.done_counter['collision'] += 1
        car_reward = -30
    # elif np.linalg.norm(env.vehicle.position - env.vehicle.goal) < 1.0:
    elif env.vehicle.goal_reached == True:
        done = True
        env.done_counter['goal reached'] += 1
        car_reward = 30 + 7*np.exp(-(env.vehicle.position[1] - env.road.vehicle_lane_center)**2/0.5)
    elif env.check_out_of_lane():
        
        done = True
        env.done_counter['out_of_lane'] += 1
        car_reward = -10
    elif env.timestep*env.dt > 15.0: #and np.linalg.norm(env.vehicle.velocity) < 0.01:
        done = True
        env.done_counter['timeout'] += 1
        car_reward = -10
    else:
        # num_corners_invading = env.check_lane_invasion()
        # angular_deviation = np.abs(env.vehicle.orientation.to180())
        
        forward_direction = env.vehicle.goal - np.array([0, env.road.vehicle_lane_center])
        speed_reward = 0.4 * np.sign(forward_direction.dot(env.vehicle.velocity)) * np.linalg.norm(env.vehicle.velocity)
        
        car_reward = speed_reward  # \
                        # - 0.01*angular_deviation                     \
                        # - 0.4*num_corners_invading  \
    
    ped_reward = 0
    if env.pedestrian.is_crossing:
        if env.pedestrian.get_distance_to_goal() > 0.1 and (env.pedestrian.position[0] > env.vehicle.position[0]):
            ped_reward = 0.5*np.linalg.norm(env.pedestrian.velocity)
        
    current_reward = env.svo.cos()*car_reward + env.svo.sin()*ped_reward 
    return current_reward, done


if __name__=="__main__":
    svo_values = [50, 60, 70]
    TOTAL_TIMESTEPS = 20_000
    training_instance_name = f"TRAINING_{TOTAL_TIMESTEPS}"
    p_aware = 0.9
    linear_schedule_value = 0.001
    
    # Create directories
    if not os.path.isdir(f"./logs/{training_instance_name}"):
        os.mkdir(f"./logs/{training_instance_name}")
        
    for svo in svo_values:
        if not os.path.isdir(f"./logs/{training_instance_name}/svo_{svo}/"):
            os.mkdir(f"./logs/{training_instance_name}/svo_{svo}/")

    # Update metadata json file if it exists    
    metadata_fn = f"./logs/{training_instance_name}/metadata.json"   
    print("Trying to open " + metadata_fn)
    try:
        with open(metadata_fn) as f:
            metadata = json.load(f)
    except:
        print(metadata_fn + " not found")
        metadata = {"path": [], "svo": [], "awareness probability": 0,
                    "linear schedule": 0}
    
    for svo in svo_values:
        if not svo in metadata["svo"]:
            metadata["path"].append(f"svo_{svo:02}")
            metadata["svo"].append(svo)
            metadata["awareness probability"] = p_aware
            metadata["linear schedule"] = linear_schedule_value  
    
    with open(metadata_fn, "w") as f:
        f.write(json.dumps(metadata, indent=4))

    # Start Trainingonly for new values  
    for svo in svo_values:
        print(f"\n\nCurrent SVO value: {svo}")
        try:
            start = time.time()
            env = CarEnv(reward_fn=social_reward_fn, 
                         track_states=True, 
                         render=False,
                         pedestrian_model='SGSFM',
                         training=True,
                         p_aware=p_aware,
                         svo=svo)
            
            # TRAINING STARTS HERE
            action_noise = NormalActionNoise(np.zeros(shape=env.action_space.shape),
                                              np.zeros(shape=env.action_space.shape) + 2.0)

            model = SAC('MlpPolicy', 
                        env, 
                        learning_rate=linear_schedule(linear_schedule_value),
                        verbose=1,
                        tensorboard_log="./runs/",
                        action_noise=action_noise)
                        # use_sde=True)
            callbacks=[]
            
            # Save the model
            checkpoint_callback = CheckpointCallback(save_freq=TOTAL_TIMESTEPS//5, 
                                                      save_path=f"./logs/{training_instance_name}/svo_{svo}/",
                                                      name_prefix='sac')
            callbacks.append(checkpoint_callback)
            
            # Evaluate model
            eval_callback = EvalCallback(model.get_env(), 
                                          verbose=1,
                                          best_model_save_path=f'./logs/{training_instance_name}/svo_{svo}',
                                          log_path=f'./logs/{training_instance_name}/svo_{svo}', 
                                          eval_freq=TOTAL_TIMESTEPS//20,
                                          n_eval_episodes=3,
                                          deterministic=True, 
                                          render=True)
            callbacks.append(eval_callback)
            
            
            model.learn(total_timesteps=TOTAL_TIMESTEPS, 
                        callback=callbacks,
                        tb_log_name=f"SVO_{svo}")
            # print(f"Counter: {env.done_counter}")
            # env.plot_visited_states()
            
            end = time.time()
            print(f"Finished training with svo {env.svo}, it took {(end - start)/60.0:.1f} minutes")
        finally:
            pygame.quit()
