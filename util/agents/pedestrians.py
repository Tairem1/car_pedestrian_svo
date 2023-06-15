#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:45:36 2021

@author: luca
"""
from util.geometry import Rectangle, Circle, Angle, Segment
from util.geometry import m2p
import numpy as np
import pygame
import os

class Pedestrian:
    idCounter=0
    
    def __init__(self, 
                 position=(0,0),
                 velocity = (0,0),
                 goal=(0,0)):
        Pedestrian.idCounter += 1
        self.id = Pedestrian.idCounter
        
        self.initial_position = np.array(position, dtype=np.float32)
        self.initial_velocity = np.array(velocity, dtype=np.float32)
        self.position = np.copy(self.initial_position)
        self.velocity = np.copy(self.initial_velocity)
        self.goal = np.array(goal)
        
        self.radius = .5
        self.shape = Circle(self.position, self.radius)
        
#        module_path = os.path.split(car_pedestrian.__file__)[0]
        module_path = './'
        sprite_path = os.path.join(module_path, 'util/agents/pedestrian.png')
        self.sprite = pygame.image.load(sprite_path)
        self.sprite_size_m = 1.0
        
        self.is_crossing = False
        self.is_aware = True
        
        self.desired_speed = 1.8


    def set_goal(self, goal):
        self.goal = np.array(goal)
        
        
    def get_direction(self) -> Angle:
        return Angle(np.arctan2(self.velocity[1], self.velocity[0]), 'rad')
    
    
    def reset(self, initial_position=None, 
              initial_velocity=None, 
              goal=None,
              is_aware=True):
        if initial_position is not None:
            self.initial_position[:] = initial_position
            
        if initial_velocity is not None:
            self.initial_velocity[:] = initial_velocity
            
        if goal is not None:
            self.goal[:] = goal
        
        self.position[:] = self.initial_position
        self.velocity[:] = self.initial_velocity
        self.is_aware = is_aware
        
        
    def step(self, dt, environment=None):
        self.position += self.velocity * dt
        self.shape = Circle(self.position, self.radius)
        
        direction = self.goal - self.position
        distance_to_goal = np.linalg.norm(direction)
        if (distance_to_goal < 0.01):
            self.velocity = np.array([0,0], dtype=np.float32)
        else:
            desired_speed = self.desired_speed * direction / distance_to_goal
            K = 2.0
            self.velocity += K * (desired_speed - self.velocity) * dt 
            # self.velocity = desired_abs_speed * direction / distance_to_goal
            
            
    def render(self, 
               image, 
               position_pixel, 
               window_size, 
               real_world_size, 
               color=(255, 125, 0)):
        ORANGE = (255, 125, 0)
        scale_factor = window_size[0] / real_world_size[0]
        new_size_x = int(self.sprite_size_m*scale_factor)
        new_size_y = int(new_size_x * self.sprite.get_height() / self.sprite.get_width())
        new_size = (new_size_x, new_size_y)
        
        # Draw Pedestrian
        pygame.draw.circle(image, ORANGE, 
                           center=position_pixel,
                           radius=scale_factor*self.shape.radius)
        
        angle = np.arctan2(*self.velocity[::-1]) * 180.0 / np.pi
        scaled_sprite = pygame.transform.scale(self.sprite, new_size)
        rotate_sprite = pygame.transform.rotate(scaled_sprite, angle - 90)
        center = np.array(position_pixel, dtype=np.int32)
        offset = np.array((-new_size[0]/2, -new_size[1]/2), dtype=np.int32)
        corner = center + offset
        
        image.blit(rotate_sprite, corner.astype(int))
            
        
    
    def __str__(self):
        return (f"---- Pedestrian {self.id:02} ----\n"
        f"| Position: {self.position}\n"
        f"| Velocity: {self.velocity}\n"
        f"| Goal: {self.goal}\n"
        f"------------------------")
    
    
class PedestrianSGSFM(Pedestrian):
    """
    Same as Pedestrian class but update method is based on Sub-Goal Social 
    Force Model from https://arxiv.org/pdf/2101.03554.pdf paper
    """ 
    def __init__(self,  
                 position=(0,0),
                 velocity = (0,0),
                 goal=(0,0)):
        super().__init__(position, velocity, goal)
        
        # Initialize SGSFM parameters
        self.tau_x = 2.00
        self.d_x = 0.50
        self.beta_veh = 3.60
        self.K_nav = 250.0
        self.m = 60.0       # Person mass
        self.N_j = 80
        self.d_nav = 3.0
        
        self.M_veh = 1_000_000    # TODO Justify this value
        self.sigma = 1     # TODO Justify this value
        self.desired_speed = 1.8      # TODO Justify this value
        self.a_max = 1.0    # TODO Justify this value
        self.v_max = 2.5    # TODO Justify this value
        self.r_nav = np.pi * 0.25 / self.N_j
        
        self.front_vehicle_area = None
        self.p_temp = None
        
    def get_distance_to_goal(self):
        return np.linalg.norm(self.goal - self.position)
        
        
    def step(self, dt, environment):
        vehicle = environment.vehicle
        
        #################################
        #### Compute repulsive force ####
        #################################
        if self.is_aware:
            lat_direction = environment.road.get_lateral_direction()
        
            # Compute lateral distance between pedestrian and vehicle
            d_lat = np.dot(lat_direction, self.position - vehicle.get_position())
            
            # Compute lateral force coefficient
            m_lat = self.M_veh * np.exp(-self.beta_veh * np.abs(d_lat))
            
            # Compute longitudinal coefficient
            p_ego = vehicle.transform.toLocal(self.position)
            p_x_ego = p_ego[0]
            L_r = vehicle.rear_distance
            L_f = vehicle.front_distance
            
            L_f += self.tau_x * vehicle.get_forward_velocity()
            
            if (p_x_ego) > -L_r and p_x_ego < L_f:
                m_lon = 1
            elif (p_x_ego > L_f) and p_x_ego < L_f + self.d_x:
                m_lon = 1 - (p_x_ego - L_f) / self.d_x
            else:
                m_lon = 0
            
            # Direction of the repulsive force
            n_rep = np.sign(d_lat) * lat_direction
            
            # Compute repulsive force
            F_rep = m_lon * m_lat * n_rep
        else:
            F_rep = 0
        
        ##################################
        #### Compute navigation force ########################################
        ##################################
        self.update_p_temp(environment)
        v_tar = self.desired_speed * (self.p_temp - self.position) / \
               np.sqrt(self.sigma**2 + np.linalg.norm(self.p_temp - self.position))
               
        F_nav = self.K_nav * (v_tar - self.velocity)
        
        #############################
        #### Compute total force #############################################
        #############################
        a = (F_nav + F_rep) / self.m
        
        if (np.linalg.norm(a) > self.a_max):
            a = self.a_max * a / np.linalg.norm(a)
            
        v_new = self.velocity + a*dt
        
        if (np.linalg.norm(v_new) > self.v_max):
            a = (self.v_max * v_new / np.linalg.norm(v_new) - self.velocity) / dt
            
        F_total = self.m * a
        
        ################################
        # Update kinematics parameters
        ################################
        if self.get_distance_to_goal() < 0.1:
            self.velocity = np.zeros((2,))
        else:
            self.velocity += F_total * dt / self.m
            
        self.position += self.velocity * dt
        self.shape = Circle(self.position, self.radius)
        return
    
    
    
    def update_p_temp(self, environment):
        if self.is_aware:
            p_des = self.goal - self.position
            phi_des = np.arctan2(p_des[1], p_des[0])
            
            veh = environment.vehicle
            v = veh.get_forward_velocity()
            self.front_vehicle_area = Rectangle(
                    center=veh.position + np.array([veh.orientation.cos(), 
                                                    veh.orientation.sin()])*self.tau_x*v/2.0, 
                    size=veh.vehicle_size + np.array([self.tau_x*v,0]),
                    orientation=veh.orientation)
    
            d   = np.zeros(shape=(self.N_j,), dtype=np.float32)
            phi = np.zeros(shape=(self.N_j,), dtype=np.float32)
            
            # 0 = None, 1 = Front, 2 = Other
            c = np.zeros(shape=(self.N_j,), dtype=np.uint8)
            for j in range(self.N_j):
                phi[j] = phi_des + (j - self.N_j//2) * self.r_nav
                start = self.position 
                end = self.position + self.d_nav * np.array([np.cos(phi[j]),
                                                                         np.sin(phi[j])])
                s = Segment(start=start, end=end)
                intersect_vehicle_front, points = s.intersect(self.front_vehicle_area)
                
                if not intersect_vehicle_front:
                    d[j] = self.d_nav
                    c[j] = 0
                else:
                    d[j] = np.min(np.linalg.norm(points - self.position)) - self.radius
                    c[j] = 1
            
          
            S_p = np.arange(self.N_j)[d == self.d_nav]   # Set of passable directions
            if S_p.size != 0:
                # Search for index j such that phi[j] is closer to phi_des
                # and d[j] == d_nav
                phi_star = phi[np.argmin(np.abs(phi[d == self.d_nav] - phi_des))]
                
                d = np.min((self.get_distance_to_goal(), self.d_nav))
                self.p_temp = np.array(self.position + d*np.array([np.cos(phi_star),
                                                                       np.sin(phi_star)]))
            else:
                # Get ego pedestrian velocity angle
                phi_ego = self.get_direction()
                phi_0 = Angle(phi[0], units='rad')
                phi_N = Angle(phi[-1], units='rad')
                
                if np.abs((phi_ego - phi_0).to180()) < np.abs((phi_ego - phi_N).to180()):
                    j = 0
                else:
                    j = -1
                self.p_temp = np.array(self.position + d[j]*np.array([np.cos(phi[j]), 
                                                                 np.sin(phi[j])]))
        else:
            self.p_temp = self.goal
            
    
    
    def render(self, 
               image, 
               position_pixel, 
               window_size, 
               real_world_size, 
               color=(255, 125, 0)):
        super().render(image, position_pixel, window_size, real_world_size)
        
        # Draw perceived front vehicle area
        # if self.front_vehicle_area is not None: 
        #     corners = [m2p(c[0], c[1], window_size, real_world_size) for c in self.front_vehicle_area.corners]
        #     pygame.draw.polygon(image, (255, 0, 125, 100), corners, width=3)
            

        # if self.p_temp is not None:
        #     # Draw Pedestrian
        #     pygame.draw.circle(image, (255, 0, 0), 
        #                         center=m2p(*self.p_temp, 
        #                                    window_size=window_size, 
        #                                    real_world_size=real_world_size),
        #                         radius=window_size[0]/real_world_size[0]*0.3)        
            
        
        
if __name__ == "__main__":
    p = PedestrianSGSFM()
    print(p)