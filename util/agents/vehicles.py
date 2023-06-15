#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:52:36 2021

@author: luca
"""
import numpy as np
from util.geometry import Transform, Angle, Rotation, Rectangle, getNormalVector

class Vehicle:
    """
    For now, without Mass and Intertia.
    """
    def __init__(self, 
                 position=np.array([0,0], dtype=np.float32), 
                 velocity=np.array([0,0], dtype=np.float32), 
                 goal=np.array([0,0], dtype=np.float32),
                 size=(4.0, 2.0),
                 orientation=Angle(), 
                 **kwargs):
        self.initial_position = position
        self.initial_velocity = velocity
        self.goal = np.array(goal, dtype=np.float32)
        
        self.goal_reached = False
        
        self.position = np.copy(self.initial_position)  # Position in World Frame
        self.velocity = np.copy(self.initial_velocity)  # Velocity in Local Frame, x axis points forward
        self.orientation = orientation # Angle of x-axis w.r.t world frame
        
        self.transform = Transform(self.position, orientation)
        self.rotation = Rotation(self.orientation)
        
        self._yaw_rate = 0.0
        self._yaw_rate_old = 0.0
        self.yaw_acceleration = 0.0
        
        self._acceleration = np.array([0,0], dtype=np.float32) # Acceleration in Vehicle Frame
        
        self.vehicle_size = size
        self.rear_distance = self.vehicle_size[0] / 2.0
        self.front_distance = self.vehicle_size[0] / 2.0
        self.shape = Rectangle(self.position, size=self.vehicle_size, orientation=self.orientation)
        
        
    def apply_control(self, yaw_rate, acc_x):
        # Steering and acceleration are normalized
        self._yaw_rate_old = self._yaw_rate
        self._yaw_rate = yaw_rate
        self._acceleration = 0.5*9.8*np.array([acc_x, 0.0], dtype=np.float32)
        
        
    def reset(self, initial_position=None, initial_velocity=None):
        if initial_position is not None:
            self.initial_position[:] = initial_position
            
        if initial_velocity is not None:
            self.initial_velocity[:] = initial_velocity
        
        self.position[:] = self.initial_position
        self.velocity[:] = self.initial_velocity
        self.orientation = Angle(0.0, 'rad')
        self.goal_reached = False
        
        
    def step(self, dt):
        self.position += self.rotation*(self.velocity*dt)    # Rotate velocity to world frame
        self.orientation += Angle(self._yaw_rate*dt, units='rad')
        self.velocity += self._acceleration * dt
        self.yaw_acceleration = (self._yaw_rate - self._yaw_rate_old) / dt
        
        self.rotation = Rotation(self.orientation)
        self.transform = Transform(self.position, self.orientation)
        
        if self.position[0] >= self.goal[0]:
            self.goal_reached = True
        
        self.shape = Rectangle(self.position, size=self.vehicle_size, orientation=self.orientation)
        
    def relative_speed(self, velocity):
        """
        Computes the speed relative to the vehicle of another velocity expressed
        in world coordinates
        """
        return Rotation(-self.orientation)*(velocity - self.velocity)
    
    def get_position(self):
        return self.position
        
    def get_velocity(self):
        return self.velocity
    
    def get_acceleration(self):
        return self._acceleration
    
    def get_forward_velocity(self):
        return self.velocity[0]
        
    def __str__(self):
        return f'Position: {self.position}, Orientation: {self.orientation}'
    

    
    
    
class Waypoint:
    def __init__(self, transform=Transform()):
        self.transform = transform
    
    def get_forward_vector(self):
        return self.transform.rotation._R[:,0]
    
    
    
class Road:
    """
    TO BE IMPLEMENTED
    """
    def __init__(self, lane_center, lane_width=8.0, orientation : Angle = Angle()):
        self._nodes = []
        self.lane_width = lane_width
        self.lane_center = lane_center
        self.y_min = self.lane_center - self.lane_width / 2.0
        self.y_max = self.lane_center + self.lane_width / 2.0
        
        self.vehicle_lane_center = self.lane_center - self.lane_width / 4.0
        
        self.orientation = orientation
        self.pavement_width = 2.0
        
        self.forward_direction = np.array([1, 0])
        self.lateral_direction = getNormalVector(self.forward_direction)
        
    def get_forward_direction(self):
        return self.forward_direction
    
    def get_lateral_direction(self):
        return self.lateral_direction
        
if __name__ == "__main__":
    print("Running as main")
        
  
    
        


