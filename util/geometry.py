#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:28:28 2021

@author: luca
"""
import numpy as np
import matplotlib.pyplot as plt



def m2p(x, y, window_size, real_world_size):
    width, _ = window_size
    width_m, height_m = real_world_size
    c = width / width_m
    return (int(x*c), int((height_m - y)*c))


class Angle:
    def __init__(self, value=0.0, units='rad'):
        """
        Parameters
        ----------
        value : TYPE, optional
            DESCRIPTION. The default is 0.0.
        units : string, 'deg' or 'rad'
            DESCRIPTION. The default is 'rad'.

        Returns
        -------
        None.

        """
        value = float(value)
        self.units = units
        if units == 'deg':
            self._value = value * np.pi / 180.0
            self._value = self._value % (2*np.pi)
        elif units == 'rad':
            self._value = value % (2*np.pi)
        else:
            self._value = 0.0

    def to360(self):
        return self._value * 180.0 / np.pi

    def to2Pi(self):
        return self._value

    def toPi(self):
        return (self._value + np.pi) % (2*np.pi) - np.pi

    def to180(self):
        return self.toPi()*180/np.pi

    def __str__(self):
        return f"[rad: {self.to2Pi()}, deg: {self.to360()}]"

    def __sub__(self, other):
        return Angle(self._value - other._value, 'rad')

    def __add__(self, other):
        return Angle(self._value + other._value, 'rad')

    def __neg__(self):
        return Angle(-self._value, 'rad')

    def cos(self):
        return np.cos(self._value)

    def sin(self):
        return np.sin(self._value)

class Rotation:
    def __init__(self, angle : Angle = Angle()):
        if isinstance(angle, Angle):
            self._angle = angle
            c = angle.cos()
            s = angle.sin()
            self.R = np.array([[c, -s], [s, c]], dtype=np.float32)
        else:
            raise Exception("Parameter 'angle' must be an object of Angle class")

    def set_angle(self, angle : Angle = Angle()):
        self._angle = angle
        c = angle.cos()
        s = angle.sin()
        self.R[0,0] = c
        self.R[0,1] = -s
        self.R[1,0] = s
        self.R[1,1] = c

    def __mul__(self, v):
        return self.R.dot(v.T).T
    
    def __str__(self):
        return f"Angle: {self._angle}, array: {self._R}"

class Transform:
    def __init__(self,
                 pos : np.ndarray = np.array([0,0], dtype=np.float32),
                 yaw : Angle = Angle()):
        """
        Parameters
        ----------
        x : float
            Position along x axis of the frame with respect to world frame.
        y : float
            Position along y axis of the frame with respect to world frame.
        yaw : Angle
            Orientation of the frame. Right hand rule, with respect to world frame

        Returns
        -------
        None.
        """
        self.position = pos
        self.yaw = yaw
        self.rotation = Rotation(self.yaw)

    def toWorld(self, local_vec : np.ndarray):
        """
        Parameters
        ----------
        local_vec : Vector2D
            Vector in local coordinates.

        Returns
        -------
        Vector2D
            Vector in world reference frame.

        """
        return self.position + self.rotation*local_vec

    def toLocal(self, world_vec : np.ndarray):
        return Rotation(-self.yaw)*(world_vec - self.position)

    def __str__(self):
        return f"Position: {self.position}, Orientation: {self.yaw}"
    

class Segment:
    def __init__(self, start, end):
        self.start = np.copy(start)
        self.end = np.copy(end)
        
    def isparallel(self, other):
        ans = False
        v1 = self.end - self.start
        v2 = other.end - other.start
        
        # Condition is v1[0]/v2[0] == v1[1]/v2[1] but is rephrased as:
        return np.isclose(v1[0]*v2[1], v1[1]*v2[0])
    
        
    def intersect(self, other):
        # XXX Fix this function
        if isinstance(other, Segment):
            if  np.all(np.isclose(other.start, self.start)) or  \
                np.all(np.isclose(other.start, self.end)):
                
                return (True, other.start)
            
            elif np.all(np.isclose(other.end, self.start))   or  \
                np.all(np.isclose(other.end, self.end)):
                # Segments share at least one end point
                return (True , other.end)
            
            else:
                v1 = self.end - self.start
                v2 = other.end - other.start
                M = np.array([[-v2[1], v2[0]],
                              [-v1[1], v1[0]]])
                det = v2[0]*v1[1] - v1[0]*v2[1]
                
                if np.isclose(det, 0):
                    # det is very small, so segments are parallel, we can have
                    # either no intersection or an infinite number of intersections
                    # XXX Finish here
                    if self.isparallel(other):
                        return (False, None)
                    else:
                        return (False, None)
                else:
                    # Find solution of the system s1 + t*v1 = s2 + u*v2, in the 
                    # unknowns u, t
                    t, u = np.matmul(M, other.start - self.start) / det
                
                if (t >= 0 and t <= 1) and (u >= 0 and u <= 1):
                    return (True, self.start + t*v1)
                else:
                    return (False, None)
        elif isinstance(other, Rectangle):
            corners = other.corners
            edges = [Segment(corners[i], corners[(i+1)%4]) for i in range(4)]
            intersects = False
            points = []
            
            for edge in edges:
                intersect_edge, point = self.intersect(edge)
                if intersect_edge:
                    points.append(point)
                    intersects = True
                                                
            return (intersects, points)
        
            
        elif isinstance(other, Circle):
            raise Exception("Please implement Circle- Segment")
        else:
            raise Exception("Please implement this mehtod")
        
        
                
    def render(self, ax, **kwargs):
        xdata = [self.start[0], self.end[0]]
        ydata = [self.start[1], self.end[1]]
        ax.plot(xdata, ydata, **kwargs)
                
            

    
    
class Circle:
    def __init__(self, center, radius):
        self.center = np.copy(np.array(center))
        self.radius = radius
        
    def render(self, **kwargs):
        return plt.Circle(self.center, self.radius, **kwargs)
    
    def __str__(self):
        return f"Center: {self.center}, Radius: {self.radius}"
    
    def intersect(self, other) -> bool:
        if isinstance(other, Circle):
            return (np.linalg.norm(other.center - self.center) < other.radius + self.radius)
        elif isinstance(other, Rectangle):
            return other.intersect(self)
        else:
            raise Exception(f"Called intersection function between {type(self)} and {type(other)}")    




class Rectangle:
    def __init__(self, 
                 center, 
                 size, 
                 orientation : Angle = Angle()):
        self.center = center
        self.width, self.height = size
        self.transform = Transform(pos=center, yaw=orientation)
        
        self.local_corners = np.array([[self.width/2, self.height/2],
                                [-self.width/2, self.height/2],
                                [-self.width/2, -self.height/2],
                                [self.width/2, -self.height/2]])
        # self.corners = np.array([self.transform.toWorld(np.array([self.width/2, self.height/2])),
        #                         self.transform.toWorld(np.array([-self.width/2, self.height/2])),
        #                         self.transform.toWorld(np.array([-self.width/2, -self.height/2])),
        #                         self.transform.toWorld(np.array([self.width/2, -self.height/2]))])
        self.corners = self.transform.toWorld(self.local_corners)
        
        
    def intersect(self, other) -> bool:
        if isinstance(other, Rectangle):
            raise Exception("Please implement Rectangle-Rectangle intersection")
        elif isinstance(other, Circle):
            center = self.transform.toLocal(other.center)
            x_c = center[0]
            y_c = center[1]
            
            if (    (x_c > - self.width/2.0) 
                and (x_c < self.width/2.0) 
                and (y_c > - self.height/2.0)
                and (y_c < self.height/2.0)):
                return True
            else:
                # Check if any of the corners is inside the circle
                distances = np.sqrt(np.sum((self.local_corners - center)**2, axis=1))
                if (np.sum(distances <= other.radius) > 0):
                    # This means that one of the corners is inside the circle
                    return True
                else:
                    retVal = False
                    
                    d1 = np.abs(x_c - self.width/2.0) - other.radius
                    retVal |= ((d1 < 0) and (y_c > - self.height/2.0) and (y_c < self.height/2.0))
                    d2 = np.abs(y_c - self.height/2.0) - other.radius
                    retVal |= ((d2 < 0) and (x_c > - self.width/2.0) and (x_c < self.width/2.0))
                    d3 = np.abs(x_c + self.width/2.0) - other.radius
                    retVal |= ((d3 < 0) and (y_c > - self.height/2.0) and (y_c < self.height/2.0))
                    d4 = np.abs(y_c + self.height/2.0) - other.radius
                    retVal |= ((d4 < 0) and (x_c > - self.width/2.0) and (x_c < self.width/2.0))
                    
                    return retVal
        else:
            raise Exception("Intersection function not yet implemented")
            
                
    def render(self):
        return plt.Rectangle(self.transform.toWorld(np.array([-self.width/2.0, -self.height/2.0])),
                             self.width,
                             self.height, 
                             angle=self.transform.yaw.to360())
                        
                    
def getNormalVector(vector):
    """
    

    Parameters
    ----------
    vector : 2D Vector
        DESCRIPTION.

    Returns
    -------
    Vector normal to the input vector.

    """  
    return np.array([-vector[1], vector[0]])       
        
        
        


if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    r = Rectangle((1, -1), size=(4, 2), orientation=Angle(120, 'deg'))
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 1)
    fig.set_figheight(30)
    fig.set_figwidth(15)
    
    
    a = 10
    xticks = np.arange(-a, a)
    
    for ax in axes:
        ax.grid(b=True, which='major', color='k', linestyle='--')
        ax.axis('equal')
        ax.axis([-a, a, -a, a])
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
        
        # Draw the rectangle
        ax.add_patch(r.render())

    
    for _ in range(10):
        c = Circle(np.random.uniform(low=-4, high=4, size=(2,)), 
                    np.random.uniform(low=0.5, high=2.0))
        if (r.intersect(c)):
            axes[0].add_patch(c.render(fill=False))
        else:
            axes[1].add_patch(c.render(fill=False))
    
 
    # Create figure and axes
    fig2, axes2 = plt.subplots(2, 1)
    fig2.set_figheight(30)
    fig2.set_figwidth(15)
    
    a = 10
    xticks = np.arange(-a, a)
    
    s0 = Segment((8, 3), (-6, 1))
    for ax in axes2:
        ax.grid(b=True, which='major', color='k', linestyle='--')
        ax.axis('equal')
        ax.axis([-a, a, -a, a])
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
        
        # Draw the rectangle
        s0.render(ax, linewidth=5.0)
        
    for _ in range(100):
        s = Segment(np.random.uniform(low=-4, high=4, size=(2,)), 
                    np.random.uniform(low=-4, high=4, size=(2,)))
        intersect, point = s.intersect(s0)
        if intersect:
            s.render(axes2[0])
        else:
            s.render(axes2[1])
    
    
    
