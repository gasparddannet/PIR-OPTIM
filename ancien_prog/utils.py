"""
utils for collision check
@author: huiming zhou
"""

import math
import numpy as np
import os
import sys

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                 "/../../Sampling_based_Planning/")

# from Sampling_based_Planning.rrt_2D import env
# from Sampling_based_Planning.rrt_2D.rrt import Node

# from rrt import Node
import env

class Node:
    def __init__(self, n):
        self.long = n[0]
        self.lat = n[1]
        self.parent = None
    
    def __repr__(self):
        return "("+str(self.long)+", "+str(self.lat)+")"

    def equal(self, node1):
        if self.long == node1.long and self.lat == node1.lat:
            return True
        return False


class Utils:
    def __init__(self):
        self.env = env.Env()

        self.delta = 0     #0.5
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obs_ellipse = self.env.obs_ellipse

    def update_obs(self, obs_cir, obs_bound, obs_rec):
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound
        self.obs_rectangle = obs_rec

    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)

        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        if div == 0:            # v2 et v3 sont orthogonaux donc direction colinéaire a ce côté du rectangle
            return (False,0)

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return (True, shot)

        return (False,0)

    # def is_intersect_rec2(self, start:Node, end:Node, c:list, d:list):
    def is_intersect_rec2(self, start:Node, end:Node, c:list, d:list):

        cx, cy = c[0], c[1]
        dx, dy = d[0], d[1]
        
        det  = (end.long - start.long) * (cy - dy) - (cx - dx)*(end.lat - start.lat)
        if det == 0:
            return False

        t1 = ((cx - start.long)*(cy - dy) - (cx - dx)*(cy - start.lat))/det
        t2 = ((end.long - start.long)*(cy - start.lat) - (cx - start.long)*(end.lat - start.lat))/det

        if t1>1 or t1<0 or t2>1 or t2<0:
            return False
        
        elif t1==0:
            return (start, t1)
        elif t1==1:
            return (end, t1)
        # elif t2==0:
        #     return Node(c)
        # elif t2==1:
        #     return Node(d)
        else:
            x = start.long + t1*(end.long - start.long)
            y = start.lat + t1*(end.lat - start.lat)
            return (Node([x,y]), t1)

    def is_intersect_rec3(self, start, end, v1, v2, v3, v4):
        # l = [v1, v2, v3, v4]
        # i = 0

        cpt=[]
        # while len(cpt) < 2:
            # result = self.is_intersect_rec2(start, end, o, direc, l[i], l[i+1])

        result1 = self.is_intersect_rec2(start, end, v1, v2)
        if result1 != False:
            cpt.append(result1)


        result2 = self.is_intersect_rec2(start, end, v2, v3)
        if result2 != False:
            cpt.append(result2)

        result3 = self.is_intersect_rec2(start, end, v3, v4)
        if result3 != False:
            cpt.append(result3)

        
        result4 = self.is_intersect_rec2(start, end, v4, v1)
        if result4 != False:
            cpt.append(result4)

        if cpt==[]:
            return False

        elif len(cpt) == 1:
            return [cpt[0][0]]

        elif len(cpt) == 2:
            return [cpt[0], cpt[1]]

        else:
            if not cpt[0][0].equal(cpt[1][0]):
                return [cpt[0], cpt[1]]
            else:
                return [cpt[0], cpt[2]]


    def is_intersect_circle(self, start, end, a, r, o, d):  # a = (x,y) du centre du cercle
        d2 = np.dot(d, d)
        delta = self.delta

        if d2 == 0:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r + delta:
                return (True,[shot])

        return False


    def is_intersect_circle2(self, start, end, center, r, o, d):
        xc = center[0]
        yc = center[1]

        a = (end.long - start.long)**2 + (end.lat - start.lat)**2
        b = 2*( (end.long - start.long)*(start.long - xc) + (end.lat - start.lat)*(start.lat - yc) )
        c = (start.long - xc)**2 + (start.lat - yc)**2 - r**2
        delta = b**2 - 4*a*c
        if delta < 0:
            return False
        elif delta == 0:
            t = -b/(2*a)
            if 0 <= t <= 1:
                x = start.long + t*(end.long - start.long)
                y = start.lat + t*(end.lat - start.lat)
                # return [Node([x,y])]
            # else:
                return False            # tangent au cercle

        else:
            t1 = (-b - math.sqrt(delta))/(2*a)
            t2 = (-b + math.sqrt(delta))/(2*a)
            if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                x1 = start.long + t1*(end.long - start.long)
                y1 = start.lat + t1*(end.lat - start.lat)
                x2 = start.long + t2*(end.long - start.long)
                y2 = start.lat + t2*(end.lat - start.lat)
                return [(Node([x1,y1]), t1), (Node([x2,y2]), t2)]
            
            elif 0 <= t1 <= 1:
                x1 = start.long + t1*(end.long - start.long)
                y1 = start.lat + t1*(end.lat - start.lat)
                return [Node([x1,y1])]
                # return False
            elif 0 <= t2 <= 1:
                x2 = start.long + t2*(end.long - start.long)
                y2 = start.lat + t2*(end.lat - start.lat)
                return [Node([x2,y2])]
                # return False
            else:
                return False



    def is_collision(self, start:Node, end:Node):
        # if self.is_inside_obs(start) or self.is_inside_obs(end):
        #     return (True,-1)

        o, d = self.get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        for (v1, v2, v3, v4) in obs_vertex:
            # result = self.is_intersect_rec2(start, end, o, d, v1, v2)
            # if result != False:
            #     return result

            # result = self.is_intersect_rec2(start, end, o, d, v2, v3)
            # if result != False:
            #     return result

            # result = self.is_intersect_rec2(start, end, o, d, v3, v4)
            # if result != False:
            #     return result
            
            # result = self.is_intersect_rec2(start, end, o, d, v4, v1)
            # if result != False:
            #     return result

            result = self.is_intersect_rec3(start, end, v1, v2, v3, v4)
            if result != False:
                return result


        for (x, y, r) in self.obs_circle:
            result = self.is_intersect_circle2(start, end, [x, y], r+self.delta, o, d)
            if result != False:
                return result

        return False

    def is_inside_obs(self, node):
        delta = self.delta

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.long - x, node.lat - y) <= r + delta:
                return True

        for (x, y, w, h) in self.obs_rectangle:
            # if 0 <= node.long - (x - delta) <= w + 2 * delta \
            #         and 0 <= node.lat - (y - delta) <= h + 2 * delta:
            if 0 <= node.long - x <= w + delta \
                    and 0 <= node.lat - y <= h + delta:
                return True

        # for (x, y, w, h) in self.obs_boundary:
        #     if 0 <= node.long - (x - delta) <= w + 2 * delta \
        #             and 0 <= node.lat - (y - delta) <= h + 2 * delta:
        #         return True
        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.long - x <= w  \
                    and 0 <= node.lat - y <= h:
                return True

        for (x, y, w, h) in self.obs_ellipse:
            if math.pow((node.long - x), 2) / math.pow(w, 2) + math.pow((node.lat - y), 2) / math.pow(h, 2) <= 1:
                return True

        return False

    @staticmethod
    def get_ray(start:Node, end:Node):
        orig = [start.long, start.lat]
        direc = [end.long - start.long, end.lat - start.lat]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.long - start.long, end.lat - start.lat)