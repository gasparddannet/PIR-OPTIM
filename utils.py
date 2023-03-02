import math
import numpy as np
from numba import jit

import env

class Node:
    def __init__(self, n):
        self.long = n[0]
        self.lat = n[1]
        self.parent = None
    
    def __repr__(self):
        return "("+str(self.long)+", "+str(self.lat)+")"


class Utils:
    def __init__(self):
        self.env = env.Env()

        self.delta = 0     #0.5
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obs_ellipse = self.env.obs_ellipse


    def is_inside_obs(self, node):
        delta = self.delta

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.long - x, node.lat - y) <= r + delta:
                return True

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= node.long - x <= w + delta \
                    and 0 <= node.lat - y <= h + delta:
                return True

        for (x, y, w, h) in self.obs_ellipse:
            if math.pow((node.long - x), 2) / math.pow(w, 2) + math.pow((node.lat - y), 2) / math.pow(h, 2) <= 1:
                return True
    
        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.long - x <= w  \
                    and 0 <= node.lat - y <= h:
                return True
        return False


    def is_intersect_segment(self, start_long, start_lat, end_long, end_lat, c:list, d:list):
        """Trouve le point d'interection entre deux segments"""
        cx, cy = c[0], c[1]
        dx, dy = d[0], d[1]
        
        det  = (end_long - start_long) * (cy - dy) - (cx - dx)*(end_lat - start_lat)
        if det == 0:
            return False

        t1 = ((cx - start_long)*(cy - dy) - (cx - dx)*(cy - start_lat))/det
        t2 = ((end_long - start_long)*(cy - start_lat) - (cx - start_long)*(end_lat - start_lat))/det

        if t1>1 or t1<0 or t2>1 or t2<0:
            return False
        
        elif t1==0:
            return ([start_long, start_lat], t1)
        elif t1==1:
            return ([end_long, end_lat], t1)
        else:
            x = start_long + t1*(end_long - start_long)
            y = start_lat + t1*(end_lat - start_lat)
            return ([x,y], t1)



    def intersect(self, start_long, start_lat, end_long, end_lat, indice_x, indice_y, coef_avant_inter, coef_apres_inter, step, long_range, lat_range):
        """Calcul le coût en prenant en compte l'intersection"""
        l = []
        result1 = self.is_intersect_segment(start_long, start_lat, end_long, end_lat, [indice_x*step+long_range[0], indice_y*step+lat_range[0]], [indice_x*step+long_range[0], indice_y*step+step+lat_range[0]])
        if result1 != False:
            l.append(result1)
        
        result2 = self.is_intersect_segment(start_long, start_lat, end_long, end_lat, [indice_x*step+long_range[0], indice_y*step+step+lat_range[0]], [indice_x*step+step+long_range[0], indice_y*step+step+lat_range[0]])
        if result2 != False:
            l.append(result2)
        
        result3 = self.is_intersect_segment(start_long, start_lat, end_long, end_lat, [indice_x*step+step+long_range[0], indice_y*step+step+lat_range[0]], [indice_x*step+step+long_range[0], indice_y*step+lat_range[0]])
        if result3 != False:
            l.append(result3)
        
        result4 = self.is_intersect_segment(start_long, start_lat, end_long, end_lat, [indice_x*step+step+long_range[0], indice_y*step+lat_range[0]], [indice_x*step+long_range[0], indice_y*step+lat_range[0]])
        if result4 != False:
            l.append(result4)
        
        if l==[]:
            return False
        else:
            (inter_pt, _) = min(l, key=cmp_key)
            return coef_avant_inter * calc_dist(start_long, start_lat, inter_pt[0], inter_pt[1]) + coef_apres_inter * calc_dist(inter_pt[0], inter_pt[1], end_long, end_lat)         




    def sweep_y(self, start_long, start_lat, end_long, end_lat, pas, cond, coef_avant_inter, coef_apres_inter, step, long_range, lat_range, grid):
        """Trouve le point d'intersection avec un obstacle en balayant les y (latitude)"""
        nombre_points = round(abs((end_lat - start_lat)) / (step/pas))
        for y in np.linspace(start_lat, end_lat, max(2, nombre_points)):
            x = calcul_x(start_long, end_long, start_lat, end_lat, y)
            indice_y = indice(y, lat_range[0], step)
            indice_x = indice(x, long_range[0], step)
            if grid[indice_x][indice_y] == cond:
                return self.intersect(start_long, start_lat, end_long, end_lat, indice_x, indice_y, coef_avant_inter, coef_apres_inter, step, long_range, lat_range)




    def sweep_x(self, start_long, start_lat, end_long, end_lat, pas, cond, coef_avant_inter, coef_apres_inter, step, long_range, lat_range, grid):
        """Trouve le point d'intersection avec un obstacle en balayant les x (longitude)"""
        nombre_points = round(abs((end_long - start_long)) / (step/pas))
        for x in np.linspace(start_long, end_long, max(2, nombre_points)):
            y = calcul_y(start_long, end_long, start_lat, end_lat, x)
            indice_y = indice(y, lat_range[0], step)
            indice_x = indice(x, long_range[0], step)
            if grid[indice_x][indice_y] == cond:
                return self.intersect(start_long, start_lat, end_long, end_lat, indice_x, indice_y, coef_avant_inter, coef_apres_inter, step, long_range, lat_range)


    def sweep_y_2(self, start_long, start_lat, end_long, end_lat, pas, cost_in_obstacles, step, long_range, lat_range, grid):
        """Trouve les deux points d'intersection avec un obstacle (si il y en a) en balayant les y (latitude)"""
        inter_1 = []
        indice_inter_1 = []
        indice_inter_2 = []
        nombre_points = round(abs((end_lat - start_lat)) / (step/pas))
        for y in np.linspace(start_lat, end_lat, max(2, nombre_points)):
            x = calcul_x(start_long, end_long, start_lat, end_lat, y)
            indice_y = indice(y, lat_range[0], step)
            indice_x = indice(x, long_range[0], step)
            if grid[indice_x][indice_y] == 1:
                inter_1 = [x, y]
                indice_inter_1 = [indice_x, indice_y]
                break
        if inter_1 == []:
            return None

        for y in np.linspace(inter_1[1], end_lat, max(2, nombre_points)):
            x = calcul_x(start_long, end_long, start_lat, end_lat, y)
            indice_y = indice(y, lat_range[0], step)
            indice_x = indice(x, long_range[0],step)
            if grid[indice_x][indice_y] == 0:
                indice_inter_2 = [indice_x, indice_y]
                break
        return self.intersect(start_long, start_lat, inter_1[0], inter_1[1], indice_inter_1[0], indice_inter_1[1], 1, cost_in_obstacles, step, long_range, lat_range) \
            + self.intersect(inter_1[0], inter_1[1], end_long, end_lat, indice_inter_2[0], indice_inter_2[1], cost_in_obstacles, 1, step, long_range, lat_range)


    def sweep_x_2(self, start_long, start_lat, end_long, end_lat, pas, cost_in_obstacles, step, long_range, lat_range, grid):
        """Trouve les deux points d'intersection avec un obstacle (si il y en a) en balayant les x (longitude)"""
        inter_1 = []
        indice_inter_1 = []
        nombre_points = round(abs((end_long - start_long)) / (step/pas))
        for x in np.linspace(start_long, end_long, max(2, nombre_points)):
            y = calcul_y(start_long, end_long, start_lat, end_lat, x)
            indice_y = indice(y, lat_range[0], step)
            indice_x = indice(x, long_range[0], step)
            if grid[indice_x][indice_y] == 1:
                inter_1 = [x, y]
                indice_inter_1 = [indice_x, indice_y]
                break
        if inter_1 == []:
            return None

        for x in np.linspace(inter_1[0], end_long, max(2, nombre_points)):
            y = calcul_y(start_long, end_long, start_lat, end_lat, x)
            indice_y = indice(y, lat_range[0], step)
            indice_x = indice(x, long_range[0], step)
            if grid[indice_x][indice_y] == 0:
                indice_inter_2 = [indice_x, indice_y]
                break
        return self.intersect(start_long, start_lat, inter_1[0], inter_1[1], indice_inter_1[0], indice_inter_1[1], 1, cost_in_obstacles, step, long_range, lat_range) \
            + self.intersect(inter_1[0], inter_1[1], end_long, end_lat, indice_inter_2[0], indice_inter_2[1], cost_in_obstacles, 1, step, long_range, lat_range)




@jit(nopython=True)
def cmp_key(e):
    return e[1]

@jit(nopython=True)
def calcul_x(xa, xb, ya, yb, y):
    m = (yb - ya)/(xb- xa)
    p = ya - m*xa
    x = (y - p)/m
    return x

@jit(nopython=True)
def calcul_y(xa, xb, ya, yb, x):
    m = (yb - ya)/(xb - xa)
    p = ya - m*xa
    y = m*x + p
    return y

@jit(nopython=True)
def calc_dist(x_start_long, x_start_lat, x_end_long, x_end_lat):
    #############################
    #### Euclidean distance #####

    # return math.hypot(x_start_long - x_end_long, x_start_lat - x_end_lat)


    #############################
    ### Great-circle distance ###

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [x_start_long, x_start_lat, x_end_long, x_end_lat])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + (math.cos(lat1) * math.cos(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371.009        # radius of earth in kilometers
    return c * r 


@jit(nopython=True)
def indice(x, x0, step):
    a = x - x0
    b = a // step
    return int(b)
