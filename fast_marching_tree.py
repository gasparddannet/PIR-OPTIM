import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from scipy.stats import norm
from scipy.stats import uniform
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from numba import jit
from numba.experimental import jitclass
from numba import float32
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
from shapely.errors import ShapelyDeprecationWarning
from pyproj import Geod

import env, plotting, utils



class Node:
    def __init__(self, n, w_speed=1, w_vect=(1,1)):
        self.long = n[0]
        self.lat = n[1]
        self.parent = None
        self.cost = np.inf
        self.w_speed = w_speed
        self.w_vect = w_vect
    def __repr__(self):
        return "("+str(self.long)+", "+str(self.lat)+")"


class FMT:
    def __init__(self, x_start, x_goal, search_radius, cost_in_obstcles, sample_numbers, step):
        self.x_init = Node(x_start)
        self.x_goal = Node(x_goal)
        self.search_radius = search_radius
        self.cost_in_obstcles = cost_in_obstcles

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.air_speed = 900 # km/h

        ###########################################################################
        ########### Pour afficher un simple graphique avec matplotlib   ###########

        # self.fig, self.ax1 = plt.subplots()
        # self.fig.set_size_inches(16.667, 10)
        ###########################################################################
        ##############  Pour afficher carte du monde avec cartopy   ###############

        self.fig = plt.figure(num = 'MAP', figsize=(16.667, 10)) #frameon=True)
        self.map = plt.axes(projection=ccrs.PlateCarree())
        # self.map.set_extent([-24, 35, 26, 65], ccrs.PlateCarree()) 
        # self.map.set_extent([-180, 180, -90, 90], ccrs.PlateCarree()) 
        self.map.set_extent([-100, 13, 33, 67], ccrs.PlateCarree()) 


        self.map.coastlines(resolution='50m')
        self.map.add_feature(cfeature.OCEAN.with_scale('50m'))
        self.map.add_feature(cfeature.LAKES.with_scale('50m'))
        self.map.add_feature(cfeature.LAND.with_scale('50m'))
        self.map.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='dotted', alpha=0.7)
        self.map.add_feature(cfeature.RIVERS.with_scale('10m'))

        # plt.title("titre")

        grid_lines = self.map.gridlines(draw_labels=True)
        grid_lines.xformatter = LONGITUDE_FORMATTER
        grid_lines.yformatter = LATITUDE_FORMATTER

        ###########################################################################
        ######## Pour afficher plusieurs graphiques sur la même figure    #########

        # self.fig = plt.figure(constrained_layout = True)
        # widths = [4, 1]
        # heights = [4, 1]
        # spec = self.fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
        # self.ax1 = self.fig.add_subplot(spec[0,0])
        
        # self.ax2 = self.fig.add_subplot(spec[0,1], sharey = self.ax1)
        # self.ax3 = self.fig.add_subplot(spec[1,0], sharex = self.ax1)
        # self.ax1.label_outer()
        
        # # self.ax2.label_outer()
        # # self.ax2.yaxis.set_ticks_position('none')
        # self.ax2.get_yaxis().set_visible(False)
        
        # self.ax3.label_outer()

        ###########################################################################



        self.delta = self.utils.delta
        self.long_range = self.env.long_range
        self.lat_range = self.env.lat_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obs_ellipse = self.env.obs_ellipse
        self.step = step

        self.grid = self.quartering(step)
        # save_quartering(self.grid, "quartering_map")
        # self.grid = load_quartering("quartering_testtt")

        self.V = set()
        self.V_unvisited = set()
        self.V_open = set()
        self.V_closed = set()
        self.sample_numbers = sample_numbers
        self.rn = self.search_radius * math.sqrt((math.log(self.sample_numbers) / self.sample_numbers))
        self.collision_set = set()

        print(f"Nombre d'échantillons : {self.sample_numbers}\nSearch radius : {self.search_radius}\nCoût dans obstacles : {self.cost_in_obstcles}\nStep : {self.step}\n")

    def Init(self):
        samples = self.SampleFree()
        # save_samples(samples, "samples_map.txt")
        # samples = self.load_samples("samples_testtt.txt")

        self.x_init.cost = 0.0
        self.V.add(self.x_init)
        self.V.update(samples)
        self.V_unvisited.update(samples)
        self.V_unvisited.add(self.x_goal)
        self.V_open.add(self.x_init)


    def Planning(self):

        start = time.time()

        self.Init()
        z = self.x_init
        n = self.sample_numbers

        eta = 1
        area = (self.long_range[1]-self.long_range[0]) * (self.lat_range[1] - self.lat_range[0])

        gamma = (1 + eta)*2*math.sqrt(1/2) * math.sqrt((area / math.pi))
        print("Gamma : ", gamma) 

        rn = self.search_radius * math.sqrt((math.log(n) / n))
        
        print("rn optimal : ", gamma*math.sqrt((math.log(n) / n)))
        
        print("rn = ", rn)
        Visited = []

        while z is not self.x_goal:
            V_open_new = set()
            X_near = self.Near(self.V_unvisited, z, rn)
            Visited.append(z)

            for x in X_near:

                Y_near = self.Near(self.V_open, x, rn)

                cost_list = {y: y.cost + self.Cost(y, x) for y in Y_near}
                y_min = min(cost_list, key=cost_list.get)

                x.parent = y_min
                V_open_new.add(x)
                self.V_unvisited.remove(x)
                x.cost = y_min.cost + self.Cost(y_min, x)
                
            self.V_open.update(V_open_new)
            self.V_open.remove(z)
            self.V_closed.add(z)

            if not self.V_open:
                print("open set empty!")
                break

            cost_open = {y: y.cost for y in self.V_open}
            z = min(cost_open, key=cost_open.get)

        # node_end = self.ChooseGoalPoint()
        path_x, path_y, path = self.ExtractPath()

        end = time.time()
        print("\nTemps d'execution : ",end-start)
        heure, minutes, sec, millisec = temps_sec(end-start)
        print(f"Temps d'exécution : {heure}h {minutes}min {sec}s {millisec}ms\n")
        # print("\nPath : ", path)

        total_cost_verif = self.calc_dist_total(path)
        print("Total cost (en calculant la distance sans aucunes pénalités): ", total_cost_verif)

        print("Total cost : ", self.x_goal.cost)
        self.animation(path_x, path_y, Visited[1: len(Visited)], path)
        # self.plot_nodes()



    def ChooseGoalPoint(self):
        Near = self.Near(self.V, self.x_goal, 2.0)
        cost = {y: y.cost + self.Cost(y, self.x_goal) for y in Near}
        return min(cost, key=cost.get)


    def ExtractPath(self):
        path_x, path_y = [], []
        path = []
        node = self.x_goal

        while node.parent:
            path_x.append(node.long)
            path_y.append(node.lat)
            path.append(node)
            node = node.parent

        path_x.append(self.x_init.long)
        path_y.append(self.x_init.lat)
        path.append(self.x_init)

        return path_x, path_y, path



    # def indice(self, x, x0):
    #     a = x - x0
    #     b = a // self.step
    #     return int(b)



    # def intersect(self, start, end, indice_x, indice_y, coef_avant_inter, coef_apres_inter, true_start, true_end):
    #     l = []
    #     result1 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.long_range[0], indice_y*self.step+self.lat_range[0]], [indice_x*self.step+self.long_range[0], indice_y*self.step+self.step+self.lat_range[0]])
    #     if result1 != False:
    #         l.append(result1)
      
    #     result2 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.long_range[0], indice_y*self.step+self.step+self.lat_range[0]], [indice_x*self.step+self.step+self.long_range[0], indice_y*self.step+self.step+self.lat_range[0]])
    #     if result2 != False:
    #         l.append(result2)
      
    #     result3 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+self.long_range[0], indice_y*self.step+self.step+self.lat_range[0]], [indice_x*self.step+self.step+self.long_range[0], indice_y*self.step+self.lat_range[0]])
    #     if result3 != False:
    #         l.append(result3)
      
    #     result4 = self.utils.is_intersect_rec2(start, end, [indice_x*self.step+self.step+self.long_range[0], indice_y*self.step+self.lat_range[0]], [indice_x*self.step+self.long_range[0], indice_y*self.step+self.lat_range[0]])
    #     if result4 != False:
    #         l.append(result4)
      
    #     if l==[]:
    #         return False
    #     else:
    #         (inter_pt, _) = min(l, key=cmp_key)
    #         self.collision_set.add((inter_pt, true_start, true_end))
    #         return coef_avant_inter * calc_dist(start.long, start.lat, inter_pt.long, inter_pt.lat) + coef_apres_inter * calc_dist(inter_pt.long, inter_pt.lat, end.long, end.lat)         


    # # @jit(nopython=True) 
    # def fonction_aux_y(self, start, end, pas, cond, coef_avant_inter, coef_apres_inter):
    #     start_x = start.long
    #     start_y = start.lat
    #     end_x = end.long
    #     end_y = end.lat

    #     nombre_points = round(abs((end_y - start_y)) / (self.step/pas))
    #     for y in np.linspace(start_y, end_y, max(2, nombre_points)):
    #         x = calcul_x(start_x, end_x, start_y, end_y, y)
    #         indice_y = self.indice(y, self.lat_range[0])
    #         indice_x = self.indice(x, self.long_range[0])
    #         if self.grid[indice_x][indice_y] == cond:

    #             return self.intersect(start, end, indice_x, indice_y, coef_avant_inter, coef_apres_inter, start, end)

    #     # print("oups 1")
    #     # print("start : ",start)
    #     # print("end : ", end)
    #     # return self.intersect(start, end, indice_end_x, indice_end_y, coef_avant_inter, coef_apres_inter)

    # # @jit(nopython=True) 
    # def fonction_aux_x(self, start, end, pas, cond, coef_avant_inter, coef_apres_inter):
    #     start_x = start.long
    #     start_y = start.lat
    #     end_x = end.long
    #     end_y = end.lat

    #     nombre_points = round(abs((end_x - start_x)) / (self.step/pas))
    #     for x in np.linspace(start_x, end_x, max(2, nombre_points)):
    #         y = calcul_y(start_x, end_x, start_y, end_y, x)

    #         indice_y = self.indice(y, self.lat_range[0])
    #         indice_x = self.indice(x, self.long_range[0])
    #         if self.grid[indice_x][indice_y] == cond:

    #             return self.intersect(start, end, indice_x, indice_y, coef_avant_inter, coef_apres_inter, start, end)

    #     # print("oups 2")
    #     # print("start : ",start)
    #     # print("end : ", end)
    #     # return self.intersect(start, end, indice_end_x, indice_end_y, coef_avant_inter, coef_apres_inter)
   
    # # @jit(nopython=True) 
    # def fonction_aux2_y(self, start, end, pas):
    #     start_x = start.long
    #     start_y = start.lat
    #     end_x = end.long
    #     end_y = end.lat
    #     inter_1 = []
    #     indice_inter_1 = []
    #     indice_inter_2 = []


    #     nombre_points = round(abs((end_y - start_y)) / (self.step/pas))
    #     for y in np.linspace(start_y, end_y, max(2, nombre_points)):
    #         x = calcul_x(start_x, end_x, start_y, end_y, y)

    #         indice_y = self.indice(y, self.lat_range[0])
    #         indice_x = self.indice(x, self.long_range[0])
    #         if self.grid[indice_x][indice_y] == 1:
    #             inter_1 = [x, y]
    #             indice_inter_1 = [indice_x, indice_y]
    #             break
    #     if inter_1 == []:
    #         return None

    #     for y in np.linspace(inter_1[1], end_y, max(2, nombre_points)):
    #         x = calcul_x(start_x, end_x, start_y, end_y, y)

    #         indice_y = self.indice(y, self.lat_range[0])
    #         indice_x = self.indice(x, self.long_range[0])
    #         if self.grid[indice_x][indice_y] == 0:
    #             inter_2 = [x, y]
    #             indice_inter_2 = [indice_x, indice_y]
    #             break

    #     return self.intersect(start, Node(inter_1), indice_inter_1[0], indice_inter_1[1], 1, self.cost_in_obstcles, start, end) \
    #         + self.intersect(Node(inter_1), end, indice_inter_2[0], indice_inter_2[1], self.cost_in_obstcles, 1, start, end)

    # # @jit(nopython=True) 
    # def fonction_aux2_x(self, start, end, pas):
    #     start_x = start.long
    #     start_y = start.lat
    #     end_x = end.long
    #     end_y = end.lat
    #     inter_1 = []
    #     indice_inter_1 = []


    #     nombre_points = round(abs((end_x - start_x)) / (self.step/pas))
    #     for x in np.linspace(start_x, end_x, max(2, nombre_points)):
    #         y = calcul_y(start_x, end_x, start_y, end_y, x)

    #         indice_y = self.indice(y, self.lat_range[0])
    #         indice_x = self.indice(x, self.long_range[0])
    #         if self.grid[indice_x][indice_y] == 1:
    #             inter_1 = [x, y]
    #             indice_inter_1 = [indice_x, indice_y]
    #             break
    #     if inter_1 == []:
    #         return None

    #     for x in np.linspace(inter_1[0], end_x, max(2, nombre_points)):
    #         y = calcul_y(start_x, end_x, start_y, end_y, x)

    #         indice_y = self.indice(y, self.lat_range[0])
    #         indice_x = self.indice(x, self.long_range[0])
    #         if self.grid[indice_x][indice_y] == 0:
    #             inter_2 = [x, y]
    #             indice_inter_2 = [indice_x, indice_y]
    #             break

    #     return self.intersect(start, Node(inter_1), indice_inter_1[0], indice_inter_1[1], 1, self.cost_in_obstcles, start, end) \
    #         + self.intersect(Node(inter_1), end, indice_inter_2[0], indice_inter_2[1], self.cost_in_obstcles, 1, start, end)

    # @jit(nopython=True) 
    def Cost(self, start:Node, end:Node) -> float:
        # start_x = start.long
        # start_y = start.lat
        # end_x = end.long
        # end_y = end.lat

        pas = 5

        vitesse_vent = start.w_speed
        vecteur_vent = start.w_vect
        vecteur_vitesse = (end.long-start.long, end.lat-start.lat)
        theta = math.acos(np.dot(vecteur_vent, vecteur_vitesse) / (np.linalg.norm(vecteur_vent) * np.linalg.norm(vecteur_vitesse)))
        vitesse_sol = math.pow(math.pow(self.air_speed, 2) - math.pow(vitesse_vent*math.sin(theta), 2), 1/2) + vitesse_vent * math.cos(theta)

        indice_start_long = indice(start.long, self.long_range[0], self.step)
        indice_start_lat = indice(start.lat, self.lat_range[0], self.step)
        indice_end_long = indice(end.long, self.long_range[0], self.step)
        indice_end_lat = indice(end.lat, self.lat_range[0], self.step)
        
        # vitesse_sol = math.pow(self.air_speed,2) - math.pow(vitesse_vent,2)

        if self.grid[indice_start_long][indice_start_lat] == 1 and self.grid[indice_end_long][indice_end_lat] == 1:
            return (self.cost_in_obstcles * calc_dist(start.long, start.lat, end.long, end.lat)) #/vitesse_sol
        

        elif self.grid[indice_start_long][indice_start_lat] == 0 and self.grid[indice_end_long][indice_end_lat] == 1:
            
            if abs(end.long - start.long) < abs(end.lat - start.lat):
                # result = self.fonction_aux_y(start, end, pas, 1, 1, self.cost_in_obstcles)
                # print("2", result)
                return fonction_aux_y(start.long, start.lat, end.long, end.lat, pas, 1, 1, self.cost_in_obstcles, self.step, self.long_range, self.lat_range, self.grid) #/ vitesse_sol
            else:
                # result = self.fonction_aux_x(start, end, pas, 1, 1, self.cost_in_obstcles)
                # print("3", result)

                return fonction_aux_x(start.long, start.lat, end.long, end.lat, pas, 1, 1, self.cost_in_obstcles, self.step, self.long_range, self.lat_range, self.grid) #/ vitesse_sol


        elif self.grid[indice_start_long][indice_start_lat] == 1 and self.grid[indice_end_long][indice_end_lat] == 0:        
            
            if abs(end.long - start.long) < abs(end.lat - start.lat):
                # result = self.fonction_aux_y(start, end, pas, 0, self.cost_in_obstcles, 1)
                # print("4", result)
                return fonction_aux_y(start.long, start.lat, end.long, end.lat, pas, 0, self.cost_in_obstcles, 1, self.step, self.long_range, self.lat_range, self.grid) #/ vitesse_sol
            else:
                # result = self.fonction_aux_x(start, end, pas, 0, self.cost_in_obstcles, 1)
                # print("5", result)
                return fonction_aux_x(start.long, start.lat, end.long, end.lat, pas, 0, self.cost_in_obstcles, 1, self.step, self.long_range, self.lat_range, self.grid) #/ vitesse_sol


        else:
            # cas 1 : ne rentre pas dans un obstalce
            # cas 2 : passe un petit peu dans un obstacle ??

            
            if abs(end.long - start.long) < abs(end.lat - start.lat):
                result = fonction_aux2_y(start.long, start.lat, end.long, end.lat, pas, self.cost_in_obstcles, self.step, self.long_range, self.lat_range, self.grid) 
            else:
                result = fonction_aux2_x(start.long, start.lat, end.long, end.lat, pas, self.cost_in_obstcles, self.step, self.long_range, self.lat_range, self.grid) 
                
            if result == None:
                return calc_dist(start.long, start.lat, end.long, end.lat) #/ vitesse_sol
            else:
                return result #/ vitesse_sol

            # return self.calc_dist(start, end)

    # # @staticmethod
    # # @jit(nopython=True) 
    # def calc_dist(self, x_start:Node, x_end:Node):
    #     # return math.hypot(x_start.long - x_end.long, x_start.lat - x_end.lat)

    #     # convert decimal degrees to radians
    #     lon1, lat1, lon2, lat2 = map(math.radians, [x_start.long, x_start.lat, x_end.long, x_end.lat])

    #     # Haversine formula
    #     dlon = lon2 - lon1
    #     dlat = lat2 - lat1
    #     a = math.sin(dlat/2)**2 + (math.cos(lat1) * math.cos(lat2)) * math.sin(dlon/2)**2
    #     c = 2 * math.asin(math.sqrt(a))
    #     r = 6391        # radius of earth in kilometers
    #     return c * r 

    def calc_dist_total(self, path):
        total_cost = 0
        length = len(path)
        for i in np.arange(0, length-1, 1):

            total_cost += calc_dist(path[i].long, path[i].lat, path[i+1].long, path[i+1].lat)
            # print(f"Cost de path[{i}] ({path[i]}) : {path[i].cost}")
        return total_cost


    def Near(self, nodelist, z, rn):
        return {nd for nd in nodelist
                if 0 < (nd.long - z.long) ** 2 + (nd.lat - z.lat) ** 2 <= rn ** 2}


    def proportion_obstacles(self):
        n = len(self.grid)
        m = len(self.grid[0])
        nombre_points_dans_obstacles = 0
        for i in np.arange(0, n, 1):
            for j in np.arange(0, m, 1):
                if self.grid[i][j] == 1:
                    nombre_points_dans_obstacles += 1
        proportion_obstacles = nombre_points_dans_obstacles / (n*m)
        # print("nombre points dans obstacles : ", nombre_points_dans_obstacles)
        # print("Proportion obstacles : ", proportion_obstacles) 
        return proportion_obstacles


    def SampleFree(self):

        start = time.time()

        n = self.sample_numbers
        delta = self.utils.delta
        Sample = set()

        porportion_obstacles = self.proportion_obstacles()
        # porportion_obstacles = 1

        nb_points_dans_obstacles = round(porportion_obstacles * self.sample_numbers)
        nb_points_dans_espace_libre = self.sample_numbers - nb_points_dans_obstacles

     
        cpt_dans_obstacles = 0
        cpt_dans_espace_libre = 0 

################################################################################
################################################################################



        distributions = []
        distributions_x = []
        distributions_y = []

        distributions_x_uniform = []
        distributions_y_uniform = []

        coefficients = []
        somme_coefficients = 0
        area = (self.long_range[1] - self.long_range[0]) * (self.lat_range[1] - self.lat_range[0])
        # print("area : ", area)
        ecart = 1
        for (x, y, r) in self.obs_circle:
            # mu_x = x 
            # sigma_x = r/4
            # mu_y = y
            # sigma_y = sigma_x
            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}}, 
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            #############################################
            xc = x
            yc = y
            radius = r
            distributions.append(("circle", (xc, yc, radius, ecart)))
            p = math.pi * radius**2
            coefficients.append(p)
            somme_coefficients += p

        for (x, y, w, h) in self.obs_ellipse:
            xc = x
            yc = y
            distributions.append(("ellipse", (xc, yc, w, h, ecart)))
            p = math.pi * w * h
            coefficients.append(p)
            somme_coefficients += p


        for (x, y, w, h) in self.obs_rectangle:
            # mu_x = x + w/2
            # sigma_x = w/4
            # mu_y = y + h/2
            # sigma_y = h/4
            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}},
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            ##########################
            ##########################
            area_rect = w * h
            # p = area_rect/4

            coef_sigma = 3

            # mu_x = x
            # sigma_x = ecart
            # mu_y = y+h/2
            # sigma_y = h/coef_sigma
            
            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}},
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            low_x = x - ecart
            high_x = x + ecart
            low_y = y - ecart
            high_y = y + h + ecart

            distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                  {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
            distributions_x_uniform.append((low_x, high_x))
            distributions_y_uniform.append((low_y, high_y))
            # p = 2*ecart * h
            p = 2*ecart*h * (area_rect/(4*ecart*(h+w)))  #*2     #*2 pour augmenter les probas car les rectangles dans ce cas là sont petits ?

            coefficients.append(p)
            somme_coefficients += p

            ##############
            # mu_x = x + w/2
            # sigma_x = w/coef_sigma
            # mu_y = y
            # sigma_y = ecart
            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}},
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            low_x = x - ecart
            high_x = x + w + ecart
            low_y = y - ecart
            high_y = y + ecart
            distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                  {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
            distributions_x_uniform.append((low_x, high_x))
            distributions_y_uniform.append((low_y, high_y))

            # p = 2*ecart * w
            p = 2 * ecart * w * (area_rect/(4*ecart*(h+w)))  #*2

            coefficients.append(p)
            somme_coefficients += p

            ##############
            # mu_x = x + w
            # sigma_x = ecart
            # mu_y = y + h/2
            # sigma_y = h/coef_sigma

            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}},
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            low_x = x + w - ecart
            high_x = x + w + ecart
            low_y = y - ecart
            high_y = y + h + ecart
            distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                  {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
            distributions_x_uniform.append((low_x, high_x))
            distributions_y_uniform.append((low_y, high_y))

            # p = 2*ecart * h
            p = 2*ecart*h * (area_rect/(4*ecart*(h+w)))  #*2
            coefficients.append(p)
            somme_coefficients += p

            ##############
            # mu_x = x + w/2
            # sigma_x = w/coef_sigma
            # mu_y = y + h
            # sigma_y = ecart
            # distributions.append(({"type": np.random.normal, "kwargs": {"loc": mu_x, "scale": sigma_x}},
            #                       {"type": np.random.normal, "kwargs": {"loc": mu_y, "scale": sigma_y}}))
            # distributions_x.append((mu_x, sigma_x))
            # distributions_y.append((mu_y, sigma_y))

            low_x = x - ecart
            high_x = x + w + ecart
            low_y = y + h - ecart
            high_y = y + h + ecart
            distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                  {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
            distributions_x_uniform.append((low_x, high_x))
            distributions_y_uniform.append((low_y, high_y))

            # p = 2*ecart * w
            p = 2*ecart*w * (area_rect/(4*ecart*(h+w)))  #*2

            coefficients.append(p)
            somme_coefficients += p


        distributions.append(({"type":np.random.uniform, "kwargs":{"low": self.long_range[0], "high": self.long_range[1]}},
                              {"type":np.random.uniform, "kwargs":{"low": self.lat_range[0], "high": self.lat_range[1]}}))
        p = (area - somme_coefficients)/4   
        coefficients.append(p)

        # print("coeeficent 1 :", coefficients)
        # somme_coefficients += p
        coefficients = np.array(coefficients)
        coefficients = coefficients / np.sum(coefficients)
        # print("coefficents : ", coefficients)
        # print("somme coefficents : ", np.sum(coefficients))



        ##########################################
        ## Pour afficher les dentsités de proba ##

        # print(distributions)
        num_distr = len(distributions)

        colors = ["gray", "red", "gold", "olivedrab", "chartreuse", "aqua", "royalblue", "blueviolet", "pink"]
        n = len(colors)
        x = np.linspace(self.long_range[0], self.long_range[1], 1000)
        density_x = 0

        ## for i, (mu_x, sigma_x) in enumerate(distributions_x):
        ##     density_x += norm.pdf(x, mu_x, sigma_x)
            
        ##     # if i==3:
        ##     self.ax3.plot(x, norm.pdf(x, mu_x, sigma_x)/num_distr, color=colors[i%n])
        
        # for i, (low_x, high_x) in enumerate(distributions_x_uniform):
        #     density_x += uniform.pdf(x, low_x, high_x)
        #     self.ax3.plot(x, uniform.pdf(x, low_x, high_x)/num_distr, color=colors[i%n])

        # density_x += uniform.pdf(x, self.long_range[0], self.long_range[1])
        # self.ax3.plot(x, uniform.pdf(x, self.long_range[0], self.long_range[1])/num_distr, linewidth = 1, color="orange")

        # density_x /= num_distr
        # self.ax3.plot(x, density_x, color="black", alpha=0.8)



        y = np.linspace(self.lat_range[0], self.lat_range[1], 1000)
        density_y = 0
        ## for i, (mu_y, sigma_y) in enumerate(distributions_y):
        ##     density_y += norm.pdf(y, mu_y, sigma_y)
            
        ##     # if i==3:
        ##     self.ax2.plot(norm.pdf(y, mu_y, sigma_y)/num_distr, y, color=colors[i%n])

        # for i, (low_y, high_y) in enumerate(distributions_y_uniform):
        #     density_y += uniform.pdf(y, low_y, high_y)
        #     self.ax2.plot(uniform.pdf(y, low_y, high_y)/num_distr, y, color=colors[i%n])

        # density_y += uniform.pdf(y, self.lat_range[0], self.lat_range[1])
        # self.ax2.plot(uniform.pdf(y, self.lat_range[0], self.lat_range[1])/num_distr, y, linewidth = 1, color="orange")

        # density_y /= num_distr
        # self.ax2.plot(density_y, y, color="black", alpha=0.8)

        ##################################################


        while cpt_dans_obstacles < nb_points_dans_obstacles or cpt_dans_espace_libre < nb_points_dans_espace_libre:

            # node = Node(get_sample(distributions, num_distr, coefficients), random.uniform(-self.air_speed, -50), (random.uniform(5,20), random.uniform(0.2,1)))
            # node = Node(get_sample(distributions, num_distr, coefficients), 100, (10, 0))
           
            node = Node(get_sample(distributions, num_distr, coefficients), 0.01, (random.uniform(5,20), random.uniform(0.2,1)))
            node = Node(get_sample(distributions, num_distr, coefficients), 0.01, (10, 0))


            # x,y = get_sample(distributions, num_distr, coefficients)
            # if y>48:
            #     node = Node((x, y), 100, (-10, 0))
            # else:
            #     node = Node((x, y), 100, (10, 0))

            # node = Node((random.uniform(self.long_range[0], self.long_range[1]),
            #              random.uniform(self.lat_range[0], self.lat_range[1])))

            if node.long>self.long_range[0] and node.long < self.long_range[1] and node.lat>self.lat_range[0] and node.lat<self.lat_range[1]:
                indice_node_x = indice(node.long, self.long_range[0], self.step)
                indice_node_y = indice(node.lat, self.lat_range[0], self.step)

                if self.grid[indice_node_x][indice_node_y] == 1 and cpt_dans_obstacles < nb_points_dans_obstacles:
                    Sample.add(node)
                    cpt_dans_obstacles += 1

                if self.grid[indice_node_x][indice_node_y] == 0 and cpt_dans_espace_libre < nb_points_dans_espace_libre:
                    Sample.add(node)
                    cpt_dans_espace_libre += 1
                

        print("Proportion obstacles : ", porportion_obstacles)
        print("Nb points dans obstacle : ", cpt_dans_obstacles)
        print("Nb points dans esapce libre : ", cpt_dans_espace_libre)
        
        end = time.time()
        print("Temps d'execution sample: ",end-start)
        return Sample

    def plot_nodes(self):
        self.plot_grid(f"Fast Marching Trees (FMT*) avec n = {self.sample_numbers}, un coût dans l'obstacle de {self.cost_in_obstcles} et step = {self.step}")


        for node in self.V:
            self.ax1.plot(node.long, node.lat, marker='.', color='green', markersize=5, alpha=0.7)
            # self.map.plot(node.long, node.lat, marker='o', color='magenta', markersize=1, alpha=0.8, transform=ccrs.PlateCarree())

            # self.ax1.add_patch(
            #     patches.Circle(
            #         (node.long, node.lat), self.rn,
            #         edgecolor='black',
            #         facecolor='gray',
            #         fill=False
            #     )
            # )

        # plt.savefig("/media/gaspard/OS_Install/Users/Gaspard/Desktop/ENAC/2A/Cours/Semestre 7/PIR OPTIM/Evitement de contrails en free flight avec des methodes de graphes aleatoires/article/images/sampling/n=8000.pdf", format='pdf')
        plt.show()
    
    def animation(self, path_x, path_y, visited, path):
        # self.plot_grid(f"Fast Marching Trees (FMT*) avec n : {self.sample_numbers}, coût dans obstacles : {self.cost_in_obstcles}, rayon de recherche : {self.rn:.2f} et step : {self.step}")
        # self.plot_grid(f"Fast Marching Trees (FMT*) avec n : {self.sample_numbers}, rayon de recherche : {self.rn:.2f}\nDistance orthodromique")

        # for node in self.V:
        #     self.ax1.plot(node.long, node.lat, marker='.', color='green', markersize=5, alpha=0.7)  #lightgrey
        #     # self.map.plot(node.long, node.lat, marker='.', color='green', markersize=5, alpha=0.7, transform=ccrs.PlateCarree())  #lightgrey

            

        # count = 0
        # for node in visited:
        #     count += 1
        #     # self.ax1.plot([node.long, node.parent.long], [node.lat, node.parent.lat], '-g')
        #     # self.map.plot([node.long, node.parent.long], [node.lat, node.parent.lat], '-g', transform=ccrs.Geodetic())
        #     self.map.plot([node.long, node.parent.long], [node.lat, node.parent.lat], '-g', transform=ccrs.PlateCarree())


        #     plt.gcf().canvas.mpl_connect(
        #         'key_press_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])         # key_release_event
        #     if count % (self.sample_numbers/5) == 0:
        #         plt.pause(0.01)

        # self.ax1.plot(path_x, path_y, linewidth=4, color='red')

        self.map.plot(path_x, path_y, linewidth=3, color='red', transform=ccrs.PlateCarree(), label="Trajectoire calculée")
        # self.map.plot([self.x_init.long, self.x_goal.long], [self.x_init.lat, self.x_goal.lat], transform = ccrs.Geodetic(), color = 'green', label="trajectoire orthodromique")
        
        ##################################################################################
        geod = Geod(ellps='WGS84')
        nb = 10000
        traj = geod.npts(self.x_init.long, self.x_init.lat, self.x_goal.long, self.x_goal.lat, nb)
        distance = 0    
        list_long = []
        list_lat = []
        for i in np.arange(0, nb, 1):
            list_long.append(traj[i][0])
            list_lat.append(traj[i][1])

        self.map.plot(list_long, list_lat, transform = ccrs.Geodetic(), color ='blue',linewidth=3, linestyle="--", label="Trajectoire orthodromique")   
        distance = geod.line_length(list_long, list_lat)
        distance/=1000
        print(f"Longueur trajectoire orthodromique : {distance:.3f} km")
        print(f"Longueur trajectoire calculée : {self.x_goal.cost:.3f} km")
        # temps = distance/self.air_speed
        #######
        self.map.text(self.x_init.long-4, self.x_init.lat-3, "Chicago", transform=ccrs.PlateCarree(),color='navy', alpha = 1, fontsize = 15)
        self.map.text(self.x_goal.long-2, self.x_goal.lat-3, "Paris", transform=ccrs.PlateCarree(),color='navy', alpha = 1, fontsize = 15)

        ###################################################################################################################
        # plt.legend(["Trajectoire orthodromique", "Trajectoire calculée"], loc="upper right")

        # plt.pause(0.01)

        #######################################################################
        ##### Pour afficher les intersections avec les obstacles ##############
        # for (node, start, end) in self.collision_set:
        #     for (i, node_i) in enumerate(path):
        #         if node_i == start and path[i-1] == end:
        #             self.ax1.plot(node.long, node.lat, marker="x", color="blue", markersize=6)
        #             # self.map.plot(node.long, node.lat, marker="x", color="blue", markersize=6, transform=ccrs.PlateCarree())

        # plt.pause(0.01)
        #######################################################################

        # self.plot_grid(f"Fast Marching Trees (FMT*) avec n : {self.sample_numbers}, rayon de recherche : {self.rn:.2f}\n\nDistance trajectoire calculée : {self.x_goal.cost:.3f} km\nDistance trajectoire orthodromique : {distance:.3f} km")
        
        ###########################################################################################
        # heure1, min1 = temps_heure_min(self.x_goal.cost)
        # heure2, min2 = temps_heure_min(temps)
        # print(f"Temps trajectoire calculée : {heure1}:{min1}")
        # print(f"Temps trajectoire orthodromique : {heure2}:{min2}")

        self.plot_grid(f"Fast Marching Trees (FMT*) avec n : {self.sample_numbers}, coût dans obstacles : {self.cost_in_obstcles}, rayon de recherche : {self.rn:.2f}\n\nTemps trajectoire calculée : {int(self.x_goal.cost):.0f}h{(self.x_goal.cost % 1)*60:.0f}\nTemps trajectoire orthodromique : ") #{heure2}h{min2}")

        plt.legend(fontsize = 13)

        # plt.savefig("/media/gaspard/OS_Install/Users/Gaspard/Desktop/ENAC/2A/Cours/Semestre 7/PIR OPTIM/Evitement de contrails en free flight avec des methodes de graphes aleatoires/article/images/trajectory/orthodromic/trajectory_ortho_n=15000_sr=40_(3).svg", format='svg')

        plt.show()

    def plot_grid(self, name):

        for (ox, oy, w, h) in self.obs_boundary:
            # self.ax1.add_patch(
            self.map.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            # self.ax1.add_patch(
            self.map.add_patch(

                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            # self.ax1.add_patch(
            self.map.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_ellipse:
            # self.ax1.add_patch(
            self.map.add_patch(
                patches.Ellipse(
                    (ox, oy), 2*w, 2*h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        # self.ax1.plot(self.x_init.long, self.x_init.lat, "bs", linewidth=3)
        # self.ax1.plot(self.x_goal.long, self.x_goal.lat, "rs", linewidth=3)

        self.map.plot(self.x_init.long, self.x_init.lat, "bs", linewidth=3)
        self.map.plot(self.x_goal.long, self.x_goal.lat, "rs", linewidth=3)

        grid_x_ticks = np.arange(self.long_range[0], self.long_range[1], self.step)
        grid_y_ticks = np.arange(self.lat_range[0], self.lat_range[1], self.step)

        # self.ax1.set_xticks(grid_x_ticks)
        # self.ax1.set_yticks(grid_y_ticks)
        # self.ax1.grid(alpha=0.5, linestyle="--")

        # self.map.set_xticks(grid_x_ticks)
        # self.map.set_yticks(grid_y_ticks)
        # self.map.grid(alpha=0.5, linestyle="--")
        
        ## plt.title(name)
        # self.fig.suptitle(name, fontsize = 20)
        ## plt.axis("equal")

    # @jit(nopython=True) 
    def quartering(self, step):
        x_low_limit = self.long_range[0] 
        x_high_limit = self.long_range[1]
        y_low_limit = self.lat_range[0]
        y_high_limit = self.lat_range[1]


        n = (x_high_limit - x_low_limit) / step
        m = (y_high_limit - y_low_limit) / step
        # n = round(n)
        # m = round(m)
        if 0<n-round(n) <= 0.5:
            n = 1+round(n)
        else:
            n = round(n)
        
        if 0 < m-round(m) <= 0.5:
            m = 1+round(m)
        else:
            m = round(m)

        print("n : ", n)
        print("m : ", m)
        tab = [[0]*m for _ in range(n)]

        indice_i = 0
        indice_j = 0
        for i in np.arange(x_low_limit, x_high_limit, step):
            for j in np.arange(y_low_limit, y_high_limit, step):
                # print("j : ", j)    
                point = Node([(i+step/2), (j+step/2)])
                if self.utils.is_inside_obs(point):
                    # print("indice_i : ", indice_i)
                    # print("indice_j : ", indice_j )
                    tab[indice_i][indice_j] = 1
                indice_j += 1
                # print("indice_j = ", indice_j)
            indice_i += 1
            indice_j = 0
            # print("i = ", i)
            # print("indice_i = ", indice_i)
        # print(tab)
        return np.array(tab)

    def load_quartering(self, filename):
        m = 0
        with open(filename, 'r') as f:
            for line in f:
                line.rstrip()
                line = line.split()
                # print(line)
                n = len(line)
                m += 1
        # print("n : ", n)
        # print("m : ", m)
        
        x_low_limit = self.long_range[0] 
        x_high_limit = self.long_range[1]
        y_low_limit = self.lat_range[0]
        y_high_limit = self.lat_range[1]

        self.step = ((x_high_limit - x_low_limit) / n ) + ((y_high_limit - y_low_limit) / m) /2

        tab = [[0]*m for _ in range(n)]
        indice_j = m-1

        with open(filename, 'r') as f:
            for line in f:
                line.rstrip()
                line = line.split()
                for i in np.arange(0, n, 1):
                    tab[i][indice_j] = int(line[i])
                indice_j -= 1
            # print("indice_j : ", indice_j)
        return tab


    def load_samples(self, filename):
        samples = set()
        nb_points = 0
        nb_points_dans_obstacles = 0
        nb_points_dans_espace_libre = 0
        with open(filename, 'r') as f:
            for line in f:
                line.rstrip()
                line = line.split()
                node = Node([float(line[0]), float(line[1])])
                samples.add(node)
                
                indice_node_x = indice(node.long, self.long_range[0], self.step)
                indice_node_y = indice(node.lat, self.lat_range[0], self.step)
                if self.grid[indice_node_x][indice_node_y] == 1:
                    nb_points_dans_obstacles += 1
                else:
                    nb_points_dans_espace_libre += 1
                nb_points += 1
        print("\nnb_points : ", nb_points)
        print("nb_points dans obstacles : ", nb_points_dans_obstacles)
        print("nombre_points dans espace libre : ", nb_points_dans_espace_libre)
        return samples


def temps_sec(temps_sec):
    heure = int(temps_sec//3600)
    min = int((temps_sec%3600)//60)
    sec = int((temps_sec%60)//1)
    millisec = round((temps_sec%1)*1000)
    return heure, min, sec, millisec

def temps_heure_min(temps):
    heure = int(temps)
    min = temps%1
    min= int(min*60)
    return heure,min


# @jit(nopython=False) 
@jit
def calcul_x(xa, xb, ya, yb, y):
    m = (yb - ya)/(xb- xa)
    p = ya - m*xa
    x = (y - p)/m
    return x

# @jit(nopython=False) 
@jit
def calcul_y(xa, xb, ya, yb, x):
    m = (yb - ya)/(xb - xa)
    p = ya - m*xa
    y = m*x + p
    return y

# @jit(nopython=True) 
def save_quartering(tab, filename):
    n = len(tab)
    m = len(tab[0])
    with open(filename, 'w') as f:
        for j in range(m-1,-1,-1):
            for i in range(n):
                f.write(str(tab[i][j]) + "  ")
            f.write("\n")


# @jit(nopython=True) 
def save_samples(samples, filename):
    with open(filename, 'w') as f:
        for node in samples:
            f.write(str(node.long) + " " + str(node.lat) + "\n")



# @jit(nopython=True) 
def get_sample(distributions, lenght_distributions, coefficients):

    num_distr = lenght_distributions
    # data = np.zeros(num_distr)
    data = [(0,0) for _ in range(num_distr)]
    for idx, (distr_x, distr_y) in enumerate(distributions):
        if distr_x == "circle":
            xc, yc, radius, ecart = distr_y
            # r = radius * random.random()
            # r = radius * math.sqrt(random.random())
            r = random.uniform(radius-ecart, radius+ecart)
            theta = 2* math.pi * random.random()
            data[idx] = (xc + r*math.cos(theta), yc + r * math.sin(theta))

        elif distr_x == "ellipse":
            xc, yc, a, b, ecart = distr_y
            a_alea = random.uniform(a-ecart, a+ecart)
            b_alea = random.uniform(b-ecart, b+ecart)
            t = 2*math.pi*random.random()
            # t = random.uniform(0, 2*math.pi)
            data[idx] = (xc + a_alea * math.cos(t), yc + b_alea * math.sin(t))

        else:
            data[idx] = (distr_x["type"](**distr_x["kwargs"]), distr_y["type"](**distr_y["kwargs"]))

    random_idx = np.random.choice(np.arange(num_distr), p=coefficients)
    sample = data[random_idx]
    return sample


# @jit(nopython=False) 
@jit(nopython=True)
def cmp_key(e):
    return e[1]



# @jit(nopython=False) 
@jit(nopython=True)
def calc_dist(x_start_long, x_start_lat, x_end_long, x_end_lat):
    # return math.hypot(x_start_long - x_end_long, x_start_lat - x_end_lat)

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [x_start_long, x_start_lat, x_end_long, x_end_lat])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + (math.cos(lat1) * math.cos(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6391        # radius of earth in kilometers
    return c * r 



# @jit(nopython=False) 
@jit(nopython=True)
def indice(x, x0, step):
    a = x - x0
    b = a // step
    return int(b)

# @jit(nopython=False) 
# @jit
def is_intersect_rec(start_long, start_lat, end_long, end_lat, c:list, d:list):

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
    # elif t2==0:
    #     return Node(c)
    # elif t2==1:
    #     return Node(d)
    else:
        x = start_long + t1*(end_long - start_long)
        y = start_lat + t1*(end_lat - start_lat)
        return ([x,y], t1)


# @jit(nopython=False) 
# @jit
def intersect(start_long, start_lat, end_long, end_lat, indice_x, indice_y, coef_avant_inter, coef_apres_inter, true_start_long, true_start_lat, true_end_long, true_end_lat, step, long_range, lat_range):
    l = []
    # start = Node(start_long, start_lat)
    # end = Node(end_long, end_lat)
    # true_start = Node(true_start_long, true_start_lat)
    # true_end = Node(true_end_long, true_end_lat)

    result1 = is_intersect_rec(start_long, start_lat, end_long, end_lat, [indice_x*step+long_range[0], indice_y*step+lat_range[0]], [indice_x*step+long_range[0], indice_y*step+step+lat_range[0]])
    if result1 != False:
        l.append(result1)
    
    result2 = is_intersect_rec(start_long, start_lat, end_long, end_lat, [indice_x*step+long_range[0], indice_y*step+step+lat_range[0]], [indice_x*step+step+long_range[0], indice_y*step+step+lat_range[0]])
    if result2 != False:
        l.append(result2)
    
    result3 = is_intersect_rec(start_long, start_lat, end_long, end_lat, [indice_x*step+step+long_range[0], indice_y*step+step+lat_range[0]], [indice_x*step+step+long_range[0], indice_y*step+lat_range[0]])
    if result3 != False:
        l.append(result3)
    
    result4 = is_intersect_rec(start_long, start_lat, end_long, end_lat, [indice_x*step+step+long_range[0], indice_y*step+lat_range[0]], [indice_x*step+long_range[0], indice_y*step+lat_range[0]])
    if result4 != False:
        l.append(result4)
    
    if l==[]:
        return False
    else:
        (inter_pt, _) = min(l, key=cmp_key)
        # self.collision_set.add((inter_pt, true_start, true_end))
        return coef_avant_inter * calc_dist(start_long, start_lat, inter_pt[0], inter_pt[1]) + coef_apres_inter * calc_dist(inter_pt[0], inter_pt[1], end_long, end_lat)         





# @jit(nopython=False) 
# @jit
def fonction_aux_y(start_long, start_lat, end_long, end_lat, pas, cond, coef_avant_inter, coef_apres_inter, step, long_range, lat_range, grid):
    # start_x = start.long
    # start_y = start.lat
    # end_x = end.long
    # end_y = end.lat

    nombre_points = round(abs((end_lat - start_lat)) / (step/pas))
    for y in np.linspace(start_lat, end_lat, max(2, nombre_points)):
        x = calcul_x(start_long, end_long, start_lat, end_lat, y)
        indice_y = indice(y, lat_range[0], step)
        indice_x = indice(x, long_range[0], step)
        if grid[indice_x][indice_y] == cond:

            return intersect(start_long, start_lat, end_long, end_lat, indice_x, indice_y, coef_avant_inter, coef_apres_inter, start_long, start_lat, end_long, end_lat, step, long_range, lat_range)

    # print("oups 1")
    # print("start : ",start)
    # print("end : ", end)
    # return self.intersect(start, end, indice_end_x, indice_end_y, coef_avant_inter, coef_apres_inter)

# @jit(nopython=False) 
# @jit
def fonction_aux_x(start_long, start_lat, end_long, end_lat, pas, cond, coef_avant_inter, coef_apres_inter, step, long_range, lat_range, grid):
    # start_x = start.long
    # start_y = start.lat
    # end_x = end.long
    # end_y = end.lat

    nombre_points = round(abs((end_long - start_long)) / (step/pas))
    for x in np.linspace(start_long, end_long, max(2, nombre_points)):
        y = calcul_y(start_long, end_long, start_lat, end_lat, x)

        indice_y = indice(y, lat_range[0], step)
        indice_x = indice(x, long_range[0], step)
        if grid[indice_x][indice_y] == cond:

            return intersect(start_long, start_lat, end_long, end_lat, indice_x, indice_y, coef_avant_inter, coef_apres_inter, start_long, start_lat, end_long, end_lat, step, long_range, lat_range)

    # print("oups 2")
    # print("start : ",start)
    # print("end : ", end)
    # return self.intersect(start, end, indice_end_x, indice_end_y, coef_avant_inter, coef_apres_inter)

# @jit(nopython=False) 
# @jit
def fonction_aux2_y(start_long, start_lat, end_long, end_lat, pas, cost_in_obstacles, step, long_range, lat_range, grid):
    # start_x = start.long
    # start_y = start.lat
    # end_x = end.long
    # end_y = end.lat
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
            inter_2 = [x, y]
            indice_inter_2 = [indice_x, indice_y]
            break

    return intersect(start_long, start_lat, inter_1[0], inter_1[1], indice_inter_1[0], indice_inter_1[1], 1, cost_in_obstacles, start_long, start_lat, end_long, end_lat, step, long_range, lat_range) \
        + intersect(inter_1[0], inter_1[1], end_long, end_lat, indice_inter_2[0], indice_inter_2[1], cost_in_obstacles, 1, start_long, start_lat, end_long, end_lat, step, long_range, lat_range)

# @jit(nopython=False) 
# @jit
def fonction_aux2_x(start_long, start_lat, end_long, end_lat, pas, cost_in_obstacles, step, long_range, lat_range, grid):
    # start_x = start.long
    # start_y = start.lat
    # end_x = end.long
    # end_y = end.lat
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
            inter_2 = [x, y]
            indice_inter_2 = [indice_x, indice_y]
            break

    return intersect(start_long, start_lat, inter_1[0], inter_1[1], indice_inter_1[0], indice_inter_1[1], 1, cost_in_obstacles, start_long, start_lat, end_long, end_lat, step, long_range, lat_range) \
        + intersect(inter_1[0], inter_1[1], end_long, end_lat, indice_inter_2[0], indice_inter_2[1], cost_in_obstacles, 1, start_long, start_lat, end_long, end_lat, step, long_range, lat_range)


# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaWarning)
warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)


# @jit(nopython=True) 
def main():
    # x_start = (18, 8)  # Starting node
    # x_goal = (37, 18)  # Goal node

    # x_start = (-2, 38)  # Starting node
    # x_goal = (17, 48)  # Goal node

    # x_start = (-9, 36)  # Starting node
    # x_goal = (18, 59)  # Goal node

    x_start = (-87.65, 41.85) # Chicago
    x_goal = (2.3519, 48.917) # Paris


    fmt = FMT(x_start, x_goal, search_radius=70, cost_in_obstcles=3, sample_numbers=15000, step=0.1) 
    fmt.Planning()

#########################################
    # fmt.plot_grid("CC")
    # plt.xlim(-1, 52)
    # plt.ylim(-1,32)
    # plt.savefig("/media/gaspard/OS_Install/Users/Gaspard/Desktop/ENAC/2A/Cours/Semestre 7/PIR OPTIM/Evitement de contrails en free flight avec des methodes de graphes aleatoires/article/images/grid/map_grid.pdf", format='pdf')
    # plt.show()
#########################################
    
    # fmt.Init()
    # fmt.plot_nodes()



if __name__ == '__main__':
    main()