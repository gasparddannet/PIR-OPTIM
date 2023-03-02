import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
from shapely.errors import ShapelyDeprecationWarning
from pyproj import Geod
from queue import PriorityQueue

import env, utils, sampling


class Node:
    def __init__(self, n, w_speed=1, w_vect=(1,1)):
        self.long = n[0]
        self.lat = n[1]
        self.parent = None
        self.cost = np.inf
        self.cout_heurist = np.inf
        self.w_speed = w_speed
        self.w_vect = w_vect
    def __repr__(self):
        return "("+str(self.long)+", "+str(self.lat)+")"


class FMT:
    def __init__(self, x_start, x_goal, search_radius, cost_in_obstacles, sample_numbers, step):
        self.x_init = Node(x_start)
        self.x_goal = Node(x_goal)
        self.search_radius = search_radius
        self.cost_in_obstacles =  1 + cost_in_obstacles

        self.env = env.Env()
        self.utils = utils.Utils()

        self.air_speed = 900 # km/h

        ###########################################################################
        ########### Pour afficher un simple graphique avec matplotlib   ###########

        # self.fig, self.map = plt.subplots()
        # self.fig.set_size_inches(16.667, 10)

        ###########################################################################
        ##############  Pour afficher carte du monde avec cartopy   ###############

        self.fig = plt.figure(num = 'MAP', figsize=(16.667, 10))
        self.map = plt.axes(projection=ccrs.PlateCarree())

        # self.map.set_extent([-100, 13, 33, 67], ccrs.PlateCarree())         # CDG - ORD
        # self.map.set_extent([-10, 150, 25, 75], ccrs.PlateCarree())         # CDG - HDN
        self.map.set_extent([-130, 10, 25, 82], ccrs.PlateCarree())         # CDG - LAX

        self.map.coastlines(resolution='50m')
        self.map.add_feature(cfeature.OCEAN.with_scale('50m'))
        self.map.add_feature(cfeature.LAKES.with_scale('50m'))
        self.map.add_feature(cfeature.LAND.with_scale('50m'))
        self.map.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle='dotted', alpha=0.7)
        self.map.add_feature(cfeature.RIVERS.with_scale('10m'))

        grid_lines = self.map.gridlines(draw_labels=True)
        grid_lines.xlabel_style = {'size' : '16'}
        grid_lines.ylabel_style = {'size' : '16'}
        grid_lines.xformatter = LONGITUDE_FORMATTER
        grid_lines.yformatter = LATITUDE_FORMATTER

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
        # self.grid = load_quartering("./quartering/quartering_testtt")

        self.V = set()
        self.V_unvisited = set()
        self.V_open = set()
        self.V_closed = set()
        self.sample_numbers = sample_numbers
        self.rn = self.search_radius * math.sqrt((math.log(self.sample_numbers) / self.sample_numbers))

        print(f"Nombre d'échantillons : {self.sample_numbers}\nSearch radius : {self.search_radius}\nCoût dans obstacles : {cost_in_obstacles}\nStep : {self.step}\n")

    def Init(self):
        # samples = self.Sample()
        # sampling.save_samples(samples, "./samples/samples_heurist_mieux.txt")
        samples = self.load_samples("./samples/samples_heurist_mieux.txt")

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
        eta = 0
        area = (self.long_range[1]-self.long_range[0]) * (self.lat_range[1] - self.lat_range[0])
        gamma = (1 + eta)*2*math.sqrt(1/2) * math.sqrt((area / math.pi))
        print(f"Gamma optimal : {gamma:.3f}") 
        rn = self.search_radius * math.sqrt((math.log(n) / n))
        print(f"rn optimal : {gamma*math.sqrt((math.log(n) / n)):.3f}")
        print(f"rn utilisé : {rn:.3f}")
        Visited = []

        while z is not self.x_goal:
            V_open_new = set()
            X_near = self.Near(self.V_unvisited, z, rn)
            Visited.append(z)

            for x in X_near:

                Y_near = self.Near(self.V_open, x, rn)

                ##################################################
                # cost_list = {y: y.cost + self.Cost(y, x) for y in Y_near}
                # y_min = min(cost_list, key=cost_list.get)

                ###### ou en utilisant une file de priorité ######
                
                q = PriorityQueue()
                for y in Y_near:
                    q.put((y.cost + self.Cost(y,x), y))
                    # q.put((y.cout_heurist + self.Cost(y,x), y))
                _, y_min = q.get()
                ##################################################

                x.parent = y_min
                V_open_new.add(x)
                self.V_unvisited.remove(x)
                x.cost = y_min.cost + self.Cost(y_min, x)
                heurist = utils.calc_dist(x.long, x.lat, self.x_goal.long, self.x_goal.lat)
                x.cout_heurist = x.cost + heurist

            self.V_open.update(V_open_new)
            self.V_open.remove(z)
            self.V_closed.add(z)

            if not self.V_open:
                print("open set empty!")
                break

            ##################################################
            # # cost_open = {y: y.cost for y in self.V_open}
            # cost_open = {y: y.cout_heurist for y in self.V_open}
            # z = min(cost_open, key=cost_open.get)

            ###### ou en utilisant une file de priorité ######

            q = PriorityQueue()
            for y in self.V_open:
                # q.put((y.cost, y))
                q.put((y.cout_heurist, y))
            _, z = q.get()
            ##################################################

        end = time.time()

        path_x, path_y, path = self.ExtractPath()

        minutes, sec, millisec = temps_sec(end-start)
        print(f"Temps d'exécution : {minutes}min {sec:02}s {millisec:03}ms\n")

        # total_cost_verif = self.calc_dist_total(path)
        # print(f"Total cost (en calculant la distance sans aucunes pénalités): {total_cost_verif:.3f}")

        print(f"Total cost : {self.x_goal.cost:.3f}")
        self.plot_results(path_x, path_y, Visited[1: len(Visited)])


    def Near(self, nodelist, z, rn):
        return {nd for nd in nodelist
                if 0 < (nd.long - z.long) ** 2 + (nd.lat - z.lat) ** 2 <= rn ** 2}


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


    def Cost(self, start:Node, end:Node) -> float:
        pas = 5

        vitesse_vent = start.w_speed
        vecteur_vent = start.w_vect
        vecteur_vitesse = (end.long-start.long, end.lat-start.lat)
        theta = math.acos(np.dot(vecteur_vent, vecteur_vitesse) / (np.linalg.norm(vecteur_vent) * np.linalg.norm(vecteur_vitesse)))
        vitesse_sol = math.pow(math.pow(self.air_speed, 2) - math.pow(vitesse_vent*math.sin(theta), 2), 1/2) + vitesse_vent * math.cos(theta)

        indice_start_long = utils.indice(start.long, self.long_range[0], self.step)
        indice_start_lat = utils.indice(start.lat, self.lat_range[0], self.step)
        indice_end_long = utils.indice(end.long, self.long_range[0], self.step)
        indice_end_lat = utils.indice(end.lat, self.lat_range[0], self.step)
        

        if self.grid[indice_start_long][indice_start_lat] == 1 and self.grid[indice_end_long][indice_end_lat] == 1:
            # si les deux extrémités du segment sont dans un obstacle
            return (self.cost_in_obstacles * utils.calc_dist(start.long, start.lat, end.long, end.lat)) #/vitesse_sol           # on divise par la vitesse si on veut le temps de vol au lieu de la distance
        

        elif self.grid[indice_start_long][indice_start_lat] == 0 and self.grid[indice_end_long][indice_end_lat] == 1:
            # si la fin du segment est dans un obstacle mais pas le debut
            if abs(end.long - start.long) < abs(end.lat - start.lat):
                # on regarde comment est le segment pour le balayer soit selon la latitude (y) ou la longitude (x)
                return self.utils.sweep_y(start.long, start.lat, end.long, end.lat, pas, 1, 1, self.cost_in_obstacles, self.step, self.long_range, self.lat_range, self.grid) #/ vitesse_sol
            else:
                return self.utils.sweep_x(start.long, start.lat, end.long, end.lat, pas, 1, 1, self.cost_in_obstacles, self.step, self.long_range, self.lat_range, self.grid) #/ vitesse_sol


        elif self.grid[indice_start_long][indice_start_lat] == 1 and self.grid[indice_end_long][indice_end_lat] == 0:        
            # si le debut du segment est dans un obstacle mais pas la fin
            if abs(end.long - start.long) < abs(end.lat - start.lat):
                return self.utils.sweep_y(start.long, start.lat, end.long, end.lat, pas, 0, self.cost_in_obstacles, 1, self.step, self.long_range, self.lat_range, self.grid) #/ vitesse_sol
            else:
                return self.utils.sweep_x(start.long, start.lat, end.long, end.lat, pas, 0, self.cost_in_obstacles, 1, self.step, self.long_range, self.lat_range, self.grid) #/ vitesse_sol

        else:
            # les deux extrémités du segment ne sont pas dans un obstacle
            # cas 1 : le segement ne rentre pas dans un obstacle
            # cas 2 : le segment passe un petit peu dans un obstacle 

            # on essaye de trouver (si ils existent) les deux d'intersections associés au passage dans un obstacle
            if abs(end.long - start.long) < abs(end.lat - start.lat):
                result = self.utils.sweep_y_2(start.long, start.lat, end.long, end.lat, pas, self.cost_in_obstacles, self.step, self.long_range, self.lat_range, self.grid) 
            else:
                result = self.utils.sweep_x_2(start.long, start.lat, end.long, end.lat, pas, self.cost_in_obstacles, self.step, self.long_range, self.lat_range, self.grid) 
                
            if result == None:
                # le segment n'intersecte aucun obstacle
                return utils.calc_dist(start.long, start.lat, end.long, end.lat) #/ vitesse_sol
            else:
                # le segment intersecte un obstacle
                return result #/ vitesse_sol


    def calc_dist_total(self, path):
        total_cost = 0
        length = len(path)
        for i in np.arange(0, length-1, 1):
            total_cost += utils.calc_dist(path[i].long, path[i].lat, path[i+1].long, path[i+1].lat)
        return total_cost


    def proportion_obstacles(self):
        n = len(self.grid)
        m = len(self.grid[0])
        nombre_points_dans_obstacles = 0
        for i in np.arange(0, n, 1):
            for j in np.arange(0, m, 1):
                if self.grid[i][j] == 1:
                    nombre_points_dans_obstacles += 1
        proportion_obstacles = nombre_points_dans_obstacles / (n*m)
        return proportion_obstacles


    def Sample(self):
        """Créer l'échantillonnage"""
        start = time.time()

        distributions, coefficients = sampling.create_distributions(self.long_range, self.lat_range, self.obs_circle, self.obs_ellipse, self.obs_rectangle)
        num_distr = len(distributions)

        Sample = set()
        proportion_obstacles = self.proportion_obstacles()
        nb_points_dans_obstacles = round(proportion_obstacles * self.sample_numbers)
        nb_points_dans_espace_libre = self.sample_numbers - nb_points_dans_obstacles

        cpt_dans_obstacles = 0
        cpt_dans_espace_libre = 0 

        while cpt_dans_obstacles < nb_points_dans_obstacles or cpt_dans_espace_libre < nb_points_dans_espace_libre:
            
            ##################################################################
            node = Node(sampling.get_sample(distributions, num_distr, coefficients))
            
            #### Pour mettre du vent contraire en fonction de la latitude ####
            # x,y = get_sample(distributions, num_distr, coefficients)
            # if y>(self.lat_range[1]-self.lat_range[0])/2:
            #     node = Node((x, y), 100, (-10, 0))
            # else:
            #     node = Node((x, y), 100, (10, 0))

            ###############      Echantillonnage uniforme      ###############
            # node = Node((random.uniform(self.long_range[0], self.long_range[1]),
            #              random.uniform(self.lat_range[0], self.lat_range[1])))
            ##################################################################

            if node.long>self.long_range[0] and node.long < self.long_range[1] and node.lat>self.lat_range[0] and node.lat<self.lat_range[1]:
                indice_node_x = utils.indice(node.long, self.long_range[0], self.step)
                indice_node_y = utils.indice(node.lat, self.lat_range[0], self.step)

                if self.grid[indice_node_x][indice_node_y] == 1 and cpt_dans_obstacles < nb_points_dans_obstacles:
                    Sample.add(node)
                    cpt_dans_obstacles += 1

                if self.grid[indice_node_x][indice_node_y] == 0 and cpt_dans_espace_libre < nb_points_dans_espace_libre:
                    Sample.add(node)
                    cpt_dans_espace_libre += 1
                
        print(f"Proportion obstacles : {proportion_obstacles:.3f}")
        print("Nb points dans obstacles : ", cpt_dans_obstacles)
        print("Nb points dans esapce libre : ", cpt_dans_espace_libre)
        
        end = time.time()
        print(f"Temps d'execution sample: {end-start:.3f}")
        return Sample

    def plot_nodes(self):
        """Affiche que l'échantillonnage"""
        self.plot_env()
        for node in self.V:
            self.map.plot(node.long, node.lat, marker='.', color='green', markersize=5, alpha=0.7)
            # self.map.plot(node.long, node.lat, marker='o', color='magenta', markersize=1, alpha=0.8, transform=ccrs.PlateCarree())

        plt.axis("off")
        plt.show()
    
    def plot_results(self, path_x, path_y, visited):

        ########################################################################################################
        ############################ Trajectoire euclidienne sans utiliser cartopy #############################
        # for node in self.V:
        #     self.map.plot(node.long, node.lat, marker='.', color='green', markersize=5, alpha=0.7)  #lightgrey

        # self.map.plot(path_x, path_y, linewidth=4, color='red')
        # self.plot_env()

        # # plt.axis("off")
        # # plt.savefig("/media/gaspard/OS_Install/Users/Gaspard/Desktop/ENAC/2A/Cours/Semestre 7/PIR OPTIM/Evitement de contrails en free flight avec des methodes de graphes aleatoires/article/images/trajectory/euclidean/presentation/trajectory_euclidean_n=10000_sr=50_cr=1_(2).pdf", format='pdf')
        # plt.show()
        
        ########################################################################################################
        ############################                 Animation                     #############################
        
        for node in self.V:
            self.map.plot(node.long, node.lat, marker='.', color='green', markersize=5, alpha=0.7)
        self.plot_env()
        count = 0
        for node in visited:
            count += 1
            self.map.plot([node.long, node.parent.long], [node.lat, node.parent.lat], linestyle="-", color="dodgerblue")
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if count % (self.sample_numbers/8) == 0:
                plt.pause(0.000001)

        self.map.plot(path_x, path_y, linewidth=4, color='red')
        plt.show()

        
        ########################################################################################################
        ###########################         Trajectoire orthodromique cartopy        ###########################
        # for node in self.V:
        #     self.map.plot(node.long, node.lat, marker='.', color='green', markersize=5, alpha=0.7, transform=ccrs.PlateCarree())
        # self.map.plot(path_x, path_y, linewidth=3, color='red', transform=ccrs.PlateCarree(), label="Computed trajectory")
        # ######
        # geod = Geod(ellps='WGS84')
        # nb = 10000
        # traj = geod.npts(self.x_init.long, self.x_init.lat, self.x_goal.long, self.x_goal.lat, nb)
        # distance = 0    
        # list_long = []
        # list_lat = []
        # for i in np.arange(0, nb, 1):
        #     list_long.append(traj[i][0])
        #     list_lat.append(traj[i][1])

        # self.map.plot(list_long, list_lat, transform = ccrs.Geodetic(), color ='blue',linewidth=3, linestyle="--", label="Great-circle trajectory")   
        # distance = geod.line_length(list_long, list_lat)
        # distance/=1000
        # print(f"Distance trajectoire orthodromique : {distance:.3f} km")
        # print(f"Distance trajectoire calculée : {self.x_goal.cost:.3f} km")
        # # temps = distance/self.air_speed
        # #######
        # self.map.text(self.x_init.long-3, self.x_init.lat-4, "Paris", transform=ccrs.PlateCarree(), color='black', alpha = 1, fontsize = 24)
        # # self.map.text(self.x_goal.long-3, self.x_goal.lat-5, "Tokyo", transform=ccrs.PlateCarree(), color='black', alpha = 1, fontsize = 24)
        # self.map.text(self.x_goal.long-10, self.x_goal.lat-4, "Los Angeles", transform=ccrs.PlateCarree(), color='black', alpha = 1, fontsize = 24) #, bbox=dict(boxstyle="square"))#, facecolor="white"))

        # self.plot_env()

        # plt.legend(fontsize = 20)
        # plt.legend(loc="lower center", fontsize = 20)
        # # plt.tight_layout()

        # # plt.savefig("/media/gaspard/OS_Install/Users/Gaspard/Desktop/ENAC/2A/Cours/Semestre 7/PIR OPTIM/Evitement de contrails en free flight avec des methodes de graphes aleatoires/article/images/trajectory/orthodromic/presentation/trajectory_great-circle_n=15000_sr=80_obstacles_conclu_2.pdf", format='pdf')

        # plt.show()
        ###########################################################################################

    def plot_env(self):
        """Affiche l'environnement (osbtacles et frontières) et point initial et point final"""
        for (ox, oy, w, h) in self.obs_boundary:
            self.map.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )
        for (ox, oy, w, h) in self.obs_rectangle:
            self.map.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        for (ox, oy, r) in self.obs_circle:
            self.map.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )
        for (ox, oy, w, h) in self.obs_ellipse:
            self.map.add_patch(
                patches.Ellipse(
                    (ox, oy), 2*w, 2*h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        self.map.plot(self.x_init.long, self.x_init.lat, "bs", linewidth=3)
        self.map.plot(self.x_goal.long, self.x_goal.lat, "rs", linewidth=3)

        # plt.axis("off")

        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)


    def quartering(self, step):
        """Créer le quadrillage, le bitmap de l'environnement"""
        x_low_limit = self.long_range[0]
        x_high_limit = self.long_range[1]
        y_low_limit = self.lat_range[0]
        y_high_limit = self.lat_range[1]


        n = (x_high_limit - x_low_limit) / step
        m = (y_high_limit - y_low_limit) / step

        if 0<n-round(n) <= 0.5:
            n = 1+round(n)
        else:
            n = round(n)
        
        if 0 < m-round(m) <= 0.5:
            m = 1+round(m)
        else:
            m = round(m)
        # print("n : ", n)
        # print("m : ", m)
        tab = [[0]*m for _ in range(n)]

        indice_i = 0
        indice_j = 0
        for i in np.arange(x_low_limit, x_high_limit, step):
            for j in np.arange(y_low_limit, y_high_limit, step):
                point = Node([(i+step/2), (j+step/2)])
                if self.utils.is_inside_obs(point):
                    tab[indice_i][indice_j] = 1
                indice_j += 1
            indice_i += 1
            indice_j = 0
        return np.array(tab)

    def load_quartering(self, filename):
        m = 0
        with open(filename, 'r') as f:
            for line in f:
                line.rstrip()
                line = line.split()
                n = len(line)
                m += 1
        
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
                
                indice_node_x = utils.indice(node.long, self.long_range[0], self.step)
                indice_node_y = utils.indice(node.lat, self.lat_range[0], self.step)
                if self.grid[indice_node_x][indice_node_y] == 1:
                    nb_points_dans_obstacles += 1
                else:
                    nb_points_dans_espace_libre += 1
                nb_points += 1
        print("\nnb_points : ", nb_points)
        print("nb_points dans obstacles : ", nb_points_dans_obstacles)
        print("nombre_points dans espace libre : ", nb_points_dans_espace_libre)
        return samples


    def dessiner_quadrillage(self):
        n = len(self.grid)
        m = len(self.grid[0])
        quadrillage_noir = []
        quadrillage_blanc = []

        for i in range(n):
            for j in range(m):
                if self.grid[i][j] == 1:
                    quadrillage_noir.append([i*self.step+self.long_range[0], j*self.step+self.lat_range[0], self.step, self.step])
                if self.grid[i][j] == 0:
                    quadrillage_blanc.append([i*self.step+self.long_range[0], j*self.step+self.lat_range[0], self.step, self.step])

        for (ox, oy, w, h) in quadrillage_noir:
            self.map.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor= 'darkgrey',
                    facecolor='grey',
                    fill=True
                )
            )
        for (ox, oy, w, h) in quadrillage_blanc:
            self.map.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='white',
                    facecolor='white',
                    fill=True
                )
            )
        for (ox, oy, w, h) in self.obs_boundary:
            self.map.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )
        plt.axis("off")

@jit(nopython=True)
def temps_sec(temps_sec):
    min = int((temps_sec)//60)
    sec = int((temps_sec%60)//1)
    millisec = round((temps_sec%1)*1000)
    return min, sec, millisec


def save_quartering(tab, filename):
    n = len(tab)
    m = len(tab[0])
    with open(filename, 'w') as f:
        for j in range(m-1,-1,-1):
            for i in range(n):
                f.write(str(tab[i][j]) + "  ")
            f.write("\n")



# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaWarning)
warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)



def main():
    ###   env_init   ###
    # x_start = (18, 8)  # Starting node
    # x_goal = (37, 18)  # Goal node

    #--- env init 2 ---#
    # x_start = (5, 25)  # Starting node
    # x_goal = (48, 9)  # Goal node


    ### cartopy en utilisant vrais aéroports ###
    x_start = (2.547778, 49.009722)             # Paris (CDG)

    # x_goal = (-87.904722, 41.978611)            # Chicago (ORD)
    # x_goal = (139.781111, 35.553333)            # Tokyo-Haneda (HND)
    x_goal = (-118.408056, 33.9425)             # Los Angeles (LAX)


    fmt = FMT(x_start, x_goal, search_radius=70, cost_in_obstacles=1, sample_numbers=8000, step=0.1)
    fmt.Planning()


    distance = utils.calc_dist(x_start[0], x_start[1], x_goal[0], x_goal[1])
    print(f"Distance orthodromique calculée avec calc_dist : {distance:.3f} km")

if __name__ == '__main__':
    main()