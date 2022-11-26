import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None

    def __repr__(self):
        return "("+str(self.x)+", "+str(self.y)+")"
    
    def equal(self, node1):
        if self.x == node1.x and self.y == node1.y:
            return True
        return False

def is_intersect_rec2(start, end, c, d):
    cx, cy = c[0], c[1]
    dx, dy = d[0], d[1]
    
    det  = (end.x - start.x) * (cy - dy) - (cx - dx)*(end.y - start.y)
    if det == 0:
        return False

    t1 = ((cx - start.x)*(cy - dy) - (cx - dx)*(cy - start.y))/det
    t2 = ((end.x - start.x)*(cy - start.y) - (cx - start.x)*(end.y - start.y))/det

    if t1>1 or t1<0 or t2>1 or t2<0:
    # if t1>1 or t1<0:
    # 
        return False
    
    elif t1==0:
        return (start,t1)
    elif t1==1:
        return (end,t1)
    # elif t2==0:
    #     return (Node(c),t1)
    # elif t2==1:
    #     return (Node(d),t1)
    else:
        x = start.x + t1*(end.x - start.x)
        print("x : ", x)
        y = start.y + t1*(end.y - start.y)
        print("y : ",y)
        print("\n")
        return (Node([x,y]), t1)




def is_intersect_rec3(start, end, v1, v2, v3, v4):
    # l = [v1, v2, v3, v4]
    # i = 0

    cpt=[]
    # while len(cpt) < 2:
        # result = self.is_intersect_rec2(start, end, o, direc, l[i], l[i+1])

    result1 = is_intersect_rec2(start, end, v1, v2)
    if result1 != False:
        cpt.append(result1)


    result2 = is_intersect_rec2(start, end, v2, v3)
    if result2 != False:
        cpt.append(result2)

    result3 = is_intersect_rec2(start, end, v3, v4)
    if result3 != False:
        cpt.append(result3)

    
    result4 = is_intersect_rec2(start, end, v4, v1)
    if result4 != False:
        cpt.append(result4)

    print("cpt = ", cpt)
    if cpt==[]:
        return False

    elif len(cpt) == 1:
        print("KWA de utils.py ?")

    elif len(cpt) == 2:
        return [cpt[0], cpt[1]]

    else:
        if not cpt[0][0].equal(cpt[1][0]):
            return [cpt[0], cpt[1]]
        else:
            return [cpt[0], cpt[2]]
        

def is_intersect_cirlce2(start, end, center, r):
    xc = center[0]
    yc = center[1]

    a = (end.x - start.x)**2 + (end.y - start.y)**2
    b = 2*( (end.x - start.x)*(start.x - xc) + (end.y - start.y)*(start.y - yc) )
    c = (start.x - xc)**2 + (start.y - yc)**2 - r**2
    delta = b**2 - 4*a*c
    if delta < 0:
        return False
    elif delta == 0:
        t = -b/(2*a)
        if 0 <= t <= 1:
            x = start.x + t*(end.x - start.x)
            y = start.y + t*(end.y - start.y)
            print("cas 1")
            return [Node([x,y])]

        else:
            return False
    else:
        t1 = (-b - math.sqrt(delta))/(2*a)
        t2 = (-b + math.sqrt(delta))/(2*a)
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            x1 = start.x + t1*(end.x - start.x)
            y1 = start.y + t1*(end.y - start.y)
            x2 = start.x + t2*(end.x - start.x)
            y2 = start.y + t2*(end.y - start.y)

            return [(Node([x1,y1]), t1), (Node([x2,y2]), t2)]

        elif 0 <= t1 <= 1:
            x1 = start.x + t1*(end.x - start.x)
            y1 = start.y + t1*(end.y - start.y)
            return [Node([x1,y1])]
        elif 0 <= t2 <= 1:
            x2 = start.x + t2*(end.x - start.x)
            y2 = start.y + t2*(end.y - start.y)
            return [Node([x2,y2])]
        else:
            return False



##########################################################################
##  DEUX SEGMENTS

# A = Node([1,2])
# B = Node([4,5])
# C = [2,0]
# D = [4,6.2]


# plt.plot(A.x, A.y)
# plt.plot(B.x, B.y)
# plt.plot(C[0], C[0])
# plt.plot(D[0], D[1])

# plt.plot([A.x, B.x], [A.y, B.y], 'red')
# plt.plot([C[0], D[0]], [C[1], D[1]], 'blue')


# result = is_intersect_rec2(A,B,C,D)
# if result != False:
#     print(result)
#     pt = result[1]
#     plt.plot(pt.x, pt.y, marker='o', color='green', markersize=5)

# plt.show()

##########################################################################
##  CERCLE


# E = Node([-10,0])
# F = Node([10, 7])


# E = Node([-4,9])
# F = Node([10, 9])

E = Node([0,4])
F = Node([10, 9])

# figure, axes = plt.subplots()

# center = [2,5]
# r = 4

# axes.add_patch(patches.Circle((center[0],center[1]), r, fill=False))

# plt.plot(E.x, E.y)
# plt.plot(F.x, F.y)
# plt.plot([E.x, F.x], [E.y, F.y], 'red')

# result = is_intersect_cirlce2(E, F, center, r)
# if result != False:
#     # print(result)
#     if len(result)==1:
#         pt = result[0]
#         plt.plot(pt.x, pt.y, marker='o', color='green', markersize=5)
#     else:
#         list_pts = result
#         for pt in list_pts:
#             plt.plot(pt[0].x, pt[0].y, marker='o', color='green', markersize=5)


# plt.xlim([-5,10])
# plt.ylim([-5,13])
# plt.show()







##########################################################################
##  RECTANGlE

# rectangle = [1, 1, 8, 3]

# figure, axes = plt.subplots()

# axes.add_patch(patches.Rectangle((rectangle[0], rectangle[1]), rectangle[2], rectangle[3],edgecolor='black', facecolor='gray',fill=True))

# # A = Node([9,0])
# # A = Node([-3,-2])
# # B = Node([9,10])

# A = Node([0,4.375])
# B = Node([10,0.625])

# C = Node([1,2])
# D = Node([1,2])

# if C==D:
#     print("equal")
# else:
#     print("not equal")

# plt.plot(A.x, A.y)
# plt.plot(B.x, B.y)
# plt.plot([A.x, B.x], [A.y, B.y], 'red')


# v1 = [rectangle[0], rectangle[1]]
# v2 = [rectangle[0]+rectangle[2], rectangle[1]]
# v3 = [rectangle[1] + rectangle[2], rectangle[1]+rectangle[3]]
# v4 = [rectangle[1], rectangle[1] + rectangle[3]]

# plt.plot(v1[0], v1[1], marker='o', color='magenta', markersize=5)
# plt.plot(v2[0], v2[1], marker='o', color='magenta', markersize=5)
# plt.plot(v3[0], v3[1], marker='o', color='magenta', markersize=5)
# plt.plot(v4[0], v4[1], marker='o', color='magenta', markersize=5)


# result = is_intersect_rec3(A,B,v1, v2, v3, v4)
# print(result)

# for pt in result:
#     plt.plot(pt[0].x, pt[0].y, marker='o', color='green', markersize=5)

# plt.xlim([-2,11])
# plt.ylim([-2,8])
# plt.show()


#########################################################################

# import numpy as np
# import matplotlib.pyplot as plt

# distributions = [
#     {"type": np.random.normal, "kwargs": {"loc": -3, "scale": 2}},
#     # {"type": np.random.uniform, "kwargs": {"low": 4, "high": 6}},
#     {"type": np.random.normal, "kwargs": {"loc": 2, "scale": 1}},
# ]
# coefficients = np.array([0.5, 0.2])
# # coefficients = np.array([0.33, 0.33, 0.33])

# coefficients /= coefficients.sum()      # in case these did not add up to 1
# sample_size = 1000

# num_distr = len(distributions)
# data = np.zeros((sample_size, num_distr))
# # data = np.zeros(num_distr)

# # print(data)
# for idx, distr in enumerate(distributions):
#     data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
#     # data[idx] = distr["type"](**distr["kwargs"])


# random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
# # random_idx = np.random.choice(np.arange(num_distr), p=coefficients)

# # print(random_idx)

# sample = data[np.arange(sample_size), random_idx]
# # sample = data[random_idx]


# # print(data)
# # print(sample)
# # print(data[1][1])

# plt.hist(sample, bins=100, density=True)


# from scipy.stats import norm



# x = np.linspace(-10,10,100)
# plt.plot(x, (norm.pdf(x, -3, 2) + norm.pdf(x, 2, 1))/2,
#        'r-', lw=5, alpha=0.6, label='norm pdf')

# plt.show()

#########################################################################


# import matplotlib.pyplot as plt
# import numpy as np
# mu = 0
# sigma = 0.1
# s = np.random.normal(mu, sigma, 1000)
# count, bins, ignored = plt.hist(s, 30, density=True)
# print(bins)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
#          linewidth=2, color='r')
# plt.show()



#########################################################################



import matplotlib.pyplot as plt
import numpy as np

# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

# fig = plt.figure()
# gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
# (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
# fig.suptitle('Sharing x per column, y per row')
# ax1.plot(x, y)
# ax2.plot(x, y**2, 'tab:orange')
# ax3.plot(x + 1, -y, 'tab:green')
# ax4.plot(x + 2, -y**2, 'tab:red')

# for ax in fig.get_axes():
#     ax.label_outer()

#########################################################################
#########################################################################


# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# fig.suptitle('Sharing x per column, y per row')
# ax1.plot(x, y)
# ax2.plot(x, y**2, 'tab:orange')
# ax3.plot(x, -y, 'tab:green')
# # ax4.plot(x, -y**2, 'tab:red')

# for ax in fig.get_axes():
#     ax.label_outer()

# plt.show()

#########################################################################
#########################################################################


# fig = plt.figure()
# gs = fig.add_gridspec(3, hspace=0)
# axs = gs.subplots(sharex=True, sharey=True)
# fig.suptitle('Sharing both axes')
# axs[0].plot(x, y ** 2)
# axs[1].plot(x, 0.3 * y, 'o')
# axs[2].plot(x, y, '+')

# # Hide x labels and tick labels for all but bottom plot.
# for ax in axs:
#     ax.label_outer()
# plt.show()






#########################################################################


# from scipy.stats import norm
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1)

# mu = 5
# sigma = 1
# # x = np.linspace(norm.ppf(0.01),
# #                 norm.ppf(0.99), 100)
# # print(x)

# x = np.linspace(0,10,100)
# ax.plot(x, norm.pdf(x, mu, sigma),
#        'r-', lw=5, alpha=0.6, label='norm pdf')
# plt.show()

#########################################################################


# fig3 = plt.figure(constrained_layout=True)
# gs = fig3.add_gridspec(3, 3)
# f3_ax1 = fig3.add_subplot(gs[0, :])
# f3_ax1.set_title('gs[0, :]')
# f3_ax2 = fig3.add_subplot(gs[1, :-1])
# f3_ax2.set_title('gs[1, :-1]')
# f3_ax3 = fig3.add_subplot(gs[1:, -1])
# f3_ax3.set_title('gs[1:, -1]')
# f3_ax4 = fig3.add_subplot(gs[-1, 0])
# f3_ax4.set_title('gs[-1, 0]')
# f3_ax5 = fig3.add_subplot(gs[-1, -2])
# f3_ax5.set_title('gs[-1, -2]')
# plt.show()
#########################################################################



# fig3 = plt.figure(constrained_layout=True)
# gs = fig3.add_gridspec(2, 2)
# f3_ax1 = fig3.add_subplot(gs[0, :])
# f3_ax1.set_title('gs[0, :]')
# f3_ax2 = fig3.add_subplot(gs[1, :-1])
# f3_ax2.set_title('gs[1, :-1]')
# f3_ax3 = fig3.add_subplot(gs[1:, -1])
# f3_ax3.set_title('gs[1:, -1]')
# f3_ax4 = fig3.add_subplot(gs[-1, 0])
# f3_ax4.set_title('gs[-1, 0]')

# plt.show()


#########################################################################

# fig5 = plt.figure(constrained_layout=True)
# widths = [4, 1]
# heights = [4, 1]
# spec5 = fig5.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
#                           height_ratios=heights)

# ax1 = fig5.add_subplot(spec5[0,0]) 
# ax2 = fig5.add_subplot(spec5[0,1])
# ax3 = fig5.add_subplot(spec5[1,0])
# plt.show()

#########################################################################


import math
import random
import matplotlib.pyplot as plt
import numpy as np

def NonUniformRandomPointInCirlce(inputRadius=1, xcenter=0, ycenter=0, ecart=0.2):
    # r = inputRadius * random.random()
    r = random.uniform(inputRadius-ecart, inputRadius+ecart)

    theta = 2 * math.pi * random.random()
    return xcenter + r*math.cos(theta), ycenter + r*math.sin(theta)


def UniformRandomPointInCirlce(inputRadius=1, xcenter=0, ycenter=0, ecart=0.2):
    # r = inputRadius * math.sqrt(random.random())
    r = math.sqrt(random.uniform(inputRadius-ecart, inputRadius+ecart))

    theta = 2 * math.pi * random.random()
    return xcenter + r*math.cos(theta), ycenter + r*math.sin(theta)


def ReplicateNTimes(func, rad, xc, yc, Ntrials=1000):
    xpoints, ypoints = [], []
    for _ in range(Ntrials):
        xp, yp = func(rad, xc,yc)
        xpoints.append(xp)
        ypoints.append(yp)
    dist = (xpoints, ypoints, rad,  xc, yc)
    return dist


def PlotDistribution(**kwargs):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    def PlotDist(xpoints, ypoints, rad, xc, yc, subtitleinfo, i, color, marker):
        plt.subplot(1, 2, i+1)
        plt.plot(xpoints, ypoints, color + marker)

        # Plot a unit cirlce
        rng = np.arange(0, math.pi * 2, 0.01)
        x_circle = [xc + rad * math.cos(v) for v in rng]
        y_circle = [yc + rad * math.sin(v) for v in rng]
        plt.plot(x_circle, y_circle, '-k')
        plt.title(subtitleinfo)
        plt.axis("square")

    colors = ['b', 'r']
    markers = ['.', '.']
    for i, key in enumerate(kwargs):
        # print("i =", i)
        # print("key =", key)
        # print("kwargs[key] =", kwargs[key])
        (xpoints, ypoints, rad, xc, yc), subtitleinfo = kwargs[key]
        PlotDist(xpoints, ypoints, rad, xc, yc, subtitleinfo, i, colors[i], markers[i])

    plt.show()

def main():
    Ntrials = 5000
    radius, xc, yc = 2, 0, 0

    dist1 = ReplicateNTimes(UniformRandomPointInCirlce, radius, xc, yc, Ntrials=Ntrials)
    dist2 = ReplicateNTimes(NonUniformRandomPointInCirlce, radius, xc, yc, Ntrials=Ntrials)

    PlotDistribution(Uniform=(dist1, "Using sqrt(rand())"), NonUnifrom=(dist2, "Using rand()"))

main()