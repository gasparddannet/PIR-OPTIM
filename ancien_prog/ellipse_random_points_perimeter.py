import numpy as np
import matplotlib.pyplot as plt
import time

def ellipse_arc(a, b, theta, n):
    """Cumulative arc length of ellipse with given dimensions"""

    # Divide the interval [0 , theta] into n steps at regular angles
    t = np.linspace(0, theta, n)

    # Using parametric form of ellipse, compute ellipse coord for each t
    x, y = np.array([a * np.cos(t), b * np.sin(t)])

    ### Compute vector distance between each successive point
    x_diffs, y_diffs = x[1:] - x[:-1], y[1:] - y[:-1]

    cumulative_distance = [0]
    c = 0

    # Iterate over the vector distances, cumulating the full arc
    for xd, yd in zip(x_diffs, y_diffs):
        c += np.sqrt(xd**2 + yd**2)
        cumulative_distance.append(c)
    cumulative_distance = np.array(cumulative_distance)

    ######################
    # coords = np.array([a * np.cos(t), b * np.sin(t)]) 
    # coords_diffs = np.diff(coords) 
    # cumulative_distance = np.cumsum(np.linalg.norm(coords_diffs, axis=0))
    ######################

    # x_diffs, y_diffs = np.diff(x), np.diff(y)
    # delta_r = np.sqrt(x_diffs**2 + y_diffs**2)
    # cumulative_distance = delta_r.cumsum()
    # c = delta_r.sum()


    # Return theta-values, distance cumulated at each theta,
    # and total arc length for convenience
    return t, cumulative_distance, c


def theta_from_arc_length_constructor(a, b, theta=2*np.pi, n=100):
    """
    Inverse arc length function: constructs a function that returns the
    angle associated with a given cumulative arc length for given ellipse."""

    # Get arc length data for this ellipse
    t, cumulative_distance, total_distance = ellipse_arc(a, b, theta, n)
    print("total distance : ", total_distance)
    # Construct the function
    def f(s):
        assert np.all(s <= total_distance), "s out of range"
        # Can invert through interpolation since monotonic increasing
        return np.interp(s, cumulative_distance, t)

    # return f and its domain
    return f, total_distance


def rand_ellipse(a=2, b=0.5, size=50, precision=100):
    """
    Returns uniformly distributed random points from perimeter of ellipse.
    """
    theta_from_arc_length, domain = theta_from_arc_length_constructor(a, b, theta=2*np.pi, n=precision)
    s = np.random.rand(size) * domain
    t = theta_from_arc_length(s)
    x, y = np.array([a * np.cos(t), b * np.sin(t)])
    return x, y


def rand_ellipse_bad(a, b, n):
    """
    Incorrect method of generating points evenly spaced along ellipse perimeter.
    Points cluster around major axis.
    """
    t = np.random.rand(n) * 2 * np.pi
    return np.array([a * np.cos(t), b * np.sin(t)])



np.random.seed(4987)

x1, y1 = rand_ellipse_bad(2, .5, 10)
debut = time.time()
x2, y2 = rand_ellipse(2, .5, size=800, precision=100)
fin = time.time()
print("Temps d'execution : ", fin-debut)

# print("x2 :", x2)
# print("y2 :", x2)

fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, sharey=True)
fig.suptitle('Generating random points on perimeter of ellipse', size=18)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].scatter(x1, y1, marker="+", alpha=0.5, color="crimson")
ax[1].scatter(x2, y2, marker="+", alpha=0.5, color="forestgreen")
ax[0].set_title("Bad method: Points clustered along major axis")
ax[1].set_title("Correct method: Evenly distributed points")

# Plot arc length as function of theta
theta_from_arc_length, domain = theta_from_arc_length_constructor(2, .5, theta=2*np.pi, n=100)
s_plot = np.linspace(0, domain, 100)
t_plot = theta_from_arc_length(s_plot)

fig, ax = plt.subplots(figsize=(7,7), sharex=True, sharey=True)
ax.plot(t_plot, s_plot)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'cumulative arc length')
plt.show()