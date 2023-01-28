import random
import math
import numpy as np

def create_distributions(long_range, lat_range, obs_circle, obs_ellipse,obs_rectangle):
    """Créer toutes les distributions associées aux obstacles et à l'espace tout entier"""
    distributions = []
    coefficients = []
    somme_coefficients = 0
    area = (long_range[1] - long_range[0]) * (lat_range[1] - lat_range[0])
    # print("area : ", area)
    ecart = 1
    for (x, y, r) in obs_circle:
        xc = x
        yc = y
        radius = r
        distributions.append(("circle", (xc, yc, radius, ecart)))
        p = math.pi * radius**2
        coefficients.append(p)
        somme_coefficients += p

    for (x, y, w, h) in obs_ellipse:
        xc = x
        yc = y
        distributions.append(("ellipse", (xc, yc, w, h, ecart)))
        p = math.pi * w * h
        coefficients.append(p)
        somme_coefficients += p

    for (x, y, w, h) in obs_rectangle:
        # pour un rectangle on ajoute 4 distributions qui représentent les 4 côtés du rectangle
        area_rect = w * h
        low_x = x - ecart
        high_x = x + ecart
        low_y = y - ecart
        high_y = y + h + ecart
        distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
        p = 2*ecart*h * (area_rect/(4*ecart*(h+w))) 
        coefficients.append(p)
        somme_coefficients += p
        ##############
        low_x = x - ecart
        high_x = x + w + ecart
        low_y = y - ecart
        high_y = y + ecart
        distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
        p = 2*ecart*w * (area_rect/(4*ecart*(h+w)))  
        coefficients.append(p)
        somme_coefficients += p
        ##############
        low_x = x + w - ecart
        high_x = x + w + ecart
        low_y = y - ecart
        high_y = y + h + ecart
        distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
        p = 2*ecart*h * (area_rect/(4*ecart*(h+w))) 
        coefficients.append(p)
        somme_coefficients += p
        ##############
        low_x = x - ecart
        high_x = x + w + ecart
        low_y = y + h - ecart
        high_y = y + h + ecart
        distributions.append(({"type": np.random.uniform, "kwargs": {"low": low_x, "high": high_x}},
                                {"type": np.random.uniform, "kwargs": {"low": low_y, "high": high_y}}))
        p = 2*ecart*w * (area_rect/(4*ecart*(h+w)))  
        coefficients.append(p)
        somme_coefficients += p

    distributions.append(({"type":np.random.uniform, "kwargs":{"low": long_range[0], "high": long_range[1]}},
                            {"type":np.random.uniform, "kwargs":{"low": lat_range[0], "high": lat_range[1]}}))
    p = (area - somme_coefficients)/4           # on divide par 4 pour diminuer la proba d'echantillonner dans l'espace libre
    coefficients.append(p)
    coefficients = np.array(coefficients)
    coefficients = coefficients / np.sum(coefficients)
    return distributions, coefficients



#####################################################################################################
################## To uniformly distribute random points from perimeter of ellipse ##################
#####################################################################################################

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

    # Return theta-values, distance cumulated at each theta,
    # and total arc length for convenience
    return t, cumulative_distance, c

def theta_from_arc_length_constructor(a, b, theta=2*np.pi, n=100):
    """
    Inverse arc length function: constructs a function that returns the
    angle associated with a given cumulative arc length for given ellipse."""

    # Get arc length data for this ellipse
    t, cumulative_distance, total_distance = ellipse_arc(a, b, theta, n)

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

###############################################################################################################
def get_sample(distributions, lenght_distributions, coefficients):
    """Donne un point à partir d'une distribution"""
    num_distr = lenght_distributions
    data = [(0,0) for _ in range(num_distr)]
    for idx, (distr_x, distr_y) in enumerate(distributions):
        if distr_x == "circle":
            xc, yc, radius, ecart = distr_y
            r = random.uniform(radius-ecart, radius+ecart)
            theta = 2* math.pi * random.random()
            data[idx] = (xc + r*math.cos(theta), yc + r * math.sin(theta))

        elif distr_x == "ellipse":
            xc, yc, a, b, ecart = distr_y
            a_alea = random.uniform(a-ecart, a+ecart)
            b_alea = random.uniform(b-ecart, b+ecart)
            
            ##### Pas uniforme #####
            # t = 2*math.pi*random.random()
            # data[idx] = (xc + a_alea * math.cos(t), yc + b_alea * math.sin(t))

            ####### Uniforme #######
            x, y = rand_ellipse(a_alea, b_alea, size=1, precision=50)        
            data[idx] = (xc + x[0], yc + y[0])
        else:
            data[idx] = (distr_x["type"](**distr_x["kwargs"]), distr_y["type"](**distr_y["kwargs"]))

    random_idx = np.random.choice(np.arange(num_distr), p=coefficients)
    sample = data[random_idx]
    return sample


def save_samples(samples, filename):
    with open(filename, 'w') as f:
        for node in samples:
            f.write(str(node.long) + " " + str(node.lat) + "\n")
