import math
import matplotlib.pyplot as plt

# ellipse major (a) and minor (b) axis parameters
a=8
b=2

# num points for transformation lookup function
npoints = 10000
delta_theta=2.0*math.pi/npoints

theta=[0.0]
delta_s=[0.0]
integ_delta_s=[0.0]

# integrated probability density
integ_delta_s_val=0.0

for iTheta in range(1,npoints+1):
    # ds/d(theta):
    delta_s_val=math.sqrt(a**2*math.sin(iTheta*delta_theta)**2+ \
                          b**2*math.cos(iTheta*delta_theta)**2)

    theta.append(iTheta*delta_theta)
    delta_s.append(delta_s_val)
    # do integral
    integ_delta_s_val = integ_delta_s_val+delta_s_val*delta_theta
    integ_delta_s.append(integ_delta_s_val)
    
# normalize integrated ds/d(theta) to make into a scaled CDF (scaled to 2*pi)
integ_delta_s_norm = []
for iEntry in integ_delta_s:
    integ_delta_s_norm.append(iEntry/integ_delta_s[-1]*2.0*math.pi)    
    
#print('theta= ', theta)
#print('delta_theta = ', delta_theta)
#print('delta_s= ', delta_s)
#print('integ_delta_s= ', integ_delta_s)
#print('integ_delta_s_norm= ', integ_delta_s_norm)

# Plot tranformation function
x_axis_range=1.5*math.pi
y_axis_range=1.5*math.pi
plt.xlim(-0.2, x_axis_range)
plt.ylim(-0.2, y_axis_range)
plt.plot(theta,integ_delta_s_norm,'+')
# overplot reference line which are the theta values.
plt.plot(theta,theta,'.')
plt.show()

# Reference ellipse without correction.
ellip_x=[]
ellip_y=[]

# Create corrected ellipse using lookup function
ellip_x_prime=[]
ellip_y_prime=[]

npoints_new=100
delta_theta_new=2*math.pi/npoints_new

for theta_index in range(npoints_new):
    theta_val = theta_index*delta_theta_new
#    print('theta_val = ', theta_val)
    
# Do lookup:
    for lookup_index in range(len(integ_delta_s_norm)):
#        print('doing lookup: ', lookup_index)
#        print('integ_delta_s_norm[lookup_index]= ', integ_delta_s_norm[lookup_index])
        if theta_val >= integ_delta_s_norm[lookup_index] and theta_val < integ_delta_s_norm[lookup_index+1]:
#            print('value found in lookup table')
            theta_prime=theta[lookup_index]
#            print('theta_prime = ', theta_prime)
#            print('---')
            break
    
    # ellipse without transformation applied for reference
    ellip_x.append(a*math.cos(theta_val))
    ellip_y.append(b*math.sin(theta_val))
    
    # ellipse with transformation applied
    ellip_x_prime.append(a*math.cos(theta_prime))
    ellip_y_prime.append(b*math.sin(theta_prime))

# Plot reference and transformed ellipses  
x_axis_range=10
y_axis_range=10
plt.xlim(-x_axis_range, x_axis_range)
plt.ylim(-y_axis_range, y_axis_range)
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(ellip_x, ellip_y, '.')
plt.title('Uncorrected')
plt.show()

plt.xlim(-x_axis_range, x_axis_range)
plt.ylim(-y_axis_range, y_axis_range)
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(ellip_x_prime, ellip_y_prime, '.')
plt.title('Corrected')
plt.show()