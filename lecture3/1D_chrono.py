##############################
####
#### 1D Voltammetry Simulation: Implicit
#### BV kinetics= 1 electron oxidation
####
#############################
# See 'UV: Simultion of Electrode Processes' 2nd Ed. pg 60- for similar simulation C++
# main differences- this implements BV kintics (not Nernst) and has an non-uniform expanding grid

#import some useful functions
import math
import numpy as np                  # provides a better way of handling numbers and arrays in python
import scipy.linalg as la           # sparse matrix solver
import matplotlib.pyplot as plt     # a basic plotting library
from datetime import datetime       # function to allow the simulation time to be reported
import seaborn as sns
import imageio
import os

plt.close()                         # close any open plots 

start = datetime.now()              # store the current time

#######################
#### Physical Constants
#######################
F = 96485                           # Faraday Constant (C/mol)
R = 8.314                           # Gas Constant (J K-1 mol-1)
Temp = 298                             # Temperature

############################
#### Experimental parameters
############################
Epot = 0.3                       # Start Potential for the Voltammogram (V)
duration = 60*180
step = 1E-5
expanding = 1E-6

D = 1.4e-10                            # Diffusion Coefficient (m2 s-1)
radius = 1e-3                       # Electrode radius (m)
k0 = 0.01                           # Electron transfer rate (m s-1)
conc = 1                            # Concentration of analyte (mol m-3)
alpha = 0.5                         # Transfer Coefficient (dimensionless)
layert = 1E-3                       #layer thickness if finite
layerT = layert/radius


##Expanding time grid
t = [0.0]
while t[-1] < duration:
    t.append(t[-1]+step)
    step += expanding

####
## Dimensionless variables
####
# Define two helper functions to Convert the potential (E) to the Dimensionless Potential (Theta)
def etotheta(E, Ef):
    return F*(E-Ef)/(R*Temp)

# does the reverse- converting the dimensionless potential back
def thetatoe(theta, Ef):                    # the conversion method is slightly different depending on whether a python list or a numpy array is passed to the function
    if type(theta) == list:                 # first check if a list has been sent to the function               
        temp = []                           # if True make an empty list
        for i in range(len(theta)):         # loop through the list and calculate the potential
            temp.append( (theta[i]*(R*Temp)/F)+Ef )
        return temp
    else:
        return (theta*(R*Temp)/F)+Ef           # nomenclature is simpler if a numpy array is passed

def ttoT(t, D, r):
    temp = []
    for i in range(len(t)):
        temp.append( D*t[i]/r**2 )
    return temp
    

K0 = (k0*radius)/D                          # Dimensionless electron transfer rate
T = ttoT(t,D,radius)
a = T[1:] 
b = T[0:-1]
deltaT = []
for i in range(len(a)):
    deltaT.append(a[i]-b[i])
T = T[1:]
t = t[1:]

maxT = T[-1] #2*abs(theta_v-theta_i)/sigma         # Duration of the experiment in dimensionless time
maxX = layerT #10*math.sqrt(maxT)   #layerT                 # Maximum distance away from the electrode

#Expanding grid
omega = 1.02                                # Expanding grid factor
h0 = 1e-4                                   # Smallest grid step

#make X grid
h = h0
x = 0.0                             # set the variable x to have the value 0.0
X = [0.0]                           # makes a list where the first value is 0.0
while x <= maxX:                    # use a while loop to calculate all of the X positions, the loop stops when the value of x is larger than maxX
    x += h                          # shorthand for, x = x + h, every loop add h to the present value of x
    X.append(x)                     # every loop put the new value of x at the end of the list                     
    h *= omega                      # every loop multiply x by omega, shorthand for, h = h*omega
X.pop()
X.append(maxX)
n = len(X)                          # find the length (ie. the number of items) in the list X
m = len(T)               # find the number of time steps and make sure you have an integer (int)



##
# Setup concentration and tridiagonal arrays
##
C = np.ones(n)                      # make an empty (filled with zeros) numpy array that is n long

ab = np.zeros((3,n))                # make a 2D empty numpy array that is comprised of 3 arrays of n length
abT = np.zeros((3,n))                # make a 2D empty numpy array that is comprised of 3 arrays of n length
## ab is a numpy array that will be used to hold the finite difference coefficients (alpha, beta, gamma)
## we only need to save the diagonal values for the matrix so only need three arrays

#fill 2D array with the finite difference coefficients 
for i in range(1,n-1):              # use of for loop to go through the different grid positions but we don't do the first or last values   
    delX_m = X[i] - X[i-1]          # find the difference between the current grid position i and the one before (i-1)
    delX_p = X[i+1] - X[i]          # find the difference between the current grid position i and the next one (i+1)
    
    ab[0,i+1] = -(2)/ (delX_p*(delX_m+delX_p))   # gamma coefficient
    ab[2,i-1] = -(2)/ (delX_m*(delX_m+delX_p))   # alpha coefficient
    ab[1,i] = - ab[0,i+1] - ab[2,i-1]                 # beta coefficient

Theta = etotheta(Epot,0.0)         # Set the Potential to the start potential
flux = []               # make a list to hold all of the calculated dimensionless fluxes

## setup up the simulation loop where we solve the sparse matrix at for each time step
# solve linearised problem to obtain concentration profile for each time step
Concs = []
Cs= []
for k in range(0,m,1):          # for loop which goes from 0 to m-1
    abT[0,:] = ab[0,:]*deltaT[k]
    abT[2,:] = ab[2,:]*deltaT[k]
    abT[1,:] = 1 + ab[1,:]*deltaT[k]

    ## boundary conditions
    abT[0,1] = -1            # inner (electrode) boundary              
    abT[1,n-1] = 1           # outer boundary 
    abT[2,n-2] = -1           # outer boundary

    C[-1] = 0
    fTheta = math.exp(alpha*Theta)                       # calculate one of the exponential terms in the BV equation
    abT[1,0] = 1 + h0*fTheta*K0*(1+math.exp(-Theta))      # Set the Butler-Volmer boundary condition- this value changes for each time step
    C[0] = h0*fTheta*K0*math.exp(-Theta)                 # Set the surface concentration of the species
    
    #####
    ## Solve the banded Matrix
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html
    ####
    C = la.solve_banded((1,1),abT,C)         # In the UV book the sparse matrix is solved using the Thomas algorithm- here we just use an imported function to do this for us
                                            # the function takes the finite difference matrix and the known concentrations and calculates the concentrations for the next time step and stores it back to the C array
    ##
    ## Calculate the flux using a simple two-point approximation
    flux.append((C[1] - C[0])/(h0))    
    Cs.append(C)    
    X_new = np.linspace(X[0],X[-1],1000)
    C_new = np.interp(X_new,X,C)
    Concs.append([C_new for i in range(10) ])

#Having completed the simulation record the time
finish = datetime.now()

def fluxtocurrent(flux):                # define a helper function to convert the calculated flux back into dimensional current
    temp = []
    for i in range(len(flux)):
        temp.append(flux[i]*(np.pi*radius*F*D*conc))
    return temp
flux = fluxtocurrent(flux)

######
## Plotting
#####
area = np.pi*radius**2
cottrell = []
for i in range(len(t)):
    cottrell.append( (F*conc*D**0.5)/(np.pi*t[i])**0.5 )
# plot the current (flux)

plt.loglog(t[60:],[y/area for y in flux][60:])     # [y*1e6 for y in flux] - this is list comprehension to multiply each value in a list by 1x10^6
plt.loglog(t,[y for y in cottrell])
plt.xlabel('Time / s')            # add x-axis label
plt.ylabel('Current density / A m$^{-2}$')
#plt.legend(loc = 2)
plt.show() 

# output some basic information about the simulation
print(finish-start)     # simulation completion time

error = []
for i in range(len(cottrell)):
    error.append(flux[i]/cottrell[i])

def my_func(i):
    sns.heatmap(Concs[i],
                yticklabels=False,
                xticklabels=False,
                cbar = True,
                vmin = 0,
                vmax = 1)
    plt.xlabel('Distance from Electrode / mm')
    plt.text(0,0,'Time= %s min' %np.round(t[i]/60))
    plt.text(0,10.5,'0.0')
    plt.text(980,10.5,'1.0')
    plt.text(1200,5.7,'Conc /M', rotation = 90)

filenames = []

percent = []
for i in range(len(Cs)):
    percent.append(1-np.trapz(Cs[i],X))

plt.semilogx(t,percent)
plt.xlabel('Time / s')
plt.ylabel('Fraction Oxidised')
plt.show()

y = range(len(t))
y = y[::1000]
for i in y:
    # plot the line chart
    my_func(i)
    
    # create file name and append it to a list
    filename = f'{i}.png'
    filenames.append(filename)
    
    # save frame
    plt.savefig(filename)
    plt.close()
# build gif
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)
#anim = FuncAnimation(fig = fig, func = my_func, frames = 34632, interval = 50, blit = False)
#writergif = PillowWriter(fps=30)
#anim.save('thin_layer.gif',writer=writergif)

#plt.show()
#X_new = np.linspace(X[0],X[-1],100)
#C_new = np.interp(x_new,X,C)
