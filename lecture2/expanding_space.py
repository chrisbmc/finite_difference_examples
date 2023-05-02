##############################
####
#### 1D Simulation: Implicit
#### Flux to a Sphere
#### Space (exponential) grids
#### 
####
#############################

#import some useful functions
import numpy as np                  # provides a better way of handling numbers and arrays in python
import scipy.linalg as la           # sparse matrix solver
import matplotlib.pyplot as plt     # a basic plotting library
from datetime import datetime       # function to allow the simulation time to be reported
plt.close()                         # close any open plots 

start = datetime.now()              # store the current time

#######################
#### Physical Constants
#######################
F = 96485                           # Faraday Constant (C/mol)
R = 8.314                           # Gas Constant (J K-1 mol-1)
T = 298                             # Temperature

############################
#### Experimental parameters
############################
duration = 10 
D = 1e-9                            # Diffusion Coefficient (m2 s-1)
radius = 1e-6                       # Electrode radius (m)
conc = 1                            # Concentration of analyte (mol m-3)


#Expanding grid space
omega = 1.02                        # Expanding grid factor
h0 = 0.001                           # Smallest grid step

#Initial time and time step
delta_time = 1e-4

####
## Dimensionless variables
####
deltaT = D*delta_time/radius**2     # Dimensionless initial time step
maxT = D*duration/radius**2         # Duration of the experiment in dimensionless time
maxX = 1+6*np.sqrt(maxT)            # Maximum distance away from the electrode
k = int(maxT/deltaT)

#make X grid
h = h0
X = [1.0, 1.0 + h]                  # makes a list where the first value is 0.0
while X[-1] <= maxX:                # use a while loop to calculate all of the X positions, the loop stops when the value of x is larger than maxX
    X.append(X[-1]+h)               # every loop put the new value of x at the end of the list                     
    h *= omega                      # every loop multiply x by omega, shorthand for, h = h*omega

n = len(X)                          # find the length (ie. the number of items) in the list X

##
# Setup concentration and tridiagonal arrays
##
C = np.ones(n)                      # make an empty (filled with zeros) numpy array that is n long

ab = np.zeros((3,n))                # make a 2D empty numpy array that is comprised of 3 arrays of n length
## ab is a numpy array that will be used to hold the finite difference coefficients (alpha, beta, gamma)
## we only need to save the diagonal values for the matrix so only need three arrays

#fill 2D array with the finite difference coefficients 
for i in range(1,n-1):              # use of for loop to go through the different grid positions but we don't do the first or last values   
    delX_m = X[i] - X[i-1]          # find the difference between the current grid position i and the one before (i-1)
    delX_p = X[i+1] - X[i]          # find the difference between the current grid position i and the next one (i+1)
    
    ab[0,i+1] = -(2)/ (delX_p*(delX_m+delX_p)) - (2)/(X[i]*(delX_m+delX_p))     # gamma coefficeint
    ab[2,i-1] = -(2)/ (delX_m*(delX_m+delX_p)) + (2)/(X[i]*(delX_m+delX_p))     # alpha coefficient
    ab[1,i] =  + ab[0,i+1] + ab[2,i-1]                                          # beta coefficient
ab_mod = ab.copy()

#take the ab array and multiply it by the dimensionless time step
ab_mod[0,:] = deltaT*ab[0,:] 
ab_mod[2,:] = deltaT*ab[2,:] 
ab_mod[1,:] = (1-deltaT*ab[1,:]) 
ab_mod[1,n-1] = 1           # outer boundary 
ab_mod[2,n-2] = 0           # outer boundary
ab_mod[1,0] = 1     

## boundary conditions
C[0] = 0                 # set interfacial concentration

tau = [0.0]              # make an empty list to hold all of the potentials equiv to, pot = list()
flux = []                # make a list to hold all of the calculated dimensionless fluxes

## setup up the simulation loop where we solve the sparse matrix at for each time step
# solve linearised problem to obtain concentration profile for each time step

for i in range(k):          # for loop which goes from 0 to m-1                   
    #####
    ## Solve the banded Matrix
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html
    ####
    C = la.solve_banded((1,1),ab_mod,C)         # In the UV book the sparse matrix is solved using the Thomas algorithm- here we just use an imported function to do this for us
                                                # the function takes the finite difference matrix and the known concentrations and calculates the concentrations for the next time step and stores it back to the C array
    
    ## Calculate the flux using a simple two-point approximation
    flux.append(-(C[1] - C[0])/(h0))        
    ## Save the Time
    tau.append(tau[-1]+deltaT)


tau.pop(0)
#Having completed the simulation record the time
finish = datetime.now()

def dimFluxtoFlux(flux):                        # define a helper function to convert the calculated flux back into dimensional current
    temp = []
    for i in range(len(flux)):
        temp.append(flux[i]*(D*conc/radius))
    return temp

flux = dimFluxtoFlux(flux)
time = [(t*radius**2)/D for t in tau]

# calculate the analytically predicted flux- this is done using list comprehension
ext_cottrell = [radius**2/(np.pi*D*t) for t in time]          
ext_cottrell = [1+ec**0.5 for ec in ext_cottrell]
ext_cottrell = [-D*conc*ec/radius for ec in ext_cottrell]

######
## Plotting
#####
# plot the current (flux)
plt.loglog(time,[-1*f for f in flux], color = 'r', label = 'Sim')     # [y*1e6 for y in flux] - this is list comprehension to muliply each value in a list by 1x10^6
plt.loglog(time,[-1*ec for ec in ext_cottrell], color = 'k', label='Analytical')     # [y*1e6 for y in flux] - this is list comprehension to muliply each value in a list by 1x10^6
plt.xlabel('Time / s')            # add x-axis label
plt.ylabel('Flux/ mol m$^{-2}$ s$^{-1}$')
plt.legend()
plt.savefig('expanding_space.png')
#plt.show() 

print(finish-start)     # simulation completion time

