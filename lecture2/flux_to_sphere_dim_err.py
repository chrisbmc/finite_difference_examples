##############################
####
#### 1D Simulation: Implicit
#### Flux to a Sphere
#### Dimensionless
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

#Space grid
h0 = 0.001                           # Smallest grid step

####
## Dimensionless variables
####
deltaT = 10
maxT = 10000
maxR = 1+6*np.sqrt(maxT)            # Maximum distance away from the electrode
k = int(maxT/deltaT)

#make X grid
h = h0
R = [1.0, 1.0 + h]                  # makes a list where the first value is 0.0
while R[-1] <= maxR:                # use a while loop to calculate all of the X positions, the loop stops when the value of x is larger than maxX
    R.append(R[-1]+h)               # every loop put the new value of x at the end of the list                     

n = len(R)                          # find the length (ie. the number of items) in the list X

##
# Setup concentration and tridiagonal arrays
##
C = np.ones(n)                      # make an empty (filled with zeros) numpy array that is n long

ab = np.zeros((3,n))                # make a 2D empty numpy array that is comprised of 3 arrays of n length
## ab is a numpy array that will be used to hold the finite difference coefficients (alpha, beta, gamma)
## we only need to save the diagonal values for the matrix so only need three arrays

#fill 2D array with the finite difference coefficients 
for i in range(1,n-1):              # use of for loop to go through the different grid positions but we don't do the first or last values   
    delR = R[i+1] - R[i]            # find the difference between the current grid position i and the next one (i+1)
    
    ab[0,i+1] = -(1)/ (delR**2) - (1)/(R[i]*(delR))     # gamma coefficeint
    ab[2,i-1] = -(1)/ (delR**2) + (1)/(R[i]*(delR))     # alpha coefficient
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

# calculate the analytically predicted flux- this is done using list comprehension
dim_cottrell = [1/(np.pi*t) for t in tau]          
dim_cottrell = [-1-ec**0.5 for ec in dim_cottrell]

######
## Plotting
#####
# plot the current (flux)
plt.loglog(tau,[-1*f for f in flux], color = 'r', label = 'Sim')     # [y*1e6 for y in flux] - this is list comprehension to muliply each value in a list by 1x10^6
plt.loglog(tau,[-1*ec for ec in dim_cottrell], color = 'k', label='Analytical')     # [y*1e6 for y in flux] - this is list comprehension to muliply each value in a list by 1x10^6
plt.xlabel('Time')            # add x-axis label
plt.ylabel('Flux')
plt.legend()
plt.savefig('flux_to_sphere_dim_err.png')
#plt.show() 

print(finish-start)     # simulation completion time

