##############################
####
#### 1D Voltammetry Simulation: Implicit
#### Nernst: 1 electron oxidation- Thin layer
####
#############################
# See 'UV: Simulation of Electrode Processes' 2nd Ed. pg 60- for similar simulation C++
# main differences- this implements a Nernst Boundary Condition and has an non-uniform expanding grid

#import some useful functions
import math
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
Ef = 0.0                            # Formal Potential (V)
Estart = -0.3                       # Start Potential for the Voltammogram (V)
Evertex = 0.3                       # Potential to which the Electrode is scanned (V)
scanrate = 0.1                      # Scan rate (V s-1)
stepsize = 0.2e-3                   # Potential step (V)

D = 1e-9                            # Diffusion Coefficient (m2 s-1)
radius = 1e-3                       # Electrode radius (m)
conc = 1                            # Concentration of analyte (mol m-3)
layer = 1e-6                        # Position of the outer no-flux boundary

#####
## Randles-Sevich Equaion
####
iprs = 2.69e5*D**0.5*conc*np.pi*radius**2*scanrate**0.5                 # Reversible form
#ipirrev = 2.99e5*D**0.5*conc*np.pi*radius**2*alpha**0.5*scanrate**0.5   # Irreversible Form

####
## Dimensionless variables
####
# Define two helper functions to Convert the potential (E) to the Dimensionless Potential (Theta)
def etotheta(E, Ef):
    return F*(E-Ef)/(R*T)

# does the reverse- converting the dimensionless potential back
def thetatoe(theta, Ef):                    # the conversion method is slightly different depending on whether a python list or a numpy array is passed to the function
    if type(theta) == list:                 # first check if a list has been sent to the function               
        temp = []                           # if True make an empty list
        for i in range(len(theta)):         # loop through the list and calculate the potential
            temp.append( (theta[i]*(R*T)/F)+Ef )
        return temp
    else:
        return (theta*(R*T)/F)+Ef           # nomenclature is simpler if a numpy array is passed
    
theta_i = etotheta(Estart, Ef)              # Convert the starting potential to theta
theta_v = etotheta(Evertex, Ef)     
sigma = (radius**2/D)*(F/(R*T))*scanrate    # Convert the scan rate to the dimensionless form (sigma) 

deltaTheta = etotheta(stepsize,0.0)         # Dimensionless step size

deltaT = deltaTheta/sigma                   # Dimensionless time step
maxT = 2*abs(theta_v-theta_i)/sigma         # Duration of the experiment in dimensionless time
maxX = layer/radius                         # Maximum distance away from the electrode

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

n = len(X)                          # find the length (ie. the number of items) in the list X
m = int(maxT/deltaT)                # find the number of time steps and make sure you have an integer (int)

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
    
    ab[0,i+1] = -(2*deltaT)/ (delX_p*(delX_m+delX_p))   # gamma coefficient
    ab[2,i-1] = -(2*deltaT)/ (delX_m*(delX_m+delX_p))   # alpha coefficient
    ab[1,i] = 1 - ab[0,i+1] - ab[2,i-1]                 # beta coefficient

## boundary conditions
ab[0,1] = 0            # inner (electrode) boundary              
ab[1,n-1] = 1           # outer boundary 
ab[2,n-2] = -1           # outer boundary


Theta = theta_i         # Set the Potential to the start potential
pot = []                # make an empty list to hold all of the potentials equiv to, pot = list()
flux = []               # make a list to hold all of the calculated dimensionless fluxes

## setup up the simulation loop where we solve the sparse matrix at for each time step
# solve linearised problem to obtain concentration profile for each time step
for k in range(0,m,1):          # for loop which goes from 0 to m-1
    if(k<(m/2)):                # use logic to determine if we are on the forwar scan
        Theta += deltaTheta     # change the potential on each timestep, equivalent to Theta = Theta - deltaTheta
    else:                       # if else cause- so that when on the reverse scan we add to the potential as opposed to subtracting
        Theta -= deltaTheta     
    
    ab[1,0] = 1 + math.exp(Theta)                       # Nernst Boundary Condition
    C[0] = 1
    C[-1] = 0
    
    #####
    ## Solve the banded Matrix
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html
    ####
    C = la.solve_banded((1,1),ab,C)         # In the UV book the sparse matrix is solved using the Thomas algorithm- here we just use an imported function to do this for us
                                            # the function takes the finite difference matrix and the known concentrations and calculates the concentrations for the next time step and stores it back to the C array
    ##
    ## Calculate the flux using a simple two-point approximation
    flux.append((C[1] - C[0])/(h0))        
    ## Save the Potential
    pot.append(Theta)

#Having completed the simulation record the time
finish = datetime.now()

pot = thetatoe(pot,Ef)                  # convert the dimensionless potentials back to a dimensional form

def fluxtocurrent(flux):                # define a helper function to convert the calculated flux back into dimensional current
    temp = []
    for i in range(len(flux)):
        temp.append(flux[i]*(np.pi*radius*F*D*conc))
    return temp
flux = fluxtocurrent(flux)

######
## Plotting
#####
#make data to plot the Randles-Sevich current as a line
rspot = [Estart,Evertex]
rsip = [iprs,iprs]   
# plot the expected current from the equation- make the line green and give is a lable
#plt.plot(rspot,[y*1e6 for y in rsip], color = 'g', label = 'D = %s m2 s-1' %(D))

# plot the current (flux)
plt.plot(pot,[y*1e6 for y in flux])     # [y*1e6 for y in flux] - this is list comprehension to multiply each value in a list by 1x10^6
plt.xlabel('Potential / V)')            # add x-axis label
plt.ylabel('Current / uA')
plt.legend(loc = 2)
plt.savefig("10_1D_thin_layer.png")
plt.show() 

# output some basic information about the simulation
print('Error on Ip')
print(max(flux)/iprs)
print(finish-start)