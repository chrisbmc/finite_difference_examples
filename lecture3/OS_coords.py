import numpy as np
import scipy.linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from datetime import datetime
plt.close()
start = datetime.now()
F = 96485
R = 8.314
Temp = 298

start_v = 0
vertex_v = -1
sr = 0.01
radius = 5E-3
Ef= -0.5
k0 = 0.1
diff_coeff = 1E-9
conc = 1

ngrid_mu = 100
ngrid_nu = 60

def v_to_theta(V,Ef):
    return (V-Ef)*F/(R*Temp)
theta_i = v_to_theta(start_v,Ef)
theta_v = v_to_theta(vertex_v,Ef)
sigma = radius**2*F*sr/(diff_coeff*R*Temp)

deltaTheta = float(0.04)
K0 = k0*radius/diff_coeff
alpha = 0.5

deltaT = deltaTheta/sigma
maxT = 2*abs(theta_v-theta_i)/sigma

maxmu = np.arccosh(1+6*np.sqrt(maxT))

t = int(maxT/deltaT)

mu_pos = np.linspace(0,maxmu,ngrid_mu)
nu_pos = np.linspace(0,np.pi/2,ngrid_nu)

deltamu = mu_pos[1]
deltanu = nu_pos[1]

conc_grid = np.ones(shape = (ngrid_mu,ngrid_nu))
coeff_grid_vals = np.zeros(shape = (ngrid_mu,ngrid_nu,5))

for i in range(1,ngrid_mu-1):
    for j in range(1,ngrid_nu-1):
        mu = mu_pos[i]
        nu = nu_pos[j]
        
        s = -deltaT/((np.sinh(mu)**2+np.sin(nu)**2))
        coeff_grid_vals[i,j][0] = s*np.tanh(mu)/(2*deltamu) + s/deltamu**2
        coeff_grid_vals[i,j][2] = 2*s/deltamu**2 - coeff_grid_vals[i,j][0]
        coeff_grid_vals[i,j][3] = s*np.tan(nu)/(2*deltanu) + s/deltanu**2
        coeff_grid_vals[i,j][4] = 2*s/deltanu**2 - coeff_grid_vals[i,j][3]
        coeff_grid_vals[i,j][1] = 1 - coeff_grid_vals[i,j][0] - coeff_grid_vals[i,j][2]\
                                    - coeff_grid_vals[i,j][3] - coeff_grid_vals[i,j][4]

row = []
column = []
data = []

for i in range(1,ngrid_mu-1):
    for j in range(1,ngrid_nu-1):
        row.append(i*ngrid_nu + j)
        column.append((i+1)*ngrid_nu + j)
        data.append(coeff_grid_vals[i,j][0])
        
        row.append(i*ngrid_nu + j)
        column.append(i*ngrid_nu + j)
        data.append(coeff_grid_vals[i,j][1])
     
        row.append(i*ngrid_nu + j)   
        column.append((i-1)*ngrid_nu + j)
        data.append(coeff_grid_vals[i,j][2])        
        
        row.append(i*ngrid_nu + j)
        column.append(i*ngrid_nu + j+1)
        data.append(coeff_grid_vals[i,j][3])  
        
        row.append(i*ngrid_nu + j)
        column.append(i*ngrid_nu + j-1)
        data.append(coeff_grid_vals[i,j][4]) 
        
for i in range(1,ngrid_mu-1):
    s = 1/(deltanu*np.sqrt(1+np.sinh(mu_pos[i])**2))
    row.append(i*ngrid_nu+ngrid_nu-1)
    column.append(i*ngrid_nu+ngrid_nu-1)
    data.append(s)
    
    row.append(i*ngrid_nu+ngrid_nu-1)
    column.append(i*ngrid_nu+ngrid_nu-2)
    data.append(-s)

for j in range(ngrid_nu):
    row.append((ngrid_mu-1)*ngrid_nu +j)
    column.append((ngrid_mu-1)*ngrid_nu +j)
    data.append(1)
    
for i in range(1,ngrid_mu-1):
    s =  1/(deltanu*np.sinh(mu_pos[i]))
    row.append(i*ngrid_nu)
    column.append(i*ngrid_nu)
    data.append(-s)
    
    row.append(i*ngrid_nu)
    column.append(i*ngrid_nu+1)
    data.append(s)
        
A = csc_matrix((data, (row, column)), shape=(ngrid_mu*ngrid_nu,ngrid_mu*ngrid_nu))
print(np.shape(A))   
A[0,0] = 1
A[0,ngrid_nu] = -1
b = conc_grid.reshape((ngrid_mu*ngrid_nu))
print(np.shape(b))
print(np.shape(A))       

Theta = theta_i
flux = list()
pot = list()
fsum = 0
for k in range(0,t):
    print(k/t)
    if (k < t/2):
        Theta -= deltaTheta
    else:
        Theta += deltaTheta
    fTheta = np.exp(-alpha*Theta)

    b[0] = 0
    b[(ngrid_mu-1)*ngrid_nu] = 1
    for j in range(1,ngrid_nu):
        s = 1/(np.sin(nu_pos[j]))
        
        A[j,j] = 1+ (deltamu/s)*K0*fTheta*(1+np.exp(Theta))
        A[j,j+ngrid_nu] = -1
        b[j] = (deltamu/s)*K0*fTheta*np.exp(Theta)
        b[(ngrid_mu-1)*ngrid_nu +j] = 1
    for i in range(1,ngrid_mu-1):
        b[i*ngrid_nu+ngrid_nu-1] = 0
        b[i*ngrid_nu] = 0
    x = spsolve(A,b)
    pot.append(Theta)
    fsum = 0
    for j in range(ngrid_nu-1):
        if j == 0:
            f1 = (x[ngrid_nu+j]-x[j])*np.cos(nu_pos[j])/(deltamu)
        else:
            f1 = (x[ngrid_nu+j]-x[j])*np.cos(nu_pos[j])/(deltamu)
        f2 = (x[ngrid_nu+j+1]-x[j+1])*np.cos(nu_pos[j+1])/(deltamu)#f2 = (x[ngrid_nu+j+1]-x[j+1])/(2*np.tan(nu_pos[j+1])*deltamus[0,0])
        fsum += (f1)*(nu_pos[j+1]-nu_pos[j])
    flux.append(fsum)
    # if k == t//2:
    #     cp= x.reshape( (ngrid_mu,ngrid_nu)).T
    #     plt.imshow(cp)
    #     plt.show() 
    b = x
finish = datetime.now()
print(finish-start)

aa = [(diff_coeff*4*np.exp(alpha*i))/(k0*radius*np.pi) + (1+np.exp(i)) for i in pot]
aa = [-4*F*diff_coeff*conc*radius/i for i in aa]
F2 = [1- (1+np.exp(i))/((3*np.pi**2/2)*np.exp(alpha*i)/((k0*radius/diff_coeff)*(np.pi/4)) + 2*(1+np.exp(i))) for i in pot]
ff = [(k0*radius/diff_coeff)*(np.pi/4)*np.exp(-alpha*i) for i in pot]
oldham = list()
for i in range(len(pot)):
    oldham.append((-4*F*diff_coeff*conc*radius)*ff[i]/(F2[i]+ff[i]))

flux = [-2*np.pi*F*diff_coeff*conc*radius*J for J in flux]
pot = [v*R*Temp/F +Ef for v in pot]

plt.plot(pot,flux) 
plt.plot(pot,aa)
plt.plot(pot,oldham) 

ip = 0.446*F*np.pi*radius**2*conc*(F*diff_coeff*sr/(R*Temp))**0.5
ipirrev = 0.496*alpha**0.5*F*np.pi*radius**2*conc*(F*diff_coeff*sr/(R*Temp))**0.5
iss = 4*F*diff_coeff*conc*radius
rspot = [start_v,vertex_v]
rsip = [-ip,-ip]
rsirrev = [-ipirrev, -ipirrev] 
riss = [-iss,-iss]  

plt.plot(rspot,rsip, color = 'Green')
plt.plot(rspot,rsirrev, color = 'Blue')
plt.plot(rspot,riss, color = 'Orange')
plt.savefig('basic.png')
#plt.show()    
  
        
        
        