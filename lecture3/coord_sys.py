import numpy as np
import matplotlib.pyplot as plt
plt.close()
mu = np.linspace(0,3,20)
nu = np.linspace(0,np.pi/2,20)

nu_lin = np.linspace(0,np.pi/2,40)
mu_array = np.ndarray(shape=(len(mu),2,len(nu_lin)))
count = 0
for mu_val in mu:
     mu_line_x = [np.cosh(mu_val)*np.cos(nu_val) for nu_val in nu_lin]
     mu_line_z = [np.sinh(mu_val)*np.sin(nu_val) for nu_val in nu_lin]
     mu_array[count][0] = mu_line_x# list(zip(mu_line_x,mu_line_z))
     mu_array[count][1] = mu_line_z
     count +=1
     
mu_lin = np.linspace(0,3,40)
nu_array = np.ndarray(shape=(len(nu),2,len(mu_lin)))
count = 0
for nu_val in nu:
     nu_line_x = [np.cosh(mu_val)*np.cos(nu_val) for mu_val in mu_lin]
     nu_line_z = [np.sinh(mu_val)*np.sin(nu_val) for mu_val in mu_lin]
     nu_array[count][0] = nu_line_x
     nu_array[count][1] = nu_line_z
     count +=1

for i in range(len(mu)):
    plt.plot(mu_array[i][0],mu_array[i][1], color='blue')

for i in range(len(nu)):
    plt.plot(nu_array[i][0],nu_array[i][1], color='red')

plt.show()    
     
# t = np.arange(-5,5,0.001)
# x = 10**t
# x1 = np.log(t)/np.log(10)
# 
# plt.plot(t,x,color="blue",label="[R]")
# plt.plot(t,x1,color="red",label="[P]")
# plt.xlabel("Time /s")
# plt.ylabel("Concentration /mol dm$^{-3}$")
# plt.title(r"R$\rightarrow$P")
# plt.grid(True, color="#93a1a1", alpha=0.3)
# plt.rc('axes', titlesize=22)
# plt.rc('axes', labelsize=12) 
# #plt.xlim((0,5))
# plt.ylim((-5,5))
# plt.legend(loc=5, prop={'size': 12})
# plt.show()