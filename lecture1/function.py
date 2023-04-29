F = 96485 # C mol-1
T = 298 # K
R = 8.314 # J K-1 mol-1

def etotheta(E, Ef):
    theta = F*(E-Ef)/(R*T)
    return theta
