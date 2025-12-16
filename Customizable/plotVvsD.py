import matplotlib.pyplot as plt
from velocities import bubble_velocities
import numpy as np
experiments = ["T87_50V1", "T87_50V2", "T87_60V1", "T87_60V2", "T87_75V1", "T87_75V2", "T87_85V1", "T87_85V2", "T87_100V1", "T87_100V2"]
# Properties at 100°C
g = 9.81  # m/s²
mu_w = 0.0002814  # Pa.s
rho_w = 958.05  # kg/m³
rho_air = 0.946  # kg/m³

cd = 0.47 # cd pour une sphere 

dd = np.arange(0, 11e-3, 1e-4)
v_stokes =  (1/3) * (rho_w - rho_air) * g * (dd/2)**2 / mu_w
v_newton = np.sqrt( (4/3) * (rho_w - rho_air) * g * dd / (cd * rho_w) )

d_all = []
v_all = []

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
plt.figure()
for i, exp in enumerate(experiments):
    attach_vel, detach_vel = bubble_velocities(r"Inputs\T87_out", exp, minPointForVelocity=4)
    # print("Detach Velocities:")
    # for arr, m in zip(detach_vel.vy_mm, detach_vel.vMeanPerBlock_mm):
    #     print(f"Array: {arr} | Mean: {m}")
    print(exp)
    print(f"attach mean velocity: {attach_vel.vMean_mm} mm/s\tdetach mean velocity: {detach_vel.vMean_mm} mm/s")
    plt.scatter(detach_vel.diameterMeanPerBlock_mm, detach_vel.vMeanPerBlock_mm, color= colors[i], marker='.', label=exp)
    plt.scatter(detach_vel.diameterMean_mm, detach_vel.vMean_mm, color= colors[i], marker='+', label=exp+"mean", s=100)
    d_all.extend(list(detach_vel.diameterMeanPerBlock_mm))
    v_all.extend(list(detach_vel.vMeanPerBlock_mm))
    
# plt.plot(dd*1e3, v_stokes*1e3, color="r", label="Terminal Velocity (Stokes)")
plt.plot(dd*1e3, v_newton*1e3, color="g", label="Terminal Velocity (Newton)")

from scipy.optimize import curve_fit

def func(x, a):
    return np.sqrt( (4/3) * (rho_w - rho_air) * g * x / (a * rho_w) )

popt, pcov = curve_fit(func, np.array(d_all)*1e-3, np.array(v_all)*1e-3)
cd_fit = popt[0]
print(f"cd_fit : {cd_fit}")
v_fit = func(dd, cd_fit)
plt.plot(dd*1e3, v_fit*1e3, color="orange", label="Fitted Curve")


plt.xlabel("Diameter (mm)")
plt.ylabel("Elevation Velocity (mm/s)")
plt.legend()
plt.show()

##################################
plt.figure()
for i, exp in enumerate(experiments):
    attach_vel, detach_vel = bubble_velocities(r"Inputs\T87_out", exp, minPointForVelocity=4)
    print(exp)
    plt.scatter(attach_vel.diameterMeanPerBlock_mm, attach_vel.vMeanPerBlock_mm, color= colors[i], marker='.', label=exp)
    plt.scatter(attach_vel.diameterMean_mm, attach_vel.vMean_mm, color= colors[i], marker='+', label=exp+"mean", s=100)

plt.xlabel("Diameter (mm)")
plt.ylabel("Growing Velocity (mm/s)")
plt.legend()
plt.show()