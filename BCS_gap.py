from scipy.optimize import curve_fit
from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

#set Tc here
Tc=7

def gap0(x):
    return np.tanh(1/(2*Tc)*x)/x
couple=integrate.quad(gap0,0,1)

#set temperature range here: (start, end, number of points)
list_linear_T = np.linspace(0.01,10,500)
array_T = np.array(list_linear_T)

list_gap=[]
for num in array_T:
    def f(d):
        def gap(x):
           return np.tanh(1/(2*num)*np.sqrt(x**2+d**2))/np.sqrt(x**2+d**2)
              
        fxs=integrate.quad(gap,0,1)
        residual=couple[0] - fxs[0]
        return residual

#set convergence here   
    p=optimize.fsolve(f,0.001)
    list_gap.append(p[0])
 
gap2=np.array(list_gap) / list_gap[0]
 
data = np.c_[array_T, gap2]

#If you want to see the numbers
#print(data)

#If you want to save 
#np.savetxt('out.csv',data,delimiter=',')

plt.plot(array_T, gap2)
plt.xlabel('Temperature(K)')
plt.ylabel('Gap (arb units)')
plt.show()