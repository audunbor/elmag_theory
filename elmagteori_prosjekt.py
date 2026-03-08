import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def fourierApprox(N,n,V):
    """
    Approximate solution to Laplace equation on rectangular 2d grid for arbitrary upper boundary potential,
    by truncated Fourier series solution

    Args:
        N (int): Grid resolution.
        n (int): Number of terms in Fourier expansion.
        V (function): Potential upper boundary (y=1)

    Returns:
        ndarray: Numpy NxN meshgrid array containing solution
    """

    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    xv,yv = np.meshgrid(x,y)
    K_array = np.zeros(n)
    output = np.ndarray.copy(xv)
    output.fill(0)
    newout = np.zeros(N)
    for i in range(1,n+1):
        K_array[i-1]= 2*scipy.integrate.quad(lambda t: V(t)*np.sin(i*np.pi*t), 0, 1)[0]/np.sinh(i*np.pi)
        output += K_array[i-1]*np.sinh(np.pi*i*yv)*np.sin(i*np.pi*xv)
    return output

#Function used for debugging and testing
def funk0(x): 
    return 1

#Functions used as example potentials:
def funk1(x):
    return np.sin(3*np.pi*x)

def funk2(x):
    return 1 - (x-0.5)**4

def funk3(x):
    return np.heaviside(x-0.5,0.5)*np.heaviside(0.75-x,0.5)

#Generates plot of Fourier sum at upper boundary together with theoretical potential, 
#and quiverplot of derived electric field on entire grid, for three example potentials
outp = fourierApprox(100,180,funk1)
V=outp[-1]
x = np.linspace(0,1,len(V))
plt.plot(x,V,label="Approximation")
plt.plot(x,V)
y = np.zeros(len(x))
for i in range(len(y)):
    y[i]=funk1(x[i])
plt.plot(x,y,label="V_0")
plt.title("Sine-potential, m=3" )
plt.xlabel("xi")
plt.ylabel("V")
plt.plot(x,y)
plt.legend()
plt.show()

xv,yv=np.meshgrid(x,x)
E = np.gradient(outp, edge_order = 2)
plt.quiver(xv,yv,-E[0],-E[1], scale=3)
plt.title("E,sine potential, vector scale 1:3")
plt.xlabel("xi")
plt.ylabel("zeta")
plt.show()


outp = fourierApprox(100,180,funk2)
V=outp[-1]
x = np.linspace(0,1,len(V))
plt.plot(x,V,label="Approximation")
y = np.zeros(len(x))
for i in range(len(y)):
    y[i]=funk2(x[i])
plt.plot(x,y,label="V_0")
plt.title("Power four-potential" )
plt.xlabel("xi")
plt.ylabel("V")
plt.legend()
plt.show()

xv,yv=np.meshgrid(x,x)
E = np.gradient(outp, edge_order = 2)
plt.quiver(xv,yv,-E[0],-E[1], scale=7.5)
plt.title("E,power four-potential, vector scale 1:7.5")
plt.xlabel("xi")
plt.ylabel("zeta")
plt.ylim(0,1.3)
plt.show()

outp = fourierApprox(100,180,funk3)
V=outp[-1]
x = np.linspace(0,1,len(V))
plt.plot(x,V,label="Approximation")
y = np.zeros(len(x))
for i in range(len(y)):
    y[i]=funk3(x[i])
plt.plot(x,y,label="V_0")
plt.title("Heaviside-potential" )
plt.xlabel("xi")
plt.ylabel("V")
plt.legend()
plt.show()

xv,yv=np.meshgrid(x,x)
print(xv)
E = np.gradient(outp, edge_order = 2)
plt.quiver(xv,yv,-E[0],-E[1], scale = 5)
plt.title("E, Heaviside-potential, vector scale 1:5")
plt.xlabel("xi")
plt.ylabel("zeta")
plt.ylim(0,1.2)
plt.show()


outp = fourierApprox(100,100,funk2)
V=outp[-1]

m= 200

#Generates error over entire grid in L1 norm as a function of number of terms in Fourier series,
#for three example potentials
errors = np.zeros(m)
ns = np.zeros(m)
y = np.zeros(len(x))
for i in range(len(y)):
    y[i]=funk1(x[i])
y=y[1:-1]
for i in range(m):
    outp = fourierApprox(100,i,funk1)
    V=outp[-1]
    V=V[1:-1]
    errors[i]=np.linalg.norm(V-y,1)
    if(errors[i]<0.1):
        print(i)
    ns[i] = i
ns=ns
errors
plt.plot(ns,np.log(errors))
plt.title("Logarithm of errors, sine,m=3")
plt.xlabel("Number of terms in series")
plt.ylabel("Logarithm of error")
plt.show()

errors = np.zeros(m)
ns = np.zeros(m)
y = np.zeros(len(x))
for i in range(len(y)):
    y[i]=funk2(x[i])
y=y[1:-1]
for i in range(m):
    outp = fourierApprox(100,i,funk2)
    V=outp[-1]
    V=V[1:-1]
    errors[i]=np.linalg.norm(V-y,1)
    if(errors[i]<0.1):
        print(i)
    ns[i] = i
ns=ns
errors
plt.plot(ns,np.log(errors))
plt.title("Logarithm of errors, power four")
plt.xlabel("Number of terms in series")
plt.ylabel("Logarithm of error")
plt.show()

errors = np.zeros(m)
ns = np.zeros(m)
y = np.zeros(len(x))
for i in range(len(y)):
    y[i]=funk3(x[i])
y=y[1:-1]
for i in range(m):
    outp = fourierApprox(100,i,funk3)
    V=outp[-1]
    V=V[1:-1]
    errors[i]=np.linalg.norm(V-y,1)
    if(errors[i]<0.1):
        print(i)
    ns[i] = i
ns=ns
errors
plt.plot(ns,np.log(errors))
plt.title("Logarithm of errors, heaviside")
plt.xlabel("Number of terms in series")
plt.ylabel("Logarithm of error")
plt.show()
