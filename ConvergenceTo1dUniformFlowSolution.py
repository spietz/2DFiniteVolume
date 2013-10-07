 #################################################################################
 # ConvergenceTo1dUniformFlowSolution.py                                         #
 #   Script for computing convergence rates to exact solution to one-dimensional #
 #   convection-diffusion equations for uniform flow.                            #
 #   Implementation is thus limited to the following convective velocity field   #
 #   and boundary conditions:                                                    #
 #     1) uniform flow [u,v] = [1,0], homogeneous Neumann BC. at north and south #
 #     walls (dTn/dy=dTs/dy=0), homogeneous Dirichlet BC at west wall (Tw=0),    #
 #     inhomogeneous Dirichlet BC at east wall (Te=1).                           #
 #   System of eqs A*T=s solved directly using matlab's backslash operator.      #
 #                                                                               #
 # Input         :                                                               #
 #   N           :  Vector with number of cells along x,y-axis,                  #
 #                   e.g. N = [10, 25, 50, 100, 200]                             #
 #   L           :  Size of square in x,y-direction                              #
 #   Pe          :  Global Peclet number                                         #
 #   problem     :  Problem, selects case of convective field and BCs            #
 #   FVSCHEME    :  Array with finite volume schemes: ['cds-cds', 'uds-cds']     #
 #                                                                               #
 # Output        :                                                               #
 #   Plot of maximum truncation errors of the FV-solutions compared to the exact #
 #   solution w.r.t. to the number of cells n, incl. the rates of convergence    #
 #   for both uds-cds and cds-cds FV-schemes                                     #
 #################################################################################

import FVConvDiff2D
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve


# Exact solution (problem 1)
def Texact(Pe, x):
    return (np.exp(Pe*x)-1)/(np.exp(Pe)-1)

## Input
N = np.array([10, 25, 50, 100, 200, 400])  # number of cells along x,y-axis
L = 1.  # size of square in x,y-direction
Pe = 10.  # global Peclet numbers
problem = 1  # problem to solve (set =1)
FVSCHEME = ['cds-cds', 'uds-cds']  # finite volume schemes analyzed

## Initialize
ERROR = np.zeros((len(N), len(FVSCHEME)))  # max. trunc. error array
ROC = np.zeros(len(FVSCHEME))  # rate of conv. array
TL = np.zeros((len(N), len(FVSCHEME)))  # tendency lines array

## Compute max. truncation errors of FV-solutions for all N, PE, FVSCHEME
for i in range(0, len(N)):
    for j in range(0, len(FVSCHEME)):

        n = N[i]
        
        A, s = FVConvDiff2D.preprocess(n, L, Pe, problem, FVSCHEME[j])
        
        # Coordinate arrays
        dx = L/n  # cell size in x,y-direction
        xf = np.arange(0., L+dx, dx)  # cell face coords along x,y-axis
        xc = np.arange(dx/2., L, dx)  # cell center coordinates along x-axis
        Xc, _ = np.meshgrid(xc, xc)  # 2D cell center x-coordinate array

        T = spsolve(A, s.reshape(1, n**2, order='F'))  # direct solution

        # Error
        T_error = T-Texact(Pe, np.reshape(Xc, (1, n**2), order='F'))
        ERROR[i, j] = np.max(np.abs(T_error))  # infinity norm

## Compute rates of convergence
TL = np.zeros((len(N), len(FVSCHEME)))
p = np.zeros((2, len(FVSCHEME)))
for j in range(0, len(FVSCHEME)):
    p[j, :] = np.polyfit(np.log(N), np.log(ERROR[:, j]), 1, None, False, N**4)
    TL[:, j] = np.polyval(p[j, :], np.log(N))

## Plot convergence
plt.ion()  # turn on interactive mode

fig1 = plt.figure(1)
fig1.clf()
ax = fig1.add_subplot(1, 1, 1)
ax.plot(N, ERROR, 'x')
ax.plot(N, np.exp(TL), 'k')
ax.legend([FVSCHEME[0]+', slope=%0.3f' % (p[0, 0]),
           FVSCHEME[1]+', slope=%0.3f' % (p[1, 0])])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('N')
ax.set_ylabel('error')
ax.grid(True)

fig1.show()
