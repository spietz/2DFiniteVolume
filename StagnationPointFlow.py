 #####################################################################################
 # StagnationPointFlow.py                                                            #
 #   Script for simulation of steady convection-diffusion heat problem on two-       #
 #   dimensional square domain using the Finite Volume Method on a cartesian,        #
 #   structured mesh with square cells. Central difference fluxes applied for        #
 #   the diffusive terms, and either central of upwinded difference fluxes           #
 #   applied for the convective terms.                                               #
 #   Implementation does not include source terms and is limited to the              #
 #   following sets of convective velocity fields and boundary conditions:           #
 #     (problem 1) see UniformFlow.py                                                #
 #     (problem 2) stagnation point flow [u,v] = [-x,y], homogeneous Neumann BC.     #
 #     at north wall (dTn/dy=0), inhomogeneous Dirichlet BC at west wall (Tw=0),     #
 #     inhomogeneous Dirichlet BC at east wall (Te=1), inhomogeneous Dirichlet BC    #
 #     at south wall (Ts=x).                                                         #
 #   Linear system of equations solved either directly using Matlab's backslash      #
 #   operator, or iteratively using either Jacobi, Gauss-Seidel or SOR stationary    #
 #   iterative methods.                                                              #
 #                                                                                   #
 # Input         :                                                                   #
 #   n           :  Number of cells along x,y-axis                                   #
 #   L           :  Size of square in x,y-direction                                  #
 #   Pe          :  Global Peclet number                                             #
 #   problem     :  Problem #: 1 or 2, selects case of convective field and BCs      #
 #   fvscheme    :  Finite volume scheme for convection-diffusion,                   #
 #                  either 'cds-cds' or 'uds-cds'                                    #
 #                                                                                   #
 # Output        :                                                                   #
 #   T           :   Temperature at cell nodes, T(1:n,1:n)                           #
 #   A           :   Convection-diffusion system matrix, A(1:n^2,1:n^2)              #
 #   s           :   Source array with BC contributions, s(1:n,1:n)                  #
 #   TT          :   Temperature field extrapolated to walls, TT(1:n+2,1:n+2)        #
 #   CF,DF       :   Conv. and diff. fluxes through walls, CF=[CFw,CFe,CFs,CFn]      #
 #   GHC         :   Global heat conservation, scalar (computed from wall fluxes)    #
 #   Plots of the temperature field and convective velocity field                    #
 #####################################################################################

import FVConvDiff2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse.linalg import spsolve

## Input
n = 100  # number of cells along x,y-axis
L = 1.0  # size of square in x,y-direction
Pe = 10.0  # global Peclet number
problem = 2  # problem to solve
fvscheme = 'uds-cds'  # finite volume scheme ('cds-cds' or 'uds-cds')

## Assemble system matrix
A, s = FVConvDiff2D.preprocess(n, L, Pe, problem, fvscheme)

## Do direct solution
T = spsolve(A, s.reshape(1, n**2, order='F')).reshape(n,n, order='F')

## Extend T-field to domain walls and get GHC-residual
TT, GHC, _, _ = FVConvDiff2D.postprocess(
    T, n, L, Pe, problem, fvscheme)

## Plot solution and streamlines of the flow
plt.ion()  # turn on interactive mode
f, axarr = plt.subplots(1, 2, sharey=True)
f.suptitle('Convection-diffusion by %s for Pe = %d, \
flux-error = %0.3e' % (fvscheme, Pe, GHC))

# Coordinate arrays
dx = L/n  # cell size in x,y-direction
xf = np.arange(0., L+dx, dx)  # cell face coordinate vector along x,y-axis
xc = np.arange(dx/2., L, dx)  # cell center coordinates along x-axis
xt = np.hstack([0., xc, 1.])  # extended cell center coor. vector, incl. walls
Xc, Yc = np.meshgrid(xc, xc)  # cell center coordinate arrays
Xt, Yt = np.meshgrid(xt, xt)  # extended cell center coor. arrays, incl. walls

# Generate convective velocity field at cell faces
if problem == 1:  # problem 1 - uniform flow
    Ut = np.ones((np.size(Xc, 0), np.size(Xc, 1)))
    Vt = np.zeros((np.size(Xc, 0), np.size(Xc, 1)))
elif problem == 2:  # problem 2 - corner flow
    Ut = -Xc.copy()
    Vt = Yc.copy()
else:
    print('problem not implemented')

axarr[0].streamplot(  # only supports an evenly spaced grid
    xc, xc, Ut, Vt, density=1, linewidth=2,
    color=T, cmap=cm.coolwarm, norm=None, arrowsize=1,
    arrowstyle='-|>', minlength=0.3)
axarr[0].set_title('Streamlines')
axarr[0].set_xlabel('x')
axarr[0].set_ylabel('y')
axarr[0].grid(True)
axarr[0].set_xlim(0, 1)
axarr[0].set_ylim(0, 1)

# Temperature field
p = axarr[1].pcolor(Xt, Yt, TT, cmap=cm.coolwarm, vmin=0, vmax=1)
axarr[1].set_title('Temperature')
axarr[1].set_xlabel('x')
axarr[1].set_ylabel('y')
axarr[1].grid(True)
axarr[1].set_xlim(0, 1)
axarr[1].set_ylim(0, 1)
f.colorbar(p, ax=axarr[1])

f.show()
