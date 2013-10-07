 #################################################################################
 # SORoptimalOmega.py                                                            #
 #   Script for computing the optimal relaxation parameter omega for             #
 #   SOR-iterations on system of linear equations A*t = s. A is the system       #
 #   matrix corresponding to the steady convection-diffusion heat problem on     #
 #   two-dimensional square domain using the Finite Volume Method.               #
 #   Implementation is limited to the following set of convective velocity       #
 #   fields and boundary conditions:                                             #
 #     1) uniform flow [u,v] = [1,0], homogeneous Neumann BC. at north and south #
 #     walls (dTn/dy=dTs/dy=0), homogeneous Dirichlet BC at west wall (Tw=0),    #
 #     inhomogeneous Dirichlet BC at east wall (Te=1).                           #
 #   Three cases analyzed: i) Pe=0, fvscheme='cds-cds'/'uds-cds' (scheme doesn't #
 #   matter), ii) Pe=10, fvscheme='cds-cds', iii) Pe=10, fvscheme='uds-cds'      #
 #                                                                               #
 # Input         :                                                               #
 #   N           :  Vector with number of cells along x,y-axis, e.g.             #
 #                   N = [10, 25, 50, 100] (should be rather small to use eigs)  #
 #   L           :  Size of square in x,y-direction                              #
 #   problem     :  Problem , (set =1), selects case of convective field         #
 #                  and BCs                                                      #
 #   PE          :  Vector of global Peclet number, PE = [0, 10, 10]             #
 #   FVSCHEME    :  Array with FV schemes: ['cds-cds', 'cds-cds', 'uds-cds']     #
 #   Nasymp      :  Verty large number of cells, used in polyfit to ensure       #
 #                  correct asymptotic behavior for omega_opt(n)                 #
 #   Oasymp      :  Asymptotic value for omega_opt(Nasymp) ~ Oasymp, used in     #
 #                  polyfit to ensure correct asymptotic behavior                #
 #                                                                               #
 # Output        :                                                               #
 #   - plot of optimal SOR relaxation parameter w.r.t the number of cells n      #
 #   including tendency curves, which illustrate the development for large n     #
 #   - polynomial coefficients                                                   #
 #################################################################################

import SIMPy
import FVConvDiff2D
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

## Input
# Note: Pe and fvscheme are set inside loop
N = np.array([10, 25, 50, 100])  # number of cells along x,y-axis
L = 1.0  # size of square in x,y-direction
problem = 1  # problem to solve (set =1)
PE = np.array([0., 10., 10.])  # global Peclet numbers
FVSCHEME = ['cds-cds', 'cds-cds', 'uds-cds']  # finite volume schemes analyzed
Nasymp = 1e8  # large number of cells for asymp.
Oasymp = 2-1e-6  # asymp. value for omega_opt(Nasymp)

## Initialize
OMopt = np.zeros((len(PE), len(N)))  # optimal omega array
P = np.zeros((len(PE), 2))  # fitting poly. coeff. for OMopt
Ni = np.round(np.logspace(1, 4, 100))  # cell-number vector for tendency curves
TC = np.zeros((len(PE), len(Ni)))  # tendency curves for OMopt-results

## Compute optimal omega_SOR for coarse grids
for i in range(0, len(PE)):
    for j in range(0, len(N)):
        n = N[j]

        # sparse system matrix
        A, _ = FVConvDiff2D.preprocess(n, L, PE[i], problem, FVSCHEME[i])
        
        # obtain iteration matrix for Jacobi method
        _, _, _, _, G = SIMPy.solve(
            A, np.ones(n**2), "jacobi",
            1, 0, 1, np.ones(n**2), True)

        evals_large, _ = eigs(G, 6, which='LM')  # 6 largest eigenvalues
        srGJ = np.max(evals_large)  # magnitude of largest is spectral radius
        OMopt[i, j] = 2. / (1 + np.sqrt(1 - srGJ**2))  # theoretical optimal

## Fit omega_opt(n) = c1/(1+c2*sin(pi/n)) using weighted-least-squares polyfit
Ne = np.hstack([N, Nasymp])  # add large number of cells to N
# add corresponding ~2 to OMopt
OMopte = np.hstack([OMopt, Oasymp*np.ones((len(PE), 1))])
w = Ne  # weight factors for polyfit

legendStrArray = []  # empty array of strings for the plot legend
for i in range(0, len(PE)):
    fitX = -OMopte[i, :] * np.sin(np.pi / Ne)  # "x" in linear relation
    fitY = OMopte[i, :]  # "y" in linear relation
    # coefficients for regression
    P[i, :] = np.polyfit(fitX, fitY, 1, None, False, w)
    TC[i, :] = P[i, 1] / (1 + P[i, 0] * np.sin(np.pi / Ni))  # tendency curves
    legendStrArray.append('FV-scheme=' + FVSCHEME[i] + ', Pe=%0.1f' % (PE[i]))

## Save regression parameters to file
np.savetxt('coefficients.txt', P)

## Plot
plt.ion()  # turn on interactive mode
fig1 = plt.figure(1)
fig1.clf()
ax = fig1.add_subplot(1, 1, 1)
ax.plot(N, OMopt.T, 'x')
ax.plot(Ni, TC.T, '-k')
ax.legend(legendStrArray, loc=0)
ax.set_xscale('log')
ax.set_yscale('linear')
ax.grid(True)
ax.set_title('Relaxation factor for SOR iterative method')
ax.set_xlabel('N')
ax.set_ylabel('optimal value')
fig1.show()
