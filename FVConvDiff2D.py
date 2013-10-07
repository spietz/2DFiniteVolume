import numpy as np
from scipy.sparse import spdiags


def preprocess(n, L, Pe, problem, fvscheme):
    """
    Pre-process input for simulation of steady convection-diffusion heat problem
    on two-dimensional square domain using the Finite Volume Method.
    Pre-processing returns the convection-diffusion system matrix A(1:n^2,1:n^2)
    and the source array s(1:n,1:n).

    Implementation does not include source terms and is limited to the
    following two sets of convective velocity fields and boundary conditions:

    1) uniform flow [u,v] = [1,0], homogeneous Neumann BC. at north and south
    walls (dTn/dy=dTs/dy=0), homogeneous Dirichlet BC at west wall (Tw=0),
    inhomogeneous Dirichlet BC at east wall (Te=1).

    2) corner flow [u,v] = [-x,y], homogeneous Neumann BC. at north wall
    (dTn/dy=0), inhomogeneous Dirichlet BC at west wall (Tw=0), inhomogeneous
    Dirichlet BC at east wall (Te=1), inhomogeneous Dirichlet BC
    at south wall (Ts=x)

    Syntax:
    -------
    A, s = preprocess(n, L, Pe, problem, fvscheme)

    Input:
    ------
    n         :  Number of cells along x,y-axis
    L         :  Size of square in x,y-direction
    Pe        :  Global Peclet number
    problem   :  Problem : 1 or 2, selects case of convective field and BCs
    fvscheme  :  Finite volume scheme: 'cds-cds' or 'uds-cds'

    Output:
    -------
    A         :  Convection-diffusion system matrix, A(1:n^2,1:n^2)
    s         :  Source array with BC contributions, s(1:n,1:n)
    """

    # Coordinate arrays
    dx = L/n  # cell size in x,y-direction
    xf = np.arange(0., L+dx, dx)  # cell face coordinate vector along x,y-axis
    xc = np.arange(dx/2., L, dx)  # cell center coordinates along x-axis

    # Generate convective velocity field at cell faces (!)
    if problem == 1:  # problem 1 - uniform flow
        UFw = np.ones((n, n))  # western face velocity, u = 1
        UFe = np.ones((n, n))  # eastern face velocity, u = 1
        VFs = np.zeros((n, n))  # southern face velocity, v = 0
        VFn = np.zeros((n, n))  # northern face velocity, v = 0
    elif problem == 2:  # problem 2 - corner flow
        UFw = np.tile(-xf[0:n], (n, 1))  # western face velocity, u = -x
        UFe = np.tile(-xf[1:n+1], (n, 1))  # eastern face velocity, u = -x
        VFs = np.tile(xf[0:n, np.newaxis], (1, n))  # southern face velocity, v = y
        VFn = np.tile(xf[1:n+1, np.newaxis], (1, n))  # northern face velocity, v = y
    else:  # problem unknown
        print('problem not implemented')

    # Generate convective face flux matrices
    Fw = dx*Pe*UFw
    Fe = dx*Pe*UFe
    Fs = dx*Pe*VFs
    Fn = dx*Pe*VFn

    # Generate diffusive flux matrix
    D = -np.ones((n, n))  # -dx/dy

    # Generate coefficient arrays
    if fvscheme.lower() == 'cds-cds':  # CDS-CDS FV-scheme applied
        aW = D-Fw/2.
        aE = D+Fe/2.
        aS = D-Fs/2.
        aN = D+Fn/2.
        aP = -(aW+aE+aS+aN)+Fe-Fw+Fn-Fs
    elif fvscheme.lower() == 'uds-cds':  # UDS-CDS FV-scheme applied
        aW = D+np.minimum(0, -Fw)
        aE = D+np.minimum(0, Fe)
        aS = D+np.minimum(0, -Fs)
        aN = D+np.minimum(0, Fn)
        aP = -(aW+aE+aS+aN)+Fe-Fw+Fn-Fs
    else:  # fvscheme unknown
        print('fvscheme not implemented')

    s = np.zeros((n, n))  # initialize source array

    # Impose BCs using ghost point approach
    if problem == 1:  # specified Tw=0, Te=1, dT/dy|s = 0, dT/dy|n = 0

        Tw = np.zeros(n)
        Te = np.ones(n)
        Ds = np.zeros(n)
        Dn = np.zeros(n)

        # Correct w domain boundary points (j,i) = [:, 0], 1st order dirichlet:
        aP[:, 0] = aP[:, 0]-aW[:, 0]
        s[:, 0] = s[:, 0] - 2 * Tw * aW[:, 0]
        aW[:, 0] = 0

        # Correct e domain boundary points (j,i) = [:, n^2-1], 1st order dirichlet:
        aP[:, -1] = aP[:, -1] - aE[:, -1]
        s[:, -1] = s[:, -1] - 2 * Te * aE[:, -1]
        aE[:, -1] = 0

        # Correct s domain boundary points (j,i) = [0, :], 2nd order NeumaNn:
        aP[0, :] = aP[0, :] + aS[0, :]
        s[0, :] = s[0, :] + aS[0, :] * dx * Ds
        aS[0, :] = 0  # Neumann: 2nd order FD

        # Correct n domain boundary points (j,i) = [n^2, :], 2nd order NeumaNn:
        aP[-1, :] = aP[-1, :] + aN[-1, :]
        s[-1, :] = s[-1, :] - aN[-1, :] * dx * Dn
        aN[-1, :] = 0

    elif problem == 2:  # specified Tw=0, Te=1, Ts = x, dT/dy|n = 0

        Te = np.ones(n)
        Tw = np.zeros(n)
        Ts = xc
        Dn = np.zeros(n)

        # Correct w domain boundary points (j,i) = [:, 0], 1st order dirichlet:
        aP[:, 0] = aP[:, 0] - aW[:, 0]
        s[:, 0] = s[:, 0] - 2 * Tw * aW[:, 0]
        aW[:, 0] = 0

        # Correct e domain boundary points (j,i) = [:, -1], 1st order dirichlet:
        aP[:, -1] = aP[:, -1] - aE[:, -1]
        s[:, -1] = s[:, -1] - 2 * Te * aE[:, -1]
        aE[:, -1] = 0

        # Correct s domain boundary points (j,i) = [0, :], 1st order dirichlet:
        aP[0, :] = aP[0, :]-aS[0, :]
        s[0, :] = s[0, :] - 2 * Ts * aS[0, :]
        aS[0, :] = 0

        # Correct n domain boundary points (j,i) = [n^2, :], 2nd order NeumaNn:
        aP[-1, :] = aP[-1, :] + aN[-1, :]
        s[-1, :] = s[-1, :] - aN[-1, :] * dx * Dn
        aN[-1, :] = 0

    else:
        print('BCs not implemented')

    # Assemble sparse 5-diagonal system matrix A (of size n^2*n^2)
    offsets = np.array([-n, -1, 0, 1, n])
    data = np.hstack([
        np.vstack([aW.reshape(n**2, 1, order='F')[n:n**2], np.zeros((n, 1))]),
        np.vstack([aS.reshape(n**2, 1, order='F')[1:n**2], np.zeros((1, 1))]),
        aP.reshape(n**2, 1, order='F')[0:n**2],
        np.vstack([np.zeros((1, 1)), aN.reshape(n**2, 1, order='F')[0:n**2-1]]),
        np.vstack([np.zeros((n, 1)), aE.reshape(n**2, 1, order='F')[0:n**2-n]])
    ])

    A = spdiags(data.T, offsets, n**2, n**2, format="csc")

    return A, s


def postprocess(T, n, L, Pe, problem, fvscheme):

    """
    Post-process results from simulation of steady convection-diffusion heat
    problem on two-dimensional square domain using the Finite Volume Method.
    Post-processing returns the temperature field TT(1:n+2,1:n+2) extrapolated
    to the walls, the global heat conservation GHC and the net convective and
    diffusive fluxes through the walls CF,DF.

    Implementation does not include source terms and is limited to the
    following two sets of convective velocity fields and boundary conditions:

    1) uniform flow [u,v] = [1,0], homogeneous Neumann BC. at north and south
    walls (dTn/dy=dTs/dy=0), homogeneous Dirichlet BC at west wall (Tw=0),
    inhomogeneous Dirichlet BC at east wall (Te=1).

    2) corner flow [u,v] = [-x,y], homogeneous Neumann BC. at north wall
    (dTn/dy=0), inhomogeneous Dirichlet BC at west wall (Tw=0), inhomogeneous
    Dirichlet BC at east wall (Te=1), inhomogeneous Dirichlet BC
    at south wall (Ts=x)

    Syntax:
    -------
    TT, GHC, CF, DF = postprocess(T, n, L, Pe, problem, fvscheme)

    Input:
    ------
    T         :  Temperature at cell nodes, T(1:n,1:n)
    n         :  Number of cells along x,y-axis
    L         :  Size of square in x,y-direction
    Pe        :  Global Peclet number
    problem   :  Problem : 1 or 2, selects case of convective field and BCs
    fvscheme  :  Finite volume scheme: 'cds-cds' or 'uds-cds'

    Output:
    -------
    TT        :  Temperature field extrapolated to walls, TT(1:n+2,1:n+2)
    GHC       :  Global heat conservation, scalar (computed from wall fluxes)
    CF,DF     :  Conv. and diff. net fluxes through walls, CF=[CFw,CFe,CFs,CFn]
    """
    # Coordinate arrays
    dx = L/n  # cell size in x,y-direction
    xf = np.arange(0., L+dx, dx)  # cell face coordinate vector along x,y-axis
    xc = np.arange(dx/2., L, dx)  # cell center coordinates along x-axis

    # Generate convective velocity field at cell faces (!)
    if problem == 1:  # problem 1 - uniform flow
        UFw = np.ones((n, n))  # western face velocity, u = 1
        UFe = np.ones((n, n))  # eastern face velocity, u = 1
        VFs = np.zeros((n, n))  # southern face velocity, v = 0
        VFn = np.zeros((n, n))  # northern face velocity, v = 0
    elif problem == 2:  # problem 2 - corner flow
        UFw = np.tile(-xf[0:n], (n, 1))  # western face velocity, u = -x
        UFe = np.tile(-xf[1:n+1], (n, 1))  # eastern face velocity, u = -x
        VFs = np.tile(xf[0:n, np.newaxis], (1, n))  # southern face velocity, v = y
        VFn = np.tile(xf[1:n+1, np.newaxis], (1, n))  # northern face velocity, v = y
    else:  # problem unknown
        print('problem not implemented')

    # Generate convective face flux matrices
    Fw = dx*Pe*UFw
    Fe = dx*Pe*UFe
    Fs = dx*Pe*VFs
    Fn = dx*Pe*VFn

    # Extrapolate temperature field to walls
    TT = np.zeros((n+2, n+2))  # init extended temp field
    TT[1:n+1, 1:n+1] = T

    if problem == 1:

        Tw = np.zeros(n)
        Te = np.ones(n)
        Ds = np.zeros(n)
        Dn = np.zeros(n)

        if fvscheme.lower() == 'cds-cds':  # CDS-CDS FV-scheme applied
            # Correct w domain boundary points (j,i) = (:,1), 1st order dirichlet:
            TT[1:n+1, 0] = Tw

            # Correct e domain boundary points (j,i) = (:,end), 1st order dirichlet:
            TT[1:n+1, -1] = Te

            # Correct s domain boundary points (j,i) = (1,:), 2nd order Neumann:
            TT[0, 1:n+1] = T[0, :] - 0.5 * dx * Ds

            # Correct n domain boundary points (j,i) = (end,:), 2nd order Neumann:
            TT[-1, 1:n+1] = T[-1, :] + 0.5 * dx * Dn

        elif fvscheme.lower() == 'uds-cds':

            # Correct w domain boundary points (j,i) = (:,1), 1st order dirichlet:
            TT[1:n+1, 0] = 2.0 * Tw - T[:, 0]

            # Correct e domain boundary points (j,i) = (:,end), 1st order dirichlet:
            TT[1:n+1, -1] = T[:, -1]

            # Correct s domain boundary points (j,i) = (1,end), 2nd order Neumann:
            TT[0, 1:n+1] = T[0, :] - dx * Ds

            # Correct n domain boundary points (j,i) = (end,:), 2nd order Neumann:
            TT[-1, 1:n+1] = T[-1, :]

    elif problem == 2:

        Tw = np.zeros(n)
        Te = np.ones(n)
        Ts = xc
        Dn = np.zeros(n)

        if fvscheme.lower() == 'cds-cds':
            # Correct w domain boundary points (j,i) = (:,1), 1st order dirichlet:
            TT[1:n+1, 0] = Tw

            # Correct e domain boundary points (j,i) = (:,end), 1st order dirichlet:
            TT[1:n+1, -1] = Te

            # Correct s domain boundary points (j,i) = (1,:), 1st order dirichlet:
            TT[0, 1:n+1] = Ts

            # Correct n domain boundary points (j,i) = (end,:), 2nd order Neumann:
            TT[-1, 1:n+1] = T[-1, :] + 0.5 * dx * Dn

        elif fvscheme.lower() == 'uds-cds':
            # Correct w domain boundary points (j,i) = (:,1), 1st order dirichlet:
            TT[1:n+1, 0] = T[:, 0]

            # Correct e domain boundary points (j,i) = (:,end), 1st order dirichlet:
            TT[1:n+1, -1] = 2 * Te - T[:, -1]

            # Correct s domain boundary points (j,i) = (1,:), 1st order dirichlet:
            TT[0, 1:n+1] = 2 * Ts - T[0, :]

            # Correct n domain boundary points (j,i) = (end,:), 2nd order Neumann:
            TT[-1, 1:n+1] = T[-1, :]

    # Compute temperature gradients at walls
    if problem == 1:
        DTw = 2. / dx * (T[:, 0] - Tw)
        DTe = 2. / dx * (Te - T[:, -1])
        DTs = np.ones(n) * Ds
        DTn = np.ones(n) * Dn
    elif problem == 2:
        DTw = 2. / dx * (T[:, 0] - Tw)
        DTe = 2. / dx * (Te - T[:, -1])
        DTs = 2. / dx * (T[0, :] - Ts)
        DTn = np.ones(n) * Dn

    # Convective and diffusive wall fluxes
    DF = dx * np.vstack([DTw, DTe, DTs, DTn])
    CFw = Fw[:, 0] * TT[1:n+1, 0]
    CFe = Fe[:, -1] * TT[1:n+1, -1]
    CFs = Fs[0, :] * TT[0, 1:n+1]
    CFn = Fn[-1, :] * TT[-1, 1:n+1]
    CF = np.vstack([CFw, CFe, CFs, CFn])

    # Global heat conservation
    GHC = np.sum((CF[1, :] - DF[1, :]) - (CF[0, :] - DF[0, :])) \
        + np.sum((CF[3, :] - DF[3, :]) - (CF[2, :] - DF[2, :]))

    return TT, GHC, CF, DF
