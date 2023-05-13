######################################
#
# Tools translated from Michael Weiler's original R code
#
# For details of the formalism refer to 
#     https://ui.adsabs.harvard.edu/abs/2023A%26A...671A..52W/abstract
# 
# This implementation:
#     F. Anders (ICCUB, 2023)
#
######################################

import numpy as np
from scipy.linalg import eigvals, svd
import scipy.stats as stats
from scipy.special import hermite, factorial
import pickle

def HermiteFunction(x, n, scipy=False):
    """
    Computes the first n Hermite functions on an array x
    
    # Inputs:
      x - the points where to evaluate the Hermite functions
      n - the number of Hermite functions to provide (0,...,n-1)
    """
    m = len(x)
    H = np.empty((n, m))
    
    if scipy:
        for ii in np.arange(n):
            H[ii, :] = hermite(ii)(x) * np.exp(-0.5 * x*x)/np.sqrt(2**ii * factorial(ii) * np.sqrt(np.pi))
    else:
        # Use MW's iterative implementation 
        pe = np.pi**-0.25
        ex = np.exp(-0.5 * x*x)

        H[0, :] = pe * ex  # the zero Hermite function

        if n > 1:
            H[1, :] = np.sqrt(2) * pe * x * ex  # the first Hermite function
            if n > 2:
                for i in np.arange(2, n):
                    # using the recurrence relation from Cohen-Tannoudji et al. 
                    # (Quantum Mechanics, Vol. 1, chapter 5.6):
                    c1 = np.sqrt(2/(i))
                    c2 = np.sqrt((i-1)/i)
                    H[i, :] = c1 * x * H[i-1, :] - c2 * H[i-2, :]    
    return H.T

def create_correlation_matrix(values, size=55):
    """
    Creates a correlation matrix with ones in the diagonal
    from the correlation array in column-major storage used by DPAC:
    
    https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/
    chap_datamodel/sec_dm_spectroscopic_tables/ssec_dm_xp_continuous_mean_spectrum.html
    
    # Input:
      values - correlation array
      
    # Optional:
      size   - default: 55
    """
    matrix = np.zeros((size, size))
    # Fill an empty matrix with the array elements using a lower triangular matrix
    inds_ml, inds_nl = np.tril_indices(size-1, 0)
    matrix[inds_ml+1, inds_nl] = values
    # Add the transposed and fill the diagonal with ones:
    return matrix + matrix.T + np.eye(size)

def corr2cov(corr_matrix, uncerts):
    """
    Transform a correlation matrix to a covariance matrix
    and a vector of uncertainties
    
    Args:
    - corr_matrix: a numpy array representing the correlation matrix
    
    Returns:
    - cov_matrix: a numpy array representing the covariance matrix
    """
    std_devs   = np.diag(uncerts)
    cov_matrix = std_devs.T.dot(corr_matrix).dot(std_devs)
    return cov_matrix

def rotate_matrix(M, Rot):
    """
    Rotate a matrix using a unitary transformation matrix
    
    Args:
    - M:   Input matrix
    - Rot: Rotation matrix
    
    Returns:
    - new matrix
    """
    return Rot.dot(M).dot(Rot.T)

class XPConstants(object):
    """
    Summarises all the relatively constant objects used in the calculations.
    From transformatices to the LSF
    """
    def __init__(self, dr="dr3", calib="dr3+weiler2023"):
        """
        Reading everything in ./ConfigurationData/
        
        Usage:
        >>> XPConstants = weiler2023_tools.XPConstants()
        >>> XPConstants.TrafoRP # for example
        """
        ### First the things that do not change:
        
        # Derivative matrices of the Hermite functions
        self.D1     = np.genfromtxt('./ConfigurationData/DerivativeMatrix_D1.csv', delimiter=',')
        self.D2     = np.genfromtxt('./ConfigurationData/DerivativeMatrix_D2.csv', delimiter=',')
        self.D3     = np.genfromtxt('./ConfigurationData/DerivativeMatrix_D3.csv', delimiter=',')
        self.D4     = np.genfromtxt('./ConfigurationData/DerivativeMatrix_D4.csv', delimiter=',')

        # Matrix used to get the roots of Hermite functions
        self.RootsH = np.genfromtxt('./ConfigurationData/RootMatrix_H.csv', delimiter=',')

        # Hermite integrals
        self.IntsH  = np.genfromtxt('./ConfigurationData/HermiteIntegrals.csv', delimiter=',')
        
        ### Then things that (probably) depend on the Gaia Data Release
        if dr=="dr3":
            # Transformation matrices for BP and RP
            self.TrafoBP = np.genfromtxt('./ConfigurationData/BasisTransformationMatrix_BP.csv', delimiter=',')
            self.TrafoRP = np.genfromtxt('./ConfigurationData/BasisTransformationMatrix_RP.csv', delimiter=',')
            # Conversion to pseudo-wavelengths
            self.aBP = 3.062231
            self.bBP = 30.00986
            self.aRP = 3.020529
            self.bRP = 30.00292
        else:
            raise ValueError("Unknown 'dr' option")

        if calib=="dr3+weiler2023":
            # Dispersion
            self.DispersionBP = np.genfromtxt("./ConfigurationData/bpC03_v375wi_dispersion.csv", delimiter=',').T
            self.DispersionRP = np.genfromtxt("./ConfigurationData/rpC03_v142r_dispersion.csv", delimiter=',').T
            # Response function
            self.ResponseBP   = np.genfromtxt("./ConfigurationData/bpC03_v375wi_response.csv", delimiter=',').T
            self.ResponseRP   = np.genfromtxt("./ConfigurationData/rpC03_v142r_response.csv", delimiter=',').T
            # Line-spread functions
            self.LSFBP        = np.genfromtxt("./ConfigurationData/LSFModel_BP.csv", delimiter=',')
            self.LSFRP        = np.genfromtxt("./ConfigurationData/LSFModel_RP.csv", delimiter=',')
        else:
            raise ValueError("Unknown 'calib' option")

class XP_Spectrum(object):
    """
    Gets all information that can be derived from a
    Gaia XP_CONTINUOUS datalink file.
    That is, this class treats the BP/RP spectra.
    
    Usage:
    """
    def __init__(self, t, setup=None, rotate_basis=True, 
                 truncate=False):
        """
        Initialise an XP spectrum.

        # Inputs:
          t     - datalink table row for XP_CONTINUOUS

        # Optional:
          setup - XP constants object or None
        """
        if setup == None:
            self.setup = XPConstants()
        else:
            self.setup = setup
        # Extract the info from the datalink table: coefficients and correlations
        self.alldata = t
        self.source_id = t["source_id"]
        self.BP      = np.array(t["bp_coefficients"])
        self.RP      = np.array(t["rp_coefficients"])
        self.BP_err  = np.array(t["bp_coefficient_errors"])
        self.RP_err  = np.array(t["rp_coefficient_errors"])
        self.BP_corr = create_correlation_matrix(np.array(t["bp_coefficient_correlations"]), 
                                                 len(self.BP))
        self.RP_corr = create_correlation_matrix(np.array(t["rp_coefficient_correlations"]), 
                                                 len(self.RP))
        self.BP_cov  = corr2cov(self.BP_corr, self.BP_err)
        self.RP_cov  = corr2cov(self.RP_corr, self.RP_err)
            
        # Rotate the basis by multiplying the coefficients with
        # the transformation matrix:
        if rotate_basis:
            self.BP     = np.dot(self.setup.TrafoBP.T, self.BP)
            self.RP     = np.dot(self.setup.TrafoRP.T, self.RP)
            self.BP_corr= rotate_matrix(self.BP_corr, self.setup.TrafoBP.T)
            self.RP_corr= rotate_matrix(self.RP_corr, self.setup.TrafoRP.T)
            self.BP_cov = rotate_matrix(self.BP_cov, self.setup.TrafoBP.T)
            self.RP_cov = rotate_matrix(self.RP_cov, self.setup.TrafoRP.T)
            self.BP_err = np.sqrt( np.diagonal(self.BP_cov) )
            self.RP_err = np.sqrt( np.diagonal(self.RP_cov) )
        
    def get_internal_spec(self, xx, instrument="bp"):
        """
        Turn the XP coefficients into an internally-calibrated 
        XP spectrum.

        # Inputs:
          xx          - array for calculating Hermite polynomials

        # Optional:
          instrument  - "bp" or "rp"

        # Returns:
          l, internal - pseudo-wavelength, flux 
        """
        # Calculate Hermite functions on the given grid xx
        H = HermiteFunction(xx, 55)
        # Transform to pseudo-wavelength
        if instrument == "bp":
            a, b     = self.setup.aBP, self.setup.bBP
            internal = np.dot(H, self.BP)
        elif instrument == "rp":
            a, b     = self.setup.aRP, self.setup.bRP
            internal = np.dot(H, self.RP)
        else:
            raise ValueError("Choose either 'bp' or 'rp' as instrument.")
        l = xx * a + b
        return l, internal

def getRoots(coef, cov, setup=None, n=None, small=1E-7, conditioning=False):
    """
    Computes the roots and their errors for a linear combination of Hermite functions.
    
    # Inputs:
      coef - the coefficients of the linear combination
      cov  - the covariance matrix for coef
    
    # Optional inputs:
      setup - list with transformation matrices: HermiteTransfromationMatrices.RData
      small - limit on the relative absolute value of the imaginary part that is tolerated (default: 1E-7)
      conditioning - if TRUE, the condition numbers of the roots are also provided.
    
    # Output 
      dictionary including all roots, and the real roots and their errors.
    """
    if n is None:
        n = len(coef) - 1

    if setup is None:
        # load from disk if not provided
        setup = XPConstants()

    # the non-standard companion matrix (eq. 26):
    B         = setup.RootsH[:n,:n] 
    B[:, -1]  = B[:, -1] - np.sqrt(n/2) * coef[:-1] / coef[-1]  # exchange the last column
    # the roots as eigenvalues of B:
    roots     = eigvals(B)
    # select real roots:
    d         = np.where(abs(np.imag(roots)) / abs(np.real(roots)) < small)[0]
    realRoots = np.real(roots[d])
    nReal     = len(d)

    # compute the covariance matrix of the real roots in linear approximation:
    Hermite = np.matmul(np.matmul(HermiteFunction(realRoots, n+2), 
                                  setup.D1[0:n+2, 0:n+2]), np.concatenate((coef, [0])))

    # compute the condition numbers of the roots if requested:
    if conditioning and nReal > 0:
        # left eigenvectors:
        LE  = HermiteFunction(realRoots, nReal)
        # right eigenvectors:
        sv  = svd(LE)
        tmp = 1. / sv[1]
        tmp[np.where(sv[1] < 1E-12 * max(sv[1]))] = 0
        RE  = sv[2].T @ np.diag(tmp) @ sv[0].T
        condition = np.abs(np.array([np.inner(LE[:,i], RE[:,i]) / 
                                     (np.sqrt(np.inner(LE[:,i], LE[:,i])) * 
                                      np.sqrt(np.inner(RE[:,i], RE[:,i]))) for i in range(nReal)]))
    else:
        condition = None

    if nReal == 1:
        J = -1/Hermite * HermiteFunction(realRoots, n=n+1)
    else:
        J = -np.diag(1/Hermite) @ HermiteFunction(realRoots, n=n+1)
    J = np.real(J)
    err = J @ cov[0:n+1, 0:n+1] @ J.T

    return {"roots": roots, 
            "realRoots": realRoots, 
            "sigma": np.sqrt(np.diag(err)), 
            "cov": err, 
            "condition": condition}


def getLocalExtrema(coef, cov, setup=None, conditioning=False):
    """
    Computes the positions and their uncertainties of local minima and maxima
    in a linear combination of Hermite functions.
    
    # Inputs:
      coef - the coefficients of the linear combination
      cov - the covariance matrix of coef
    
    # Optional inputs:
      setup: list with the required matrices for the zero and first derivatives
                    (computed with getDerivativeMatrices())
      conditioning - if TRUE, the condition numbers of the real roots are also computed.
                     Much slower computation than without.

    # Output 
      dictionary including all computed information
    """
    n = len(coef)
    
    if setup is None:
        setup = XPConstants()

    coef1 = np.matmul(setup.D1[0:n+1, 0:n+1], np.concatenate([coef, [0]]))
    cov1  = np.matmul(setup.D1[0:n+1, 0:n+1], 
                      np.vstack([np.hstack([cov, np.zeros((n, 1))]), 
                                 np.zeros((1, n+1))])) @ np.transpose(setup.D1[0:n+1, 0:n+1])
    cov2  = np.zeros((n+2, n+2))
    cov2[:n, :n] = cov

    roots = getRoots(coef1, cov1, setup=setup, conditioning=conditioning)

    # computing the values and errors of the second derivative at the roots:
    M = np.matmul(HermiteFunction(roots['realRoots'], n+2), setup.D2[0:n+2, 0:n+2])
    v = np.matmul(M, np.concatenate([coef, [0, 0]]))
    E = np.matmul(np.matmul(M, cov2), np.transpose(M))

    kind = np.repeat("minimum", len(roots['realRoots']))
    kind[np.where(v < 0)] = "maximum"

    return {'location': roots['realRoots'], 
            'error': roots['sigma'], 
            'cov': roots['cov'], 
            'condition': roots['condition'], 
            'kind': kind,
            'secondDerivativeAtRoots': v, 
            'ErrorOnSecondDerivativeAtRoots': np.sqrt(np.diag(E)), 
            'CovForSecondDerivatives': E, 
            'roots': roots}

def getInflectionPoints(coef, cov, setup=None, conditioning=False):
    """
    Computes the positions and their uncertainties of inflection points
    in a linear combination of Hermite functions.
    
    # Inputs:
    coef - the coefficients of the linear combination
    cov - the covariance matrix of coef
    
    # Optional inputs:
    setup: list with the required matrices for the zero and first derivatives
           (computed with getDerivativeMatrices())
    conditioning - if TRUE, the condition numbers of the real roots are also computed.
                   Much slower computation than without.

    # Output 
      dictionary including all computed information
    """
    n = len(coef)
    if setup is None:
        # use the pre-defined HermiteTransformationMatrices from RData file
        # (assuming the file is saved in the same directory as this script)
        setup = XPConstants()

    coef1 = setup.D2[:(n+2), :(n+2)].dot(np.append(coef, [0, 0]))
    cov1  = setup.D2[:(n+2), :(n+2)].dot(
            np.block([[cov, np.zeros((n, 2))], [np.zeros((2, n+2))]])).dot(
            setup.D2[:(n+2), :(n+2)].T)
    cov2  = np.zeros((n+3, n+3))
    cov2[:n, :n] = cov

    roots = getRoots(coef1, cov1, setup=setup, conditioning=conditioning)

    # computing the values and errors of the third derivative at the roots:
    M = HermiteFunction(roots["realRoots"], n+3).dot(setup.D3[:(n+3), :(n+3)])
    v = M.dot(np.append(coef, [0, 0, 0]))
    E = M.dot(cov2).dot(M.T)

    kind = np.repeat("increasing", len(roots["realRoots"]))
    kind[v < 0] = "decreasing"

    return {"location": roots["realRoots"],
            "error": roots["sigma"],
            "covariance": roots["cov"],
            "condition": roots["condition"],
            "kind": kind,
            "thirdDerivativeAtRoots": v,
            "ErrorOnThirdDerivativeAtRoots": np.sqrt(np.diag(E)),
            "CovForThirdDerivatives": E,
            "roots": roots}


def getLinesInNDeriv(coefIn, covIn, N=0, instrument="none", 
                     setup=None):
    """
    Extracts the basic line parameters. 
    First it converts the input to its N-th derivative.
    
    # Output:
      Dictionary of extrema positions, errors on the extrema positions, 
                    the S/N for the second derivatives at the positions 
                    of the extrema, and the p-value of the extrema. 
    """
    n = len(coefIn)

    if setup is None:
        setup = XPConstants()
    if instrument == "BP":
        a = setup.aBP
        b = setup.bBP
    elif instrument == "RP":
        a = setup.aRP
        b = setup.bRP
    else:
        a = 1
        b = 0

    coefPad = np.concatenate((coefIn, np.zeros(N)))
    covPad  = np.zeros((n + N, n + N))
    covPad[:n, :n] = covIn

    if N == 0:
        coef   = coefIn
        cov    = covIn
    else:
        if N == 1:
            Transf = setup.D1[:n + N, :n + N]
        elif N == 2:
            Transf = setup.D2[:n + N, :n + N]
        elif N == 3:
            Transf = setup.D3[:n + N, :n + N]
        else:
            raise ValueError("N too high. Don't be stupid.")
        coef   = np.dot(Transf, coefPad)
        cov    = np.dot(np.dot(Transf, covPad), Transf.T)

    e = getLocalExtrema(coef, cov, setup=setup)
    p = e["location"] * a + b
    x = e["secondDerivativeAtRoots"] / e["ErrorOnSecondDerivativeAtRoots"]
    signif = 1 - np.exp(-x*x / 2)

    inf = getInflectionPoints(coef, cov, setup=setup)
    pinf = inf["location"] * a + b
    xinf = inf["thirdDerivativeAtRoots"] / inf["ErrorOnThirdDerivativeAtRoots"]
    signifInf = 1 - np.exp(-xinf*xinf / 2)

    widths = [np.min(pinf[pi < pinf]) - np.max(pinf[pi > pinf]) for pi in p]
    # TBD!
    widthsError = [np.sqrt(inf["covariance"][np.where(pinf == min(pinf[pi > pinf]))[0][0], 
                                             np.where(pinf == min(pinf[pi > pinf]))[0][0]] + 
                           inf["covariance"][np.where(pinf == max(pinf[pi < pinf]))[0][0], 
                                             np.where(pinf == max(pinf[pi < pinf]))[0][0]] -
                           2 * inf["covariance"][np.where(pinf == min(pinf[pi > pinf]))[0][0], 
                                                 np.where(pinf == max(pinf[pi < pinf]))[0][0]])
                   for pi in p]

    res = {"N": N, 
           "estimLinePos": p, 
           "estimLineErr": e["error"] * a, 
           "SNonSecondDerivative": x, 
           "estimSignif": signif,
           "lineWidths": widths, 
           "lineWidthsError": widthsError, 
           "secondDerivativeAtRoots": e["secondDerivativeAtRoots"] / a**(N+2),
           'ErrorOnSecondDerivativeAtRoots': e['ErrorOnSecondDerivativeAtRoots']/a**(N+2),
           'CovForSecondDerivatives': e['CovForSecondDerivatives']/a**(2*N+4),
           'kind': e['kind'],
           'estimInfPos': pinf, 
           'estimInfErr': inf['error']*a, 
           'estimInfCov': inf['covariance']*a*a, 
           'estimInfSignif': signifInf, 
           'infKind': inf['kind'],
           'instrument': instrument}
    return res

def getLSFWidth(LSF, u0, instrument, config, HermiteTransformationMatrices, order=0):
    """
    Get the width of the line spread function.
    """
    D = 0.1
    if instrument == "BP":
        a = setup.aBP
        b = setup.bBP
        n = setup.LSFBP
    elif instrument == "RP":
        a = setup.aRP
        b = setup.bRP
        n = setup.LSFRP

    if LSF['n'] > 0:
        H = HermiteFunction((u0-b)/a, LSF['n']+LSF['dn'])
        lsf = np.dot(H, LSF['L'].T)
    else:
        lsf = LSF['L']

    if order == 0:
        c1 = np.dot(HermiteTransformationMatrices['D1'][0:(n+1), 0:(n+1)], np.concatenate((lsf, np.zeros(1))))
        l = getLines(c1, np.diag(np.ones(len(c1))), instrument=instrument)
    else:
        c3 = np.dot(HermiteTransformationMatrices['D3'][0:(n+3), 0:(n+3)], np.concatenate((lsf, np.zeros(3))))
        l = getLines(c3, np.diag(np.ones(len(c3))), instrument=instrument)

    idx1 = np.argmin(np.abs(l['estimLinePos']-u0))
    idx2 = np.argmin(np.abs(l['estimLinePos']-u0+D))
    idx3 = np.argmin(np.abs(l['estimLinePos']-u0-D))

    idx = np.unique(np.concatenate(([idx1], [idx2], [idx3])))

    return {'p1': l['estimLinePos'][idx[0]], 
            'p2': l['estimLinePos'][idx[1]], 
            'D': np.abs(l['estimLinePos'][idx[0]] - l['estimLinePos'][idx[1]])}
