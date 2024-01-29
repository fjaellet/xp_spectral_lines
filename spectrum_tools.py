######################################
#
#   xp_spectral_lines/spectrum_tools.py
#
# Tools translated from Michael Weiler's original R code
# For details of the formalism refer to 
#     https://ui.adsabs.harvard.edu/abs/2023A%26A...671A..52W/abstract
# 
# This implementation:
#     F. Anders (ICCUB, 2023-24)
#
######################################
import numpy as np
from scipy.linalg import eigvals, svd
import scipy.stats as stats
import pickle

import math_tools, line_analysis

class XPConstants(object):
    """
    Summarises all the relatively constant objects used in the calculations.
    From Hermite transformation matrices to the LSF.

    Usage examples:
    >>> XPConstants = weiler2023_tools.XPConstants()
    >>> XPConstants.TrafoRP
    >>> XPConstants.get_pseudowavelength(770., instrument="rp", shift=0.)
    """
    def __init__(self, dr="dr3", calib="dr3+weiler2023"):
        """
        Reading everything in ./ConfigurationData/
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
        
    def get_pseudo_wavelength(self, l, instrument="bp", shift=0.):
        """
        Calculate the pseudowavelength for a given wavelength 
        by interpolating the dispersion relation.
        
        # Arguments:
            l  - Wavelength in nm
        """
        if instrument=="bp":
            return np.interp(l, self.DispersionBP[:,0], self.DispersionBP[:,1]) + shift
        elif instrument=="rp":
            return np.interp(l, self.DispersionRP[:,0], self.DispersionRP[:,1]) + shift

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
        self.BP_corr = math_tools.create_correlation_matrix(np.array(t["bp_coefficient_correlations"]), 
                                                 len(self.BP))
        self.RP_corr = math_tools.create_correlation_matrix(np.array(t["rp_coefficient_correlations"]), 
                                                 len(self.RP))
        self.BP_cov  = math_tools.corr2cov(self.BP_corr, self.BP_err)
        self.RP_cov  = math_tools.corr2cov(self.RP_corr, self.RP_err)
            
        # Rotate the basis by multiplying the coefficients with
        # the transformation matrix:
        if rotate_basis:
            self.BP     = np.dot(self.setup.TrafoBP.T, self.BP)
            self.RP     = np.dot(self.setup.TrafoRP.T, self.RP)
            self.BP_corr= math_tools.rotate_matrix(self.BP_corr, self.setup.TrafoBP.T)
            self.RP_corr= math_tools.rotate_matrix(self.RP_corr, self.setup.TrafoRP.T)
            self.BP_cov = math_tools.rotate_matrix(self.BP_cov, self.setup.TrafoBP.T)
            self.RP_cov = math_tools.rotate_matrix(self.RP_cov, self.setup.TrafoRP.T)
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
        H = math_tools.HermiteFunction(xx, 55)
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
    
def get_LSF_width(u0, setup=None, instrument="bp", 
                  order=0, n=55, nmax=100, D=0.1):
    """
    Get the width of the line spread function
    at one particular pseudo-wavelength.

    This uses the LSF matrix stored in self.LSFBP/self.LSFRP
    (the LSF is a function of u and u' in Weiler+2023 and can be 
    developed in the same basis of Hermite functions).

    # Parameters:
        u0   - pseudowavelength at which the LSF width is to be computed
    # Optional:
        setup      - XPConstants() object
        instrument - "bp" or "rp"
        order      - 0 or 2
        n          - number of relevant coefficients (default: 55)
        nmax       - maximum number of coefficients needed for the 
                     u' dimension of the LSF (default: 100)
        D          - tolerance (default: 0.1)
    # Returns:
        {'p1':p1, 'p2':p2, 'D':D} - Dictionary
    """
    if setup == None:
        setup = XPConstants()
    if instrument == "bp":
        a   = setup.aBP
        b   = setup.bBP
        LSF = setup.LSFBP[:,:nmax]
    elif instrument == "rp":
        a   = setup.aRP
        b   = setup.bRP
        LSF = setup.LSFRP[:,:nmax]

    # Evaluate the Hermite functions at u0
    H   = math_tools.HermiteFunction(np.array([ (u0-b) / a ]), nmax).ravel() #
    lsf = np.dot(H, LSF.T)
    # Determine the extrema of the LSF at that point
    if order == 0:
        c1 = np.dot(setup.D1[0:(n+1), 0:(n+1)], np.concatenate((lsf, np.zeros(1))))
        l  = line_analysis.getLinesInNDeriv(c1, np.diag(np.ones(len(c1))), 
                              setup=setup, instrument=instrument)
    elif order == 2:
        c3 = np.dot(setup.D3[0:(n+3), 0:(n+3)], np.concatenate((lsf, np.zeros(3))))
        l  = line_analysis.getLinesInNDeriv(c3, np.diag(np.ones(len(c3))), 
                              setup=setup, instrument=instrument)
    idx1 = np.argmin(np.abs(l['estimLinePos']-u0))
    idx2 = np.argmin(np.abs(l['estimLinePos']-u0+D))
    idx3 = np.argmin(np.abs(l['estimLinePos']-u0-D))

    idx = np.unique(np.concatenate(([idx1], [idx2], [idx3])))
    return {'p1': l['estimLinePos'][idx[0]], 
            'p2': l['estimLinePos'][idx[1]], 
            'D': np.abs(l['estimLinePos'][idx[0]] - l['estimLinePos'][idx[1]])}


def getRsNew(u0, L, c, cov, 
             setup=None, instrument="bp", K=2, filter=None):
    """
    Computes the product of Response x SPD in Taylor approximation
    including the computation of the errors. See Vol. IX, p. 92
    
    # Inputs:
        u0  - sample position
        L   - development of the LSF in Hermite functions
        c   - the vector of coefficients of the source in Hermite functions (continuum approximation)
        cov - the covariance matrix of c
        instrument - the instrument ("bp" or "rp", needed for the Hermite basis configuration only)
        setup - the configuration of Hermite basis functions
        K   - the order of the approximation in the deconvolution (default: 2)
    # Output:
        vector with the 0th to Kth derivative at u0
    """
    if setup == None:
        setup = XPConstants()
    if instrument == "bp":
        a, b     = self.setup.aBP, self.setup.bBP
    elif instrument == "rp":
        a, b     = self.setup.aRP, self.setup.bRP

    H   = np.zeros((K+1, n+4))
    tmp = math_tools.HermiteFunction((u0-b) / a, n+4)

    if K == 0:
        S = np.dot(tmp[:, :n], c)
        covS = np.dot(tmp[:, :n], np.dot(cov, tmp[:, :n]))
    else:
        H[0, :] = tmp
        if K > 0:
            H[1, :] = np.dot(tmp, setup.D1[0:(n+4), 0:(n+4)]) / a
            if K > 1:
                H[2, :] = np.dot(tmp, setup.D2[0:(n+4), 0:(n+4)]) / (a*a)
                if K > 2:
                    H[3, :] = np.dot(tmp, setup.D3[0:(n+4), 0:(n+4)]) / (a*a*a)
                    if K > 3:
                        H[4, :] = np.dot(tmp, setup.D4[0:(n+4), 0:(n+4)]) / (a*a*a*a)

        f = np.dot(H, np.concatenate((c, np.zeros(4))))

        Cov = np.zeros((n+4, n+4))
        Cov[:n, :n] = cov
        covf = np.dot(H, np.dot(Cov, H.T))

        LM = makeLMatrix(L, u0, setup, K=K) * a

        if filter is None:
            LM = np.linalg.solve(LM)  # CHECK HERE!
        else:
            sv = np.linalg.svd(LM)
            d = sv[1] / max(sv[1])
            SI = 1 / sv[1]
            SI[np.where(d < filter)] = 0
            LM = np.dot(np.dot(sv[0], np.diag(SI)), sv[2])

        S = np.asarray(np.dot(LM, f)).flatten()
        covS = np.dot(LM, np.dot(covf, LM.T))

    return {"u":    u0, 
            "S":    S, 
            "sigS": np.sqrt(np.diag(covS)), 
            "f":    f, 
            "LH":   np.dot(LM, H)}


