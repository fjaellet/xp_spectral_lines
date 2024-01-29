######################################
#
#   xp_spectral_lines/math_tools.py
#
# Tools translated from Michael Weiler's original R code
# For details of the formalism refer to 
#     https://ui.adsabs.harvard.edu/abs/2023A%26A...671A..52W/abstract
# 
# This implementation:
#     F. Anders (ICCUB, 2023)
#
######################################
import numpy as np
import scipy
from scipy.special import hermite, factorial

def HermiteFunction(x, n, scipy=False):
    """
    Computes the first n Hermite functions on an array x
    
    # Inputs:
      x - the points where to evaluate the Hermite functions
      n - the number of Hermite functions to provide (0,...,n-1)
      
    # Returns:
      H.T - array of Hermite functions evaluated at x
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
      
    # Returns:
      quadratic correlation matrix
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
