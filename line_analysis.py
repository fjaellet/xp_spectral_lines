######################################
#
#   xp_spectral_lines/line_analysis_tools.py
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
from scipy.linalg import eigvals, svd
import scipy.stats as stats
import pickle

import math_tools, spectrum_tools

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
        setup = spectrum_tools.XPConstants()

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
    Hermite = np.matmul(np.matmul(math_tools.HermiteFunction(realRoots, n+2), 
                                  setup.D1[0:n+2, 0:n+2]), np.concatenate((coef, [0])))

    # compute the condition numbers of the roots if requested:
    if conditioning and nReal > 0:
        # left eigenvectors:
        LE  = math_tools.HermiteFunction(realRoots, nReal)
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
        J = -1/Hermite * math_tools.HermiteFunction(realRoots, n=n+1)
    else:
        J = -np.diag(1/Hermite) @ math_tools.HermiteFunction(realRoots, n=n+1)
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
        setup = spectrum_tools.XPConstants()

    coef1 = np.matmul(setup.D1[0:n+1, 0:n+1], np.concatenate([coef, [0]]))
    cov1  = np.matmul(setup.D1[0:n+1, 0:n+1], 
                      np.vstack([np.hstack([cov, np.zeros((n, 1))]), 
                                 np.zeros((1, n+1))])) @ np.transpose(setup.D1[0:n+1, 0:n+1])
    cov2  = np.zeros((n+2, n+2))
    cov2[:n, :n] = cov

    roots = getRoots(coef1, cov1, setup=setup, conditioning=conditioning)

    # computing the values and errors of the second derivative at the roots:
    M = np.matmul(math_tools.HermiteFunction(roots['realRoots'], n+2), setup.D2[0:n+2, 0:n+2])
    v = np.matmul(M, np.concatenate([coef, [0, 0]]))
    E = np.matmul(np.matmul(M, cov2), np.transpose(M))

    kind = np.repeat("minimum", len(roots['realRoots']))
    kind[np.where(v < 0)] = "maximum"

    return {'location':   roots['realRoots'], 
            'error':      roots['sigma'], 
            'cov':        roots['cov'], 
            'condition':  roots['condition'], 
            'kind':       kind,
            'secondDerivativeAtRoots':        v, 
            'ErrorOnSecondDerivativeAtRoots': np.sqrt(np.diag(E)), 
            'CovForSecondDerivatives':        E, 
            'roots':      roots}

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
        setup = spectrum_tools.XPConstants()

    coef1 = setup.D2[:(n+2), :(n+2)].dot(np.append(coef, [0, 0]))
    cov1  = setup.D2[:(n+2), :(n+2)].dot(
            np.block([[cov, np.zeros((n, 2))], [np.zeros((2, n+2))]])).dot(
            setup.D2[:(n+2), :(n+2)].T)
    cov2  = np.zeros((n+3, n+3))
    cov2[:n, :n] = cov

    roots = getRoots(coef1, cov1, setup=setup, conditioning=conditioning)

    # computing the values and errors of the third derivative at the roots:
    M = math_tools.HermiteFunction(roots["realRoots"], n+3).dot(setup.D3[:(n+3), :(n+3)])
    v = M.dot(np.append(coef, [0, 0, 0]))
    E = M.dot(cov2).dot(M.T)

    kind = np.repeat("increasing", len(roots["realRoots"]))
    kind[v < 0] = "decreasing"

    return {"location":   roots["realRoots"],
            "error":      roots["sigma"],
            "covariance": roots["cov"],
            "condition":  roots["condition"],
            "kind":       kind,
            "thirdDerivativeAtRoots":        v,
            "ErrorOnThirdDerivativeAtRoots": np.sqrt(np.diag(E)),
            "CovForThirdDerivatives":        E,
            "roots":      roots}


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
        setup = spectrum_tools.XPConstants()
    if instrument == "bp":
        a = setup.aBP
        b = setup.bBP
    elif instrument == "rp":
        a = setup.aRP
        b = setup.bRP
    else:
        print("Warning: You are using getLinesInNDeriv without specifying the instrument - no scaling is applied")
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
    widthsError = [np.sqrt(inf["covariance"][np.where(pinf == min(pinf[pi > pinf]))[0][0], 
                                             np.where(pinf == min(pinf[pi > pinf]))[0][0]] + 
                           inf["covariance"][np.where(pinf == max(pinf[pi < pinf]))[0][0], 
                                             np.where(pinf == max(pinf[pi < pinf]))[0][0]] -
                           2 * inf["covariance"][np.where(pinf == min(pinf[pi > pinf]))[0][0], 
                                                 np.where(pinf == max(pinf[pi < pinf]))[0][0]])
                   for pi in p]

    res = {"N":              N, 
           "estimLinePos":   p, 
           "estimLineErr":   e["error"] * a, 
           "SNonSecondDerivative": x, 
           "estimSignif":    signif,
           "lineWidths":     widths, 
           "lineWidthsError":                widthsError, 
           "secondDerivativeAtRoots":        e["secondDerivativeAtRoots"] / a**(N+2),
           'ErrorOnSecondDerivativeAtRoots': e['ErrorOnSecondDerivativeAtRoots']/a**(N+2),
           'CovForSecondDerivatives':        e['CovForSecondDerivatives']/a**(2*N+4),
           'kind':           e['kind'],
           'estimInfPos':    pinf, 
           'estimInfErr':    inf['error']*a, 
           'estimInfCov':    inf['covariance']*a*a, 
           'estimInfSignif': signifInf, 
           'infKind':        inf['kind'],
           'instrument':     instrument}
    return res



def analyseLine(f, P, wavelength, setup, HI, V, LSF, K=2, dispShift=0):
    """
    Derives the equivalent widths for a narrow line

    # Inputs:
        res      - dictionary containing the extrema of the spectrum
        lambda   - the wavelength of the line to look for, in nm
        LSF      - if not provided, an LSF from the disc is read
        LSFwidth - the min and max value between which the line has to be

    """
    # Select the instrument to use:
    if wavelength < 650:
        instrument = "bp"
    else:
        instrument = "rp"

    im = readDR3InstrumentModel(XP)
    uL = im.disp(wavelength) + dispShift  # nominal position of the line in pseudo-wavelength

    if P[0]["N"] == 2:
        higherOrder = True
    else:
        higherOrder = False

    if higherOrder:
        w = getLSFWidth(LSF, uL, XP, setup, order=2)
    else:
        w = getLSFWidth(LSF, uL, XP, setup, order=0)

    LSFwidth = sorted([w["p1"], w["p2"]])

    n = len(P)  # the number of spectra in the input file

    result = []

    for i in range(n):
        spectrum = extractSpectrumFromSingeFile(f, i, V=V)
        if XP == "BP":
            coef = spectrum["bp"]["coef"]
            idx = np.where((P[i]["bp"]["estimLinePos"] > LSFwidth[0]) & (P[i]["bp"]["estimLinePos"] < LSFwidth[1]))[0]
        else:
            coef = spectrum["rp"]["coef"]
            idx = np.where((P[i]["rp"]["estimLinePos"] > LSFwidth[0]) & (P[i]["rp"]["estimLinePos"] < LSFwidth[1]))[0]

        hit = len(idx)

        if hit == 1:
            if higherOrder:
                W = getNarrowLineEquivalentWidthSecondOrder(spectrum, P[i], idx, XP, setup, 
                                                            LSF, im.dispInv, HermiteTransformationMatrices, 
                                                            HI, K=K, uLine=uL)
            else:
                W = getNarrowLineEquivalentWidth(spectrum, P[i], idx, XP, setup, LSF, im.dispInv, 
                                                 HermiteTransformationMatrices, HI, K=K, uLine=uL)
            coefCont = np.asarray(W["coefCont"], dtype=float)
        else:  
            # get upper limit:
            tmp = getNarrowLineUpperLimit(spectrum, uL, XP, setup, 
                                          LSF, im.dispInv, HermiteTransformationMatrices, HI, K=K, Q=7.709)
            W = {"W": 0, "errW": tmp["upperLimit"]}
            coefCont = np.nan

        result.append({"source_id":      f["source_id"][i], 
                       "W":              W["W"], 
                       "errW":           W["errW"], 
                       "coef":           coef, 
                       "coefCont":       coefCont, 
                       "index":          idx, 
                       "ExtremaInRange": hit, 
                       "dispShift":      dispShift, 
                       "LSFwidth":       w["D"]})

    return result
