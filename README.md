# xp_spectral_lines - Tools to analyse lines in Gaia XP spectra

The Gaia XP spectra are very low-resolution ($R\sim25$), but typically very high signal-to-noise spectra taken by [Gaia](https://www.cosmos.esa.int/web/gaia/home)'s blue and red photometers. 
They were published for the first time as part of [Gaia DR3](https://www.cosmos.esa.int/web/gaia/data-release-3) and described in detail in [De Angeli et al. (2023)](https://doi.org/10.1051/0004-6361/202243680).

And already [before that](https://cosmos.esa.int/web/gaia/iow_20200812), it was clear that these spectra can possibly be used to search for spectral lines and/or broad absorption bands:

![](https://cosmos.esa.int/documents/29201/239681/IoW20200804_RPspectra_SpT_cycle3_b.png/dffe25a8-8860-bf8f-be11-b5680642e721?t=1595591435624)

## Implementing the formalism developed in [Weiler et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...671A..52W/abstract)

Michael Weiler has developed a formalism that allows to infer the properties of spectral lines directly from the XP coefficients (the format the XP spectra are actually stored in).
This repository aims at implementing his formalism in python (by translating from his original R code). 

CURRENTLY UNDER DEVELOPMENT... please stand by.
