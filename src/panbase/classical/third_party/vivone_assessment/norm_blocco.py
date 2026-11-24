# SPDX-License-Identifier: LicenseRef-Vivone-PyToolbox
#
# Vendored for the PanBase release (2026) from the Python assessment toolbox of G. Vivone
# ("Pansharpening_Toolbox_Assessment_Python", as redistributed in https://github.com/yotick/AWFLN @ 210232c),
# file norm_blocco.py (upstream md5 848ad69b5dd8e56296326fecceaa5a30).  Terms: Vivone's licence (LICENSE.txt in this folder).  NOT covered by the Apache-2.0 licence of the rest of
# the PanBase repository.  The original notices below are kept unchanged.
# Modifications: none.
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

""" Q2n aux. function """
import numpy as np

def norm_blocco(x):
    
    a = np.mean(x)
    c = np.std(x, ddof=1)
    
    if (c==0):
    	c = 2.2204 * 10**(-16)
    
    y = ((x - a)/c) + 1
    
    return y, a, c
