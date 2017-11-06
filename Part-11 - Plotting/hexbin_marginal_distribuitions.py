#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hexbin plot show how the heatmap into wide data and
how the distribuition of marginal of this data

the focus is in plots an hexagonal bin from a set of 
positions. Can plot the number of occurrences in 
each bin (hexagon) or give a weight to each 
occurrence
"""

import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
sns.set(style="ticks")

rs = np.random.RandomState(11)
x = rs.gamma(2, size=1000)
y = -.5 * x + rs.normal(size=1000)

sns.jointplot(x, y, kind="hex", stat_func=kendalltau, color="#4CB391")