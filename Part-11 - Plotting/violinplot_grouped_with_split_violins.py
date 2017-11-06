# -*- coding: utf-8 -*-

"""
The Violinplot coming to understand about the dimension of data
from a wide-form dataset.

This gives us a rough comparison of the distribution in each group,
but sometimes it’s nice to visualize the kernel density estimates instead.
"""

import seaborn as sns
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True,
               inner="quart", palette={"Male": "b", "Female": "y"})
sns.despine(left=True)