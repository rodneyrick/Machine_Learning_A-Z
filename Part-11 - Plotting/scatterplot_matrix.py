# -*- coding: utf-8 -*-

"""
Describe the observations for each categoric data into 
same column.
"""

import seaborn as sns
sns.set()

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")