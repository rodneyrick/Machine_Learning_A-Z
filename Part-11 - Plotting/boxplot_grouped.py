# -*- coding: utf-8 -*-

"""
The grouped boxplot it's possbile to review with this 
property is important or not to include into analysis

It's possble too to undestarding the maximum.

Other thing it's how we can understand the outliers.
"""

import seaborn as sns
sns.set(style="ticks")

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and sex
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn")
sns.despine(offset=10, trim=True)