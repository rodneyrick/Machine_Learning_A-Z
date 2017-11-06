# -*- coding: utf-8 -*-

"""
The grouped barplot it's possbile to review with this 
property is important or not to include into analysis

It's possble too to undestarding the maximun
"""

import seaborn as sns
sns.set(style="whitegrid")

# Load the example Titanic dataset
titanic = sns.load_dataset("titanic")

# Draw a nested barplot to show survival for class and sex
g = sns.factorplot(x="class", y="survived", hue="sex", data=titanic,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")