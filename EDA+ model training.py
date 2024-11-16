import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc

from sklearn.ensemble import RandomForestClassifier,VotingClassifier,GradientBoostingClassifier

import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('data.csv')

g = sns.pairplot(data, hue='class', markers=['o','s'], kind='reg',
                 plot_kws={'line_kws': {'color' : 'blue','lw':.8},
                 'scatter_kws':{'alpha':.3,'s':3}},height=1.5)

fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(18,6))
sns.heatmap(data.drop(columns=['class']).corr(), cmap ="YlGnBu", annot=True, ax=ax)
ax.set_title("Correlation Heat Map", fontsize = 20)

plt.show()
