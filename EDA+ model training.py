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
#  plt.show()
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(18,6))
sns.heatmap(data.drop(columns=['class']).corr(), cmap ="YlGnBu", annot=True, ax=ax)
ax.set_title("Correlation Heat Map", fontsize = 20)

# plt.show()

def plot_categorical(dataset, categorical_feature, rows, cols, kind):
    fig,ax = plt.subplots(nrows=2,ncols=4, figsize=(12,8))
    features = dataset.columns.values[:-1]
    
    counter = 0
    dataset['class'].value_counts().plot.bar(ax = ax[0,0])
    dataset['class'].value_counts().plot.pie(ax = ax[0,1], autopct='%1.1f%%')
    
    for i in range(rows):
        for j in range(cols):
            feature = features[counter]
            if (i == 0 and j == 0) or (i == 0 and j == 1): continue
            else:
                if kind == 'swarm':
                    sns.swarmplot(data=dataset, 
                                  x=categorical_feature, 
                                  y=feature,
                                  hue=categorical_feature, 
                                  ax=ax[i,j])
                if kind == 'box':
                    sns.boxplot(data=dataset,
                                x=categorical_feature,
                                y=feature,
                                hue=categorical_feature,
                                ax=ax[i,j])
                counter += 1
                if counter >= len(features): break
    plt.tight_layout()
    plt.show()
# Before Outlier 
# plot_categorical(dataset=data, categorical_feature='class', rows=2, cols=4, kind='swarm')

# Dealing with Outliers
# First Option
def cap_outliers(data):
    
    data_capped = data.copy()
    numeric_columns = data_capped.select_dtypes(include = [float,int]).columns
    
    for column in numeric_columns:
        Q1 = data_capped[column].quantile(0.25)
        Q3 = data_capped[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR 
        
        data_capped[column] = data_capped[column].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
        
    return data_capped

data_capped = cap_outliers(data)
# After Outlier
plot_categorical(dataset=data_capped, categorical_feature='class', rows=2, cols=4, kind='swarm')

# Data transformation
data['class'] = [1 if each == 'Abnormal' else 0 for each in data['class']]

y = data['class'].values
x = data.drop(['class'],axis=1)

# Standartizion,Normalization

scaler = StandardScaler()
data_scaled = scaler.fit_transform(x)
columns_data = x.columns
scaled_x = pd.DataFrame(data_scaled, columns=columns_data)
x = scaled_x

# train

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.2,random_state=42)

# training models

scores = dict()

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
print('Logistic Regression Score: ',logreg.score(x_test, y_test))
scores['logreg'] = logreg.score(x_test, y_test)

svm = SVC(probability=True) # Probability parameter is important for voting classifier (soft).
svm.fit(x_train,y_train)
print('Support Vector Machine (SVM) Score:', svm.score(x_test, y_test))
scores['svm'] = svm.score(x_test, y_test)

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
print('Decision Tree Classifier Score:', decisiontree.score(x_test, y_test))
scores['decisiontree'] = decisiontree.score(x_test, y_test)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print('K-Nearest Neighbors Score:', knn.score(x_test, y_test))
scores['knn'] = knn.score(x_test, y_test)

naivebayes = GaussianNB()
naivebayes.fit(x_train, y_train)
print('Naive Bayes Score:', naivebayes.score(x_test, y_test))
scores['naivebayes'] = naivebayes.score(x_test, y_test)

catboost = CatBoostClassifier(verbose=0)
catboost.fit(x_train, y_train)
print('Cat Boost Classifier Score:', catboost.score(x_test, y_test))
scores['catboost'] = catboost.score(x_test, y_test)

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
print('Random Forest Classifier Score:', randomforest.score(x_test, y_test))
scores['randomforest'] = randomforest.score(x_test, y_test)

gradientboosting = GradientBoostingClassifier()
gradientboosting.fit(x_train, y_train)
print('Gradient Boosting Classifier Score:', gradientboosting.score(x_test, y_test))
scores['gradientboosting'] = gradientboosting.score(x_test, y_test)

voting_clf = VotingClassifier(estimators=[('logreg', logreg), ('svm', svm), ('decisiontree', decisiontree),
                                          ('knn', knn), ('naivebayes', naivebayes), ('catboost', catboost),
                                          ('randomforest', randomforest), ('gradientboosting', gradientboosting)],
                                          voting='soft')
voting_clf.fit(x_train, y_train)
print('Train Voting Classifier Score:', voting_clf.score(x_test, y_test))
print('Test Voting Classifier Score:', voting_clf.score(x_test, y_test))

top_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5])
print("Top 5 score:")
for model, score in top_scores.items(): print(" *", f"{model}: {score}")





                
    
                

    
