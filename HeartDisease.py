
#importantion de modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import *
from sklearn.utils import resample
from sklearn.compose import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
""" Data Viz """
import matplotlib.pyplot as plt
import seaborn as sns


data_file_pathname = "C:\\Users\\kokou\\Desktop\\Deploiement\\heart_2020_cleaned.csv"


# reading tsv files
df = pd.read_csv(data_file_pathname, sep=",", header=0)


# I-Analyse exploratoire des données

def summary(df, pred=None):
    obs = df.shape[0]
    Types = df.dtypes
    Counts = df.apply(lambda x: x.count())
    Min = df.min()
    Max = df.max()
    Uniques = df.apply(lambda x: x.unique().shape[0])
    Nulls = df.apply(lambda x: x.isnull().sum())
    print('Data shape:', df.shape)

    if pred is None:
        cols = ['Types', 'Counts', 'Uniques', 'Nulls', 'Min', 'Max']
        str = pd.concat([Types, Counts, Uniques, Nulls, Min, Max], axis = 1, sort=True)

    str.columns = cols
    print('___________________________\nData Types:')
    print(str.Types.value_counts())
    print('___________________________')
    return str
print(summary(df).sort_values(by='Nulls', ascending=False))

# Les types de données

Typ = pd.DataFrame({"types de données": df.dtypes})
t = Typ["types de données"].value_counts() / len(Typ)
plt.subplots(facecolor='lightgray')
t.plot.pie(autopct='%1.1f%%')
plt.show()

print(df.tail())
print(df.info())
print(df.describe())
print(df.columns)

# Vérifions l'existance des valeurs manquantes
print(df.isnull().sum())


# Séparons les colonnes par types de données
# 1- données catégoricielles
data_cat = df[['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking',
'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth',
'Asthma', 'KidneyDisease', 'SkinCancer']]

# 2- données numériques
data_num = df[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']]


import matplotlib.pyplot as plt
for i in data_num.columns:
    plt.hist(data_num[i],bins=9, align='right', color='blue', edgecolor='black')
    plt.show()

# 3 - Vérifions l'existence de valeurs abérrantes pour data_num
for i in data_num.columns:
 sns.boxplot(x=data_num[i])
 plt.show()

# Nous allons construire des graphes pour des données catégoricielles VS variable cible :HeartDisease

# Sex VS HeartDisease
plt.figure(figsize=(15,6))
sns.countplot('Sex',hue='HeartDisease', data = df)
plt.xticks(rotation = 0)
plt.show()

# Smoking VS HeartDisease
plt.figure(figsize=(15,6))
sns.countplot('Smoking',hue='HeartDisease',data=df)
plt.xticks(rotation = 0)
plt.show()

# Stroke VS HeartDisease
plt.figure(figsize=(15,6))
sns.countplot('Stroke',hue='HeartDisease',data=df)
plt.xticks(rotation = 0)
plt.show()

# AgeCategory VS HeartDisease
plt.figure(figsize=(20,6))
sns.countplot('AgeCategory',hue='HeartDisease',data=df)
plt.xticks(rotation = 0)
plt.show()

# Race VS HeartDisease
plt.figure(figsize=(20,6))
sns.countplot('Race',hue='HeartDisease',data=df)
plt.xticks(rotation = 0)
plt.show()

# GenHealth VS HeartDisease
plt.figure(figsize=(16,6))
sns.countplot('GenHealth',hue='HeartDisease',data=df)
plt.xticks(rotation = 0)
plt.show()

# PhysicalHealth VS HeartDisease
plt.figure(figsize=(10,6))
sns.countplot('PhysicalHealth',hue='Sex',data=df)
plt.xticks(rotation = 0)
plt.show()

# MentalHealth VS HeartDisease
plt.figure(figsize=(10,6))
sns.countplot('MentalHealth',hue='Sex',data=df)
plt.xticks(rotation = 0)
plt.show()


# Tranformationdes variables catégoricielles en numériquesen utilisant l'encoder
for i in data_cat.columns:
    le=LabelEncoder()
    label=le.fit_transform(data_cat[i])
    data_cat[i]=label

# Nouvelle dataset après conversion
full_pipeline = pd.concat([data_cat,data_num],axis=1)

# Affichage de la nouvelle base de données
print(full_pipeline.head())

# Affichage de la corrélation entre les variables
print(full_pipeline.corr())

plt.figure(figsize=(20,8))
sns.heatmap(full_pipeline.corr())
plt.show()

full_pipeline.drop(['Race','BMI'],axis=1,inplace=True)

# Affichage
full_pipeline.head()

# separation entre les données d'entrainement(tba) et test
tba, test_set = train_test_split(full_pipeline, test_size=0.3, random_state=42)
tba.shape

tba.head()
tba.head()
# Test le pipeline
from sklearn.pipeline import Pipeline
leData = X_train
pd.DataFrame(leData).head()


#transformationn de la sortie HeartDisease en sortie binaire.
tba_ss = tba.drop(["HeartDisease"], axis=1)
sortie=tba["HeartDisease"]
tba=tba_ss.assign(HeartDisease=sortie)
tba.head(10)

#il faut séparer les entrées/sorties
X_train=train_set.drop(['HeartDisease'], axis=1)
y_train=train_set['HeartDisease']

#échantillonnage stratifié par rapport a la tba
train_set = resample(tba, n_samples=4000, replace=False,stratify=tba['HeartDisease'], random_state=5)
#calcul de la proportion des indvidus par rapport a la classe de sortie HeartDisease dans le jeu d'entrainement.
#Il y a autant d'individus par classe
print(train_set['HeartDisease'].value_counts()/len(train_set))
train_set.shape


#exemple d'utilisation d'un arbre de décision
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier(criterion="entropy")
clf.fit(leData, y_train)

# Test le pipeline pour le jeu de test
X_test=test_set.drop(['HeartDisease'], axis=1)
y_test=test_set['HeartDisease']
X_test.head()


#test
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
print('Results on the test set:')
print(classification_report(y_test, y_pred))

# le taux de détection est de l'ordre de :


#exemple d'utilisation de la régression logistic
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Results on the test set:')
print(classification_report(y_test, y_pred))


#exemple d'utilisation de l'ARBRE DE DÉCISION'
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
clf= DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)
cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")


#k=3
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(clf, X_train, y_train, cv=3)
from sklearn import metrics
print(metrics.confusion_matrix(y_train,y_pred))

# le taux de détection est de l'ordre de :
print(metrics.classification_report(y_train, y_pred))
# Optimiser les hyper-paramètres en utilisant la recherche par quadrillage –GridSearchCV

from sklearn.model_selection import GridSearchCV
params = {'max_depth': list(range(2, 20)), 'min_samples_split': [10, 15, 20]}

grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42),
                              params, n_jobs=-1, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV
params = {'max_depth': list(range(2, 20)), 'min_samples_split': [10, 15, 20]}
#le meilleur estimateur (modele) trouvé est:
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42),
                              params, n_jobs=-1, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)

# testons le meilleur moèle de L'arbre de décission:
from sklearn.model_selection import GridSearchCV
params = {'max_depth':list(range(0, 2)), 'min_samples_split':[1,10]}

grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42),
                            params, n_jobs=-1, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)

from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(clf, X_train, y_train, cv=3)
from sklearn import metrics
print(metrics.confusion_matrix(y_train,y_pred))

# le taux de détection est de l'ordre de :
print(metrics.classification_report(y_train, y_pred))

#test
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
print('Results on the test set:')
print(classification_report(y_test, y_pred))
# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

GaussianNB()

# Nous avaons obtenu les memes résultats avant l'optimisation des paramètres.

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred))


# making predictions on the testing set
y_pred = gnb.predict(X_test)

# Predict on test set
pred_y = gbr.predict(X_test)

# gradient boosting
# Instantiate Gradient Boosting Regressor
# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

gbr = GradientBoostingRegressor(n_estimators = 200, max_depth = 1, random_state = 42)
# Fit to training set
gbr.fit(X_train, y_train)

# test set RMSE
test_rmse = MSE(y_test, y_pred)** (1 / 2)

# Print rmse
print('RMSE test set: {:.2f}'.format(test_rmse))

# Bilan
from prettytable import PrettyTable
table = PrettyTable(['Séparation ','Model Names', 'Accuracy'])
table.align = 'l'
table.add_row(['80/20', 'Logistic Regression', '0.91'])
table.add_row(['80/20', 'DecisionTreeClassifier', '0.86'])
table.add_row(['80/20', 'Gaussian Naive Bayes', '0.84'])
table.add_row(['80/20', 'Gradient Boosting', '0.40'])
print(table)














