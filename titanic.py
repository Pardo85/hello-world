# TITANIC MODEL COMPETITION
# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

%matplotlib inline
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

# IMPORT DATASETS (training & test)
data_train = pd.read_csv('C:/Users/sardo_ae/AppData/Local/Continuum/Anaconda2/Lib/site-packages/pandas/io/data/train.csv')
data_test = pd.read_csv('C:/Users/sardo_ae/AppData/Local/Continuum/Anaconda2/Lib/site-packages/pandas/io/data/test.csv')

# FIRST LOOK AT MAIN STATISTICS
data_train.describe()


# UNIVARIATE ANALYSIS
#Sex
survived_sex = data_train[data_train['Survived']==1]['Sex'].value_counts()
dead_sex = data_train[data_train['Survived']==0]['Sex'].value_counts()
uni_sex = pd.DataFrame([survived_sex,dead_sex])
uni_sex.plot(kind='bar',stacked=True, figsize=(10,6))
# women's survival probability are higher than men's 

#Age
#As we see thanks to describe function, Age has ~20% null values.
#We have to put a good value into Age in order to keep information

# we can replace null values with the mean
data1 = data_train
data1['Age'].fillna(data1['Age'].mean(), inplace=True)

# we can replace null values with the median
data2 = data_train
data2['Age'].fillna(data2['Age'].median(), inplace=True)

# median is more robust than mean
data_train['Age'].fillna(data_train['Age'].median(), inplace=True)
data_train.describe()

figure = plt.figure(figsize=(10,6))
plt.hist([data_train[data_train['Survived']==1]['Age'],data_train[data_train['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
# passengers <10y are more likely to survive

#Fare
figure = plt.figure(figsize=(10,6))
plt.hist([data_train[data_train['Survived']==1]['Fare'],data_train[data_train['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
# passengers who paid low fares have more likely to die


#Fare - Age
plt.figure(figsize=(10,6))
ax = plt.subplot()
ax.scatter(data_train[data_train['Survived']==1]['Age'],data_train[data_train['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(data_train[data_train['Survived']==0]['Age'],data_train[data_train['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
# evidences: children,high fares - survived
# evidences: 15-50,low fares - died


# Correlation Fare - Pclass
ax = plt.subplot()
ax.set_ylabel('Average fare')
data_train.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(10,6), ax = ax)
# first class pays higher fare etc.

# Of course Fare and Pclass have a positive correlation,
# now we control if there are some particular groups
# looking at Age and Pclass

#Pclass - Age
plt.figure(figsize=(10,6))
ax = plt.subplot()
ax.scatter(data_train[data_train['Survived']==1]['Age'],data_train[data_train['Survived']==1]['Pclass'],c='green',s=40)
ax.scatter(data_train[data_train['Survived']==0]['Age'],data_train[data_train['Survived']==0]['Pclass'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Pclass')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
#Last plot shows better the correlation betwen Age e Pclass:
#<10 survived is true for the first two classes.
# >15 third class has no chances ( except few cases)   

# Embarked 
survived_embark = data_train[data_train['Survived']==1]['Embarked'].value_counts()
dead_embark = data_train[data_train['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,6))
# no evidences, except that people embarked in "C" are more likely to survive

# Variables manipulation & Features engineering

def status(feature):

    print 'La funzione',feature,': è attivata'

# new dataset: train + test
def combi_dataset():
   
    train = pd.read_csv('C:/Users/sardo_ae/AppData/Local/Continuum/Anaconda2/Lib/site-packages/pandas/io/data/train.csv')
    test = pd.read_csv('C:/Users/sardo_ae/AppData/Local/Continuum/Anaconda2/Lib/site-packages/pandas/io/data/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    
    return combined
combined = combi_dataset()

# TREATMENT1 : titles
a = combined
b = a['Name'].apply(lambda x: pd.Series(x.split(',')[1].split('.')[0].strip()))
b.rename(columns={0:'Title'},inplace=True)
b_cat = b.Title.value_counts()
# In b_cat we have the list of passenger's Titles
# Jonkheer is the dutch word for "Scudiero"

def get_titles():

    global combined
    
    # title extraction from name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # map of titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Titled",
                        "Don":        "Titled",
                        "Sir" :       "Titled",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Titled",
                        "Dona":       "Titled",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Titled"

                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
get_titles()
# Now we control the variable 'Title' to be sure that the
# dictionary worked correctly

c_cat = combined.Title.value_counts()

# TREATMENT1 : Names
# we create a dummy variable for each value of the variable Title
# after the feature 'treat names' we have for example the new dummy
# variable Title_Miss with value eq 1 if Title = 'Miss' and
# value eq 0 in all the othere cases.
def treat_names():
    
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
   
    status('names')
    
treat_names()

# TREATMENT2 : Age
# Before treating the Age(remembering the 20% of Nulls),
# we may control if there is a correlation between Pclass
# and Title, we can imagine that there is a high probability
# that a 'Titled' passenger would not take a 3class ticket
ax = plt.subplot()
ax.set_ylabel('Average fare')
combined.groupby('Title').mean()['Fare'].plot(kind='bar',figsize=(10,6), ax = ax)

# We saw that Title could be linked with fare (so Pclass), so we 
# put into Age the age of a the particular group considering Sex,Pclass & Title

def treat_age():
    
    global combined
    
    # a function that fills the missing values of the Age variable
    
    def nullAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    
    combined.Age = combined.apply(lambda r : nullAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
    status('age')
treat_age()

combined.info()

# Now we can remove the title variable
combined.drop('Title',axis=1,inplace=True)

# TREATMENT3 : Fares  
combined.describe()
# There is 1 missing value for the variable Fare, we can put the mean
def treat_fares():
    
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.Fare.mean(),inplace=True)
    
    status('fare')     
        
treat_fares()

# TREATMENT4 : Pclass
def treat_pclass():
    
    global combined
    # format in 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    combined = pd.concat([combined,pclass_dummies],axis=1)
    combined.drop('Pclass',axis=1,inplace=True)
    
    status('pclass')

treat_pclass()
# TREATMENT5 : Sex

def treat_sex():
    
    global combined
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    
    status('sex')
treat_sex()

# TREATMENT6 : Embarked 
# There are 2 missing values for the variable Embarked, we can put the mode
def treat_embarked():
    
    global combined
    combined.Embarked.fillna('S',inplace=True)   
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)
    
    status('embarked') 

treat_embarked()

# TREATMENT7 : Family size
def treat_family_size():
    
    global combined
    # passenger + parents/children & spouse/siblings
    combined['Family_Size'] = combined['Parch'] + combined['SibSp'] + 1
    combined['Singleton'] = combined['Family_Size'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['Family_Size'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['Family_Size'].map(lambda s : 1 if 5<=s else 0)
    
    status('family')
    
treat_family_size()

# SELECT ALL FEATURES - PassengerId has no utility
def scale_all_features():
    
    global combined
    
    features = list(combined.columns)
    features.remove('PassengerId')
    features.remove('Cabin')
    features.remove('Ticket')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
    
    print 'Features scaled successfully !'
    
scale_all_features()  

#MODELLING

# Useful libraries
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# TRAIN - TEST - TARGET
# We separe combined in train & test again. Targets has only the Survived variable
combined2 = combined
combined2.drop('Cabin',axis=1,inplace=True)
combined2.drop('Ticket',axis=1,inplace=True)

def ttt():
    global combined2
    

    
    train0 = pd.read_csv('C:/Users/sardo_ae/AppData/Local/Continuum/Anaconda2/Lib/site-packages/pandas/io/data/train.csv')
    
    targets = train0.Survived
    train = combined2.ix[0:890]
    test = combined2.ix[891:]
    
    return train,test,targets

train,test,targets = ttt()

X_train = train.drop("PassengerId", axis=1)
Y_train = targets
X_test  = test.drop("PassengerId", axis=1).copy()
# we look at the dimensions of the 3 datasets
X_train.shape, Y_train.shape, X_test.shape

# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# KNeighbors 
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Decision Tree
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(X_train, targets)

features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort(['importance'],ascending=False)
importance_list = features.sort(['importance'],ascending=False)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)


# We choose the Random Forest score as long as Decision's tree has the
# habit of overfitting to the training dataset.

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'KNN', 
              'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree','Random Forest' ],
    'Score': [acc_log,acc_svc,acc_knn,acc_gaussian,acc_perceptron,
              acc_linear_svc,acc_decision_tree,acc_random_forest]})
models.sort_values(by='Score', ascending=False)


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
        
  
submission.to_csv('C:/Users/sardo_ae/AppData/Local/Continuum/Anaconda2/submission.csv',index= False )    
