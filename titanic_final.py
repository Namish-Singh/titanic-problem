# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:09:13 2018

@author: Namish Kaushik
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

pd.set_option('max_columns',10)
A = pd.read_csv('test.csv',index_col =0)
A['Survived'] = 0
B = pd.read_csv('train.csv',index_col =0)
df = pd.concat([B,A],sort = 'False')
df['Embarked'].fillna(value ='S', inplace = True) 

df['title1']=[list(df['Name'])[i].split(",")[1] for i in range(len(df.Name))]
df['title']=[list(df['title1'])[i].split(".")[0] for i in range(len(df.Name))]
del df['title1']
df.columns
df['Age'].fillna(df.groupby(['title'])['Age'].transform(np.mean),inplace = True)
df5 = pd.get_dummies(df['title'])
df['title'].value_counts()
df5['others'] = df5[' Col']+df5[' Don']+df5[' Dona']+df5[' Dr']+df5[' Jonkheer']+df5[' Lady']
+df5[' Mlle']+df5[' Mme']+df5[' Ms']+df5[' Rev']+df5[' Sir']+df5[' the Countess']+df5[' Capt']
# get dummies then spliiting as below then again spliyting
df6= df5[['others',' Mr',' Miss',' Master',' Mrs']]
df1= pd.get_dummies(df['Sex'],drop_first = 'True')
df2= pd.get_dummies(df['Embarked'],drop_first = 'True')
df4 = pd.get_dummies(df['Pclass'],drop_first = 'True',prefix = 'class')
df3= df.join([df1])
df3= df3.join([df2])
df3= df3.join([df4])
df3= df3.join([df6])
train= df3.head(891)

test =df3.tail(418)
x= train.loc[:,['Age', 'Fare', 'Parch',
       'SibSp', 'male', 'Q', 'S','class_2',
       'class_3','others', ' Mr', ' Miss', ' Master', ' Mrs']]
y= train['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
temp = []
temp1= []
for i in range(1,20):
    c1_tree = tree.DecisionTreeClassifier(criterion= 'gini',random_state= 1 ,max_depth =i)
    c1_tree = c1_tree.fit(x_train,y_train)
    pred = c1_tree.predict(x_test)
    pred2 = c1_tree.predict(x_train)
    c = pd.crosstab(y_test,pred)
    c2= pd.crosstab(y_train,pred2)
    temp.append((c.iloc[0,0]+c.iloc[1,1])/(c.iloc[0,0]+c.iloc[0,1]+c.iloc[1,0]+c.iloc[1,1]))
    temp1.append((c2.iloc[0,0]+c2.iloc[1,1])/(c2.iloc[0,0]+c2.iloc[0,1]+c2.iloc[1,0]+c2.iloc[1,1]))
temp  
x_val= test.loc[:,['Age', 'Fare', 'Parch', 'SibSp', 'male', 'Q', 'S', 'class_2', 'class_3',
       'others', ' Mr', ' Miss', ' Master', ' Mrs']]
c1_tree = tree.DecisionTreeClassifier(criterion= 'gini',random_state= 1 ,max_depth =3)
c1_tree = c1_tree.fit(x_train,y_train)
x_val.Fare.fillna(value = x_val.Fare.mean(),inplace = True)  
y_val = pd.Series(c1_tree.predict(x_val) )
y_val.index=test.index 
df_fin = pd.concat([test,y_val],axis =1)
df_final= df_fin.iloc[:,-1]
df_final.to_csv('titanic_decision_2.csv')

#using cross val
depth = []
depth1= []
depth2= []
for i in range(3,20):
    clf = tree.DecisionTreeClassifier(criterion= 'gini',random_state= 1 ,max_depth = i)
    scores = cross_val_score(estimator = clf, X = x_train,y=y_train , cv =5)
    scores1 = cross_val_score(estimator = clf, X = x_test,y=y_test , cv =3)
    depth1.append(scores.mean())
    depth2.append(scores1.mean())
    depth.append((i,scores.mean()))
print(depth1)



# using random forest
temp = []
temp1 = []
for i in range(1,10):
    c1_tree = RandomForestClassifier(n_estimators= 500,criterion = 'entropy',max_features = i)
    c1_tree = c1_tree.fit(x_train,y_train)
    pred = c1_tree.predict(x_test)
    c = pd.crosstab(y_test,pred)
    temp.append((c.iloc[0,0]+c.iloc[1,1])/(c.iloc[0,0]+c.iloc[0,1]+c.iloc[1,0]+c.iloc[1,1]))
print(temp) 
x_val= test.loc[:,['Age', 'Fare', 'Parch', 'SibSp', 'male', 'Q', 'S', 'class_2', 'class_3',
       'others', ' Mr', ' Miss', ' Master', ' Mrs']]
x_val.Fare.fillna(value = x_val.Fare.mean(),inplace = True) 
c1_tree = RandomForestClassifier(n_estimators= 500,max_features = 7)
c1_tree = c1_tree.fit(x_train,y_train)


y_pred = pd.Series(c1_tree.predict(x_val) )
y_pred.index=test.index 
df_fina = pd.concat([test,y_pred],axis =1)
df_finale= df_fina.iloc[:,-1]
df_finale.to_csv('titanic_random2.csv')
[0.7649253731343284, 0.7649253731343284, 0.7686567164179104, 0.7649253731343284, 
 0.7649253731343284, 
 0.7686567164179104, 0.7723880597014925, 0.7761194029850746, 0.7723880597014925]

