import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from datetime import datetime


df=pd.read_csv("train.csv")
df1=pd.read_csv("test.csv")
#df.head(5)
X=df.iloc[:,1:11].values
X = np.delete(X,[5,7,8,9], 1)
Y=df.iloc[:,0]
Xtest=df1.iloc[:,0:10].values
Xtest = np.delete(Xtest, [5,7,8,9], 1)
AccId=df1.iloc[:,10].values



severity={'Highly_Fatal_And_Damaging':4,
 'Significant_Damage_And_Serious_Injuries':3,
 'Minor_Damage_And_Injuries':2,
 'Significant_Damage_And_Fatalities':1}

severity2={4:'Highly_Fatal_And_Damaging',
 3:'Significant_Damage_And_Serious_Injuries',
 2:'Minor_Damage_And_Injuries',
 1:'Significant_Damage_And_Fatalities'}

Y=Y.map(severity).as_matrix(columns=None)

'''from sklearn.preprocessing import LabelEncoder
labelEn=LabelEncoder()
labelEn.fit([
 'Significant_Damage_And_Serious_Injuries',
 'Minor_Damage_And_Injuries',
 'Significant_Damage_And_Fatalities','Highly_Fatal_And_Damaging'])

Y=labelEn.transform(Y)'''
#sn.residplot(df["Safety_Score"],Y)

#sn.boxplot(x='Control_Metric',y='Severity',data=df)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.001,random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=9,n_estimators=990,learning_rate=0.05,\
                           gamma=0.15,reg_alpha=0.01,reg_lambda=0.75,min_child_weight=5,colsample_bytree=0.7,\
                          n_jobs=-1)
classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_test)

pred=pd.Series(classifier.predict(Xtest)).map(severity2).as_matrix(columns=None)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10,n_jobs=-1)
accuracies.mean()
accuracies.std()

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

from sklearn.model_selection import RandomizedSearchCV
parameter={'n_estimators':[200,400,600,700,800,900,1000],'min_child_weight': [2,3,4,4.5,5,6,7],\
            'learning_rate':[0.10,0.15,0.17,0.20,0.23],'max_depth':[6,7,8,9,10,12,13],\
            'gamma':[0.0,0.1,0.2,0.3],'colsample_bytree': [0.7,0.8,0.9,0.95,1],'reg_alpha':[0, 0.001,0.002, 0.005, 0.01, 0.05],\
            'reg_lambda':[0.70,0.75,0.8,0.85,0.9,1]}
grid=RandomizedSearchCV(estimator=classifier,param_distributions=parameter,n_iter=10,scoring='accuracy',cv=10,n_jobs=-1,verbose=3)
start_time = timer(None)    
grid=grid.fit(X_train,Y_train)
timer(start_time)
best_score=grid.best_score_
best_para=grid.best_params_

df2=pd.concat([pd.DataFrame(AccId,columns=["Accident_ID"]),pd.DataFrame(pred,columns=["Severity"])],axis=1)

df2.to_csv('pre11Final.csv',index=False)
