# EXP 9: Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary packages.
 
2.Read the given csv file and display the few contents of the data.

3.Assign the features for x and y respectively.

4.Split the x and y sets into train and test sets.

5.Convert the Alphabetical data to numeric using CountVectorizer.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.
 

## Program:
Program to implement the SVM For Spam Mail Detection..

Developed by:R.BRINDHA

RegisterNumber:212222230023  
```
import chardet
file='/content/spam (1).csv'
with open(file, 'rb') as rawdata:
     print('Result output')
    result = chardet.detect(rawdata.read(10000))
result

import pandas as pd
data=pd.read_csv("/content/spam (1).csv",encoding="windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
print("y_pred")
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy")
accuracy
```


## Output:
![ml 9 0](https://github.com/Brindha77/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889143/63a7309d-0412-46ed-8290-77d342199dde)

![ml 9 i](https://github.com/Brindha77/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889143/53442df8-1fbe-4fc7-a2ba-e5b4d9a345b5)

![ml 9 ii](https://github.com/Brindha77/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889143/eaab1122-5271-410c-b7df-76e66ba68d49)

![ml 9 iii](https://github.com/Brindha77/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889143/3ed57365-a128-4ca6-ba1b-d7c7264e05ae)

![ml 9 iv](https://github.com/Brindha77/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889143/d7a4e10e-bed8-4cbf-8dd0-200a7f85a5c5)

![ml 9 v](https://github.com/Brindha77/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889143/156c27f2-d718-4206-9621-ab6ed999c0be)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
