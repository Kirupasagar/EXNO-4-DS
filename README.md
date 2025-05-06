# EXNO:4-DS
REGISTER NUMBEER: 212224230126
NAME : KIRUPASAGAR.S
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()
```
![439185130-d0e6ca6e-b991-4984-b560-fd0d60f69def](https://github.com/user-attachments/assets/7cb2aa4f-b0a1-46b6-bcba-cf5e5311d0c1)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![439186699-642c250e-107a-4d76-91d3-fdf9756374dd](https://github.com/user-attachments/assets/87744fa5-5c26-4b73-b153-bb680c1afb63)
```
df.dropna()
```
![439186816-70c9248e-46a1-4622-b0fb-a3abe35477a0](https://github.com/user-attachments/assets/0dd15f6b-6cdb-420e-8784-b2d1d4be7435)
```
max_values = np.max(np.abs(df[['Height','Weight']]),axis=0)
max_values
```
![439186950-5b28f6a3-2b99-4d60-b7b4-26efd5f2e59b](https://github.com/user-attachments/assets/660ee598-c80f-4781-bb49-ad24c70b0f13)
```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![439187154-a1859fb4-64c8-4182-8be6-52d5b7aa5017](https://github.com/user-attachments/assets/fa5c05ca-bea7-43d0-b2aa-40b42a83255c)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![439187250-f97ae87b-8f3f-47e5-b165-2516af538016](https://github.com/user-attachments/assets/bd7abd95-7226-424f-9e6e-76d273afe92c)
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![439187379-a71851ca-a547-4fc2-a430-a6a553553cdd](https://github.com/user-attachments/assets/6a0b6a4e-b619-4c29-81a5-94eb3f2db633)
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
![439187499-82ebe9c4-635a-4cce-bfec-8b465abfd059](https://github.com/user-attachments/assets/844a1c44-c444-4b90-bb2e-e94e22d347f2)
```
df=pd.read_csv("income(1) (1).csv")
df.info()
```
![439187601-7a596228-6b6c-4188-b584-87af2f68b202](https://github.com/user-attachments/assets/33f317d9-a5d7-4321-bd6c-82177957b427)
```
df_nullvalues_sum=df.isnull().sum()
df_nullvalues_sum
```
![439187852-652e793b-87fe-48d9-a507-4f580a8f6efd](https://github.com/user-attachments/assets/3f5884de-4565-4f0e-b96f-489d060ef505)
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![439187954-fff7b85e-e097-427d-ab95-05087704c9b8](https://github.com/user-attachments/assets/60aca3a5-4b8b-489a-af62-b94f47f9e24b)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![439188084-da87ed36-852c-48cc-be3b-d89e6ef38d37](https://github.com/user-attachments/assets/a80f996d-45cb-4a69-8ebb-a1b2e2e93c25)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![439189384-a91c0bf2-3e3f-4c1f-acc5-81fa1002a765](https://github.com/user-attachments/assets/d4c60aad-5332-40cf-ac5c-3081750e78a6)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![439189449-3fa994d5-3049-437c-86c5-904b6727a3f1](https://github.com/user-attachments/assets/81ebb8f5-3e35-43df-8894-82883cfbf927)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![439188557-63023e45-8180-4696-8672-9bfb1f9c9157](https://github.com/user-attachments/assets/41cda52f-1281-48d6-9fba-214139a19ae4)
```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![439188656-285d2521-d68f-42c9-8cd6-8e141c23e337](https://github.com/user-attachments/assets/69fb1009-4468-47a1-a5f7-2f6803a6bfda)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![439188783-2fa46a6b-258e-4604-a797-8eef06cebbe3](https://github.com/user-attachments/assets/ae936769-7c0f-48c6-8d6f-c0ee97740690)
```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("income(1) (1).csv")
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![439189000-b7b88076-36bb-4f57-b0e9-045b01cce45b](https://github.com/user-attachments/assets/6d7612c6-a2e8-4e1b-8161-4e180efc7ccf)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![439189120-3a8ec261-fc89-4c8a-8479-4f7ded23c312](https://github.com/user-attachments/assets/bf60d141-0ce2-4c44-9826-c97c72c963c8)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset
