from operator import index
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import col
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import re
import seaborn as sns
import math

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("py/train/training.csv")

cleanDf = df
df_gender = df
df_embark = df
df_salu = df

gender = df_gender["Sex"]
female = gender.str.contains("female")
df_gender = np.where(female,1,0)


#embarking clean data
df_embark = df['Embarked']
cList = df_embark.str.contains("C")
sList = df_embark.str.contains("S")
df_embark['Embarked'] = np.where(cList, 1, (np.where(sList, 2, 3)))
#fill empty Embarked
df_embark.fillna(df_embark.mode(), inplace = False)
df_embark["Sex"]=df_gender

df_salu["Salutations"] = df_salu["Name"].apply(lambda name: name.split(",")[1].split(".")[0])
grp = df_salu.groupby(["Salutations","Pclass"])
print(grp)
median_age = grp["Age"].median()

missingAge = []
for ind in df_salu.index:
    Title = df_salu["Salutations"][ind]
    Class = df_salu["Pclass"][ind]
    if math.isnan(median_age[Title, Class]):
        missingAge.append(df_salu["Age"].median())
    else:
        missingAge.append(median_age[Title,Class])

for ind in df_salu.index:
    age = df_salu['Age'][ind]
    if math.isnan(age):
        df_salu['Age'][ind] = missingAge[ind]

cleanDf['Sex']= df_embark["Sex"]
cleanDf["Age"]=df_salu['Age']
cleanDf['Embarked']= df_embark["Embarked"]


cleanDf= cleanDf.drop(columns= ["Cabin"])
cleanDf=cleanDf.drop(columns= ["Name"])
cleanDf=cleanDf.drop(columns= ["Ticket"])
cleanDf=cleanDf.drop(columns= ["Salutations"])
cleanDf=cleanDf.drop(columns= ["Embarked"])

cleanDfSex = cleanDf.drop(columns=["Sex"])
cleanDfSurvived = cleanDf.drop(columns=["Survived"])

x = cleanDfSex.drop(columns=['Survived'])
x = x.to_numpy()
z = cleanDfSurvived.drop(columns=["Sex"])
z = z.to_numpy()
y = cleanDf["Survived"].values
w = cleanDf["Sex"].values
model = LogisticRegression(solver="liblinear", random_state=0)
model2 = LogisticRegression(solver="liblinear", random_state=0)

model.fit(x,y)
model2.fit(z,w)


dft = pd.read_csv("py/testData/test.csv")

cleanDf = dft
df_gender = dft
df_embark = dft
df_salu = dft



#embarking clean data
df_embark = dft['Embarked']
cList = df_embark.str.contains("C")
sList = df_embark.str.contains("S")
df_embark['Embarked'] = np.where(cList, 1, (np.where(sList, 2, 3)))
#fill empty Embarked
df_embark.fillna(df_embark.mode(), inplace = False)


df_salu["Salutations"] = df_salu["Name"].apply(lambda name: name.split(",")[1].split(".")[0])
grp = df_salu.groupby(["Salutations","Pclass"])
print(grp)
median_age = grp["Age"].median()

missingAge = []
for ind in df_salu.index:
    Title = df_salu["Salutations"][ind]
    Class = df_salu["Pclass"][ind]
    if math.isnan(median_age[Title, Class]):
        missingAge.append(df_salu["Age"].median())
    else:
        missingAge.append(median_age[Title,Class])

for ind in df_salu.index:
    age = df_salu['Age'][ind]
    if math.isnan(age):
        df_salu['Age'][ind] = missingAge[ind]


cleanDf["Age"]=df_salu['Age']
cleanDf['Embarked']= df_embark["Embarked"]
cleanDf['Fare'] = cleanDf['Fare'].fillna(cleanDf['Fare'].median())


cleanDf= cleanDf.drop(columns= ["Cabin"])
cleanDf=cleanDf.drop(columns= ["Name"])
cleanDf=cleanDf.drop(columns= ["Ticket"])
cleanDf=cleanDf.drop(columns= ["Salutations"])
cleanDf=cleanDf.drop(columns= ["Embarked"])

texts = cleanDf
lived = model.predict(texts)

passId = cleanDf['PassengerId'].tolist()
result = pd.DataFrame(passId, columns = ['PassengerId'])
predictSex = cleanDf
sex = model2.predict(predictSex)
result["Survived"] = lived
result["Sex"] = sex

for ind in result.index:
    if (result["Sex"][ind] == 0):
        result["Sex"][ind]= "Male"
    else:
        result["Sex"][ind] = "Female"

result.to_csv(r'py/results/submission.csv', index = False)