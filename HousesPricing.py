
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)

os.chdir('E:\\Python\\HousesPricing')

###check NA, fill by mean where street = pave
def checkDAta(data):
    colsWitNa = []
    for i in range(0, len(data.columns)):
        if data[data.columns[i]].isnull().values.any():
            colsWitNa.extend([data.columns[i]])
    print colsWitNa


def fillNulls(data):
    data.loc[data.LotFrontage.isnull(), "LotFrontage"]= data.LotFrontage[data.Street=="Pave"].mean()
    data.loc[data.Alley.isnull(), "Alley"]= 'None'
    data.loc[data.MSZoning.isnull(), "MSZoning"]= 'None'
    data.loc[data.Fence.isnull(), "Fence"]= 'None'
    data.loc[data.MiscFeature.isnull(), "MiscFeature"]= 'None'
    data.loc[data.BsmtCond.isnull(), "BsmtCond"]= 'None'
    data.loc[data.BsmtExposure.isnull(), "BsmtExposure"]= 'None'
    data.loc[data.BsmtFinSF1.isnull(), "BsmtFinSF1"]= 'None'
    data.loc[data.BsmtFinSF2.isnull(), "BsmtFinSF2"]= 'None'
    data.loc[data.BsmtFinType1.isnull(), "BsmtFinType1"]= 'None'
    data.loc[data.BsmtFinType2.isnull(), "BsmtFinType2"]= 'None'
    data.loc[data.BsmtFullBath.isnull(), "BsmtFullBath"]= 'None'
    data.loc[data.BsmtHalfBath.isnull(), "BsmtHalfBath"]= 'None'
    data.loc[data.BsmtUnfSF.isnull(), "BsmtUnfSF"]= 'None'
    data.loc[data.BsmtQual.isnull(), "BsmtQual"]= 'None'
    data.loc[data.Electrical.isnull(), "Electrical"]= 'None'
    data.loc[data.Functional.isnull(), "Functional"]= 'None'
    data.loc[data.Exterior1st.isnull(), "Exterior1st"]= 'None'
    data.loc[data.Exterior2nd.isnull(), "Exterior2nd"]= 'None'
    data.loc[data.FireplaceQu.isnull(), "FireplaceQu"]= 'None'
    data.loc[data.GarageArea.isnull(), "GarageArea"]= 'None'
    data.loc[data.GarageCars.isnull(), "GarageCars"]= 'None'
    data.loc[data.GarageCond.isnull(), "GarageCond"]= 'None'
    data.loc[data.GarageQual.isnull(), "GarageQual"]= 'None'
    data.loc[data.GarageType.isnull(), "GarageType"]= 'None'
    data.loc[data.GarageYrBlt.isnull(), "GarageYrBlt"]= 'None'
    data.loc[data.GarageFinish.isnull(), "GarageFinish"]= 'None'
    data.loc[data.KitchenQual.isnull(), "KitchenQual"]= 'None'
    data.loc[data.MasVnrArea.isnull(), "MasVnrArea"]= 'None'
    data.loc[data.MasVnrType.isnull(), "MasVnrType"]= 'None'
    data.loc[data.PoolQC.isnull(), "PoolQC"]= 'None'
    data.loc[data.SaleType.isnull(), "SaleType"]= 'None'
    data.loc[data.TotalBsmtSF.isnull(), "TotalBsmtSF"]= 'None'
    data.loc[data.Utilities.isnull(), "Utilities"]= 'None'
    return data


def fillNulls2(data):
    ###Fo numeric fill by mean
    ###For categorical by most frequantely used
    data.loc[data.LotFrontage.isnull(), "LotFrontage"]= data.LotFrontage[data.Street=="Pave"].mean()
    data.loc[data.Alley.isnull(), "Alley"]= data.Alley.value_counts().index[0]
    data.loc[data.MSZoning.isnull(), "MSZoning"]= data.MSZoning.value_counts().index[0]
    data.loc[data.Fence.isnull(), "Fence"]= data.Fence.value_counts().index[0]
    data.loc[data.MiscFeature.isnull(), "MiscFeature"]= data.MiscFeature.value_counts().index[0]
    data["IsBsmt"]= data.apply(lambda x: True if (not x.BsmtCond.isnull().any()) and  (x.BsmtExposure.isnull().any()) and (x.BsmtFinSF1.isnull().any()) and (x.BsmtCond.BsmtHalfBath().any()) else False)
    data.loc[data.BsmtCond.isnull(), "BsmtCond"]= 'None'
    data.loc[data.BsmtExposure.isnull(), "BsmtExposure"]= 'None'
    data.loc[data.BsmtFinSF1.isnull(), "BsmtFinSF1"]= 'None'
    data.loc[data.BsmtFinSF2.isnull(), "BsmtFinSF2"]= 'None'
    data.loc[data.BsmtFinType1.isnull(), "BsmtFinType1"]= 'None'
    data.loc[data.BsmtFinType2.isnull(), "BsmtFinType2"]= 'None'
    data.loc[data.BsmtFullBath.isnull(), "BsmtFullBath"]= 'None'
    data.loc[data.BsmtHalfBath.isnull(), "BsmtHalfBath"]= 'None'
    data.loc[data.BsmtUnfSF.isnull(), "BsmtUnfSF"]='None'
    data.loc[data.BsmtQual.isnull(), "BsmtQual"]= 'None'
    data.loc[data.Electrical.isnull(), "Electrical"]= data.Electrical.value_counts().index[0]
    data.loc[data.Functional.isnull(), "Functional"]= data.Functional.value_counts().index[0]
    data.loc[data.Exterior1st.isnull(), "Exterior1st"]= data.Exterior1st.value_counts().index[0]
    data.loc[data.Exterior2nd.isnull(), "Exterior2nd"]= data.Exterior2nd.value_counts().index[0]
    data.loc[data.FireplaceQu.isnull(), "FireplaceQu"]= data.FireplaceQu.value_counts().index[0]
    data.loc[data.GarageArea.isnull(), "GarageArea"]= data.GarageArea.mean()
    data.loc[data.GarageCars.isnull(), "GarageCars"]= data.GarageCars.value_counts().index[0]
    data.loc[data.GarageCond.isnull(), "GarageCond"]= data.GarageCond.value_counts().index[0]
    data.loc[data.GarageQual.isnull(), "GarageQual"]= data.GarageQual.value_counts().index[0]
    data.loc[data.GarageType.isnull(), "GarageType"]= data.GarageType.value_counts().index[0]
    data.loc[data.GarageYrBlt.isnull(), "GarageYrBlt"]= data.GarageYrBlt.value_counts().index[0]
    data.loc[data.GarageFinish.isnull(), "GarageFinish"]= data.GarageFinish.value_counts().index[0]
    data.loc[data.KitchenQual.isnull(), "KitchenQual"]= data.KitchenQual.value_counts().index[0]
    data.loc[data.MasVnrArea.isnull(), "MasVnrArea"]= data.MasVnrArea.mean()
    data.loc[data.MasVnrType.isnull(), "MasVnrType"]= data.MasVnrType.value_counts().index[0]
    data.loc[data.PoolQC.isnull(), "PoolQC"]= data.PoolQC.value_counts().index[0]
    data.loc[data.SaleType.isnull(), "SaleType"]= data.SaleType.value_counts().index[0]
    data.loc[data.TotalBsmtSF.isnull(), "TotalBsmtSF"]= data.TotalBsmtSF.mean()
    data.loc[data.Utilities.isnull(), "Utilities"]= data.Utilities.value_counts().index[0]

    return data


def Error(y, y_pred):
    return(np.mean(abs(y-y_pred)/y))


train= pd.read_csv("train.csv")
test= pd.read_csv("test.csv")
data= pd.concat([train, test], axis=0)


data.loc[(data.PoolQC.isnull()) & (data.PoolArea==0), "PoolQC"]= "None"
temp = data[data.PoolQC.isnull()]
cat= temp.PoolArea.unique()
catmeans= data[["PoolArea", "PoolQC"]].groupby("PoolQC", as_index=False).mean()
for i in range(0, len(cat)):
    tt= data[(data.PoolQC.isnull()) & (data.PoolArea==cat[i])][["PoolArea", "PoolQC"]]
    catmeans["Diff"]= (catmeans.PoolArea- tt.PoolArea.iloc[0]).abs()
    replaceCat= catmeans.PoolQC[catmeans.Diff==catmeans.Diff.min()].iloc[0]
    data.loc[(data.PoolQC.isnull()) & (data.PoolArea == cat[i]), "PoolQC"]= replaceCat


data= data[list(train.columns)]

###Cathegory convert to numeric by count
#train.MSZoning= train.MSZoning.astype('category').cat.codes
columnsToInt = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", \
               "Neighborhood",	"Condition1",	"Condition2",	"BldgType",	"HouseStyle" , "RoofStyle", \
               "RoofMatl",	"Exterior1st",	"Exterior2nd",	"MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", \
               "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "GarageType", "GarageFinish", \
               "GarageQual",	"GarageCond",	"PavedDrive", "PoolQC",	"Fence",	"MiscFeature", "SaleType", "SaleCondition"]

for i in range(0, len(columnsToInt)):
    tt= pd.DataFrame(data[columnsToInt[i]].value_counts())
    #if len(tt)<2:
    tt["Category"]= tt.index
    tt["CategoryNew"]= range(0, len(tt))
    categories= tt.Category.astype(str)
    for j in categories:
        data.loc[data[columnsToInt[i]] == j, columnsToInt[i]] = tt.CategoryNew[tt.Category == j].iloc[0]


###Add new variables
# add total area
cols= ["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", \
       "GrLivArea", "GarageArea", "OpenPorchSF", "WoodDeckSF"]
totalArea= 0
for i in cols:
    totalArea= totalArea + data[i].fillna(0)

#data["TotalArea"]= data.MasVnrArea + data.BsmtFinSF1 + data.BsmtFinSF2 + data.TotalBsmtSF + \
#                   data['1stFlrSF'] + data['2ndFlrSF'] + data.GrLivArea + data.GarageArea + \
#                   data.OpenPorchSF
#temp= data[["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", \
#            "GrLivArea", "GarageArea", "OpenPorchSF", "PoolArea", "TotalArea", "SalePrice"]]

data["IsGarage"]= ~data.GarageType.isnull()*1
cols= ["IsGarage", "FullBath", "BsmtFullBath", "BsmtHalfBath", "HalfBath"]
totalRoms= 0
for i in cols:
    totalRoms= totalRoms + data[i].fillna(0)

#data["TotalRooms"]= data.IsGarage + data.FullBath + data.BsmtFullBath + data.BsmtHalfBath + data.HalfBath
#temp= data[["IsGarage", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotalRooms", "SalePrice"]]

data["TotalArea"]= totalArea
data["TotalRooms"]= totalRoms

data= fillNulls2(data=data)

###Split on train and test set
train= data[:1460]
test= data[1460:]
testId= test.Id
#test = test.drop("SalePrice", axis=1)
tCorr= train.corr()
features= list(tCorr[tCorr.index=="SalePrice"][tCorr.abs()>0.5].dropna(axis=1).columns)
features.remove("SalePrice")
#train = train.drop("SalePrice", axis=1)

train= train[train.SalePrice< train.SalePrice.quantile(.95)]
y= train.SalePrice
train= train[features]
test= test[features]


###check Nans in features
for i in features:
    tt= train[train[i].isnull()]
    if len(tt)>0:
        print i


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=0)

rf = RandomForestRegressor(n_jobs = 1)
rf.fit(X_train, y_train)
pred= rf.predict(X_test)

print "MSE: " + str(Error(y_test, pred))
print "R2 train: " + str(rf.score(X_train, y_train))
print "R2 test: " + str(rf.score(X_test, y_test))


for n_estimators in [50, 100, 300, 500, 1000, 4000]:
    rf = RandomForestRegressor(n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    print(n_estimators, ': ', Error(y_test, pred))
    print (str(n_estimators)+ " Score: "+ str(rf.score(X_test, y_test)))



###Predict test set

rf = RandomForestRegressor(n_jobs = 2, n_estimators=300)
rf.fit(train, y)
pred= rf.predict(test)

print "R2 train: " + str(rf.score(train, y))

test["SalePrice"]= pred
test["Id"]= testId
test[["Id", "SalePrice"]].to_csv("TestPrediction.csv", index=False)




check= X_test.copy()
check["TruPrice"]= y_test
check["PredPrice"]= pred
check["Id"]= range(0, len(check))


plt.scatter(check.Id, check.TruPrice, c="g", alpha=0.7, s=5)
plt.scatter(check.Id, check.PredPrice, c="b", alpha=0.7, s=5)
plt.grid(True)


plt.scatter(train.index, train.SalePrice, c="g", alpha=0.7, s=5)
plt.grid(True)

train.SalePrice.quantile(.99)