

###imported required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/Ravi/Downloads/decision tree/Datasets_DTRF/Diabetes.csv")
###checking null values
df.columns
df.isnull().sum()
df.isna().sum()
df.dropna()
df.drop_duplicates()
df.dtypes
df.shape
p = df.hist(figsize = (20,20))
df[" Class variable"].hist(),plt.title("Class variable")

import seaborn as sns

plt.boxplot(df.iloc[:,:8])


# Detection of outliers (find limits for salary based on IQR)
IQR = df[' Number of times pregnant'].quantile(0.75) - df[' Number of times pregnant'].quantile(0.25)
lower_limit = df[' Number of times pregnant'].quantile(0.25) - (IQR * 1.5)
upper_limit = df[' Number of times pregnant'].quantile(0.75) + (IQR * 1.5)


############### 3. Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=[' Number of times pregnant'])
df_t = winsor.fit_transform(df[[' Number of times pregnant']])
plt.boxplot(df_t)
df[' Number of times pregnant']=df_t
plt.boxplot(df[' Number of times pregnant'])


IQR = df[' Plasma glucose concentration'].quantile(0.75) - df[' Plasma glucose concentration'].quantile(0.25)
lower_limit = df[' Plasma glucose concentration'].quantile(0.25) - (IQR * 1.5)
upper_limit = df[' Plasma glucose concentration'].quantile(0.75) + (IQR * 1.5)

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=[' Plasma glucose concentration'])
df_t[' Plasma glucose concentration'] = winsor.fit_transform(df[[' Plasma glucose concentration']])
plt.boxplot(df_t[' Plasma glucose concentration']);plt.title([' Plasma glucose concentration'])
df[' Plasma glucose concentration']=df_t[' Plasma glucose concentration']



IQR = df[' Diastolic blood pressure'].quantile(0.75) - df[' Diastolic blood pressure'].quantile(0.25)
lower_limit = df[' Diastolic blood pressure'].quantile(0.25) - (IQR * 1.5)
upper_limit = df[' Diastolic blood pressure'].quantile(0.75) + (IQR * 1.5)

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=[' Diastolic blood pressure'])
df_t[' Diastolic blood pressure'] = winsor.fit_transform(df[[' Diastolic blood pressure']])
plt.boxplot(df_t[' Diastolic blood pressure']);plt.title([' Diastolic blood pressure'])
df[' Diastolic blood pressure']=df_t[' Diastolic blood pressure']



IQR = df[' Triceps skin fold thickness'].quantile(0.75) - df[' Triceps skin fold thickness'].quantile(0.25)
lower_limit = df[' Triceps skin fold thickness'].quantile(0.25) - (IQR * 1.5)
upper_limit = df[' Triceps skin fold thickness'].quantile(0.75) + (IQR * 1.5)

winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=[' Triceps skin fold thickness'])
df_t[' Triceps skin fold thickness'] = winsor.fit_transform(df[[' Triceps skin fold thickness']])
plt.boxplot(df_t[' Triceps skin fold thickness']);plt.title([' Triceps skin fold thickness'])
df[' Triceps skin fold thickness']=df_t[' Triceps skin fold thickness']

IQR=df[' 2-Hour serum insulin'].quantile(0.75)-df[' 2-Hour serum insulin'].quantile(0.25)
lower_limit=df[' 2-Hour serum insulin'].quantile(0.25)-(IQR*1.5)
upper_limit=df[' 2-Hour serum insulin'].quantile(0.75)-(IQR*1.5)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,
                  variables=[' 2-Hour serum insulin'])

df[' 2-Hour serum insulin']=winsor.fit_transform(df[[' 2-Hour serum insulin']])
plt.boxplot(df[' 2-Hour serum insulin']);plt.title([' 2-Hour serum insulin'])
df[' 2-Hour serum insulin']=df[' 2-Hour serum insulin']


IQR=df[' Body mass index'].quantile(0.75)-df[' Body mass index'].quantile(0.25)
lower_limit=df[' Body mass index'].quantile(0.25)-(IQR*1.5)
upper_limit=df[' Body mass index'].quantile(0.75)-(IQR*1.5)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,
                  variables=[' Body mass index'])

df[' Body mass index']=winsor.fit_transform(df[[' Body mass index']])
plt.boxplot(df[' Body mass index']);plt.title([' Body mass index'])
df[' Body mass index']=df[' Body mass index']


IQR=df[' Diabetes pedigree function'].quantile(0.75)-df[' Diabetes pedigree function'].quantile(0.25)
lower_limit=df[' Diabetes pedigree function'].quantile(0.25)-(IQR*1.5)
upper_limit=df[' Diabetes pedigree function'].quantile(0.75)-(IQR*1.5)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,
                  variables=[' Diabetes pedigree function'])

df[' Diabetes pedigree function']=winsor.fit_transform(df[[' Diabetes pedigree function']])
plt.boxplot(df[' Diabetes pedigree function']);plt.title([' Diabetes pedigree function'])
df[' Diabetes pedigree function']=df[' Diabetes pedigree function']


IQR=df[' Age (years)'].quantile(0.75)-df[' Age (years)'].quantile(0.25)
lower_limit=df[' Age (years)'].quantile(0.25)-(IQR*1.5)
upper_limit=df[' Age (years)'].quantile(0.75)-(IQR*1.5)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,
                  variables=[' Age (years)'])

df[' Age (years)']=winsor.fit_transform(df[[' Age (years)']])
plt.boxplot(df[' Age (years)']);plt.title([' Age (years)'])
df[' Age (years)']=df[' Age (years)']

plt.boxplot(df.iloc[:,:8]);plt.title('outliers for diabetes')

sns.distplot(df[' 2-Hour serum insulin'])




# seaborn has an easy method to showcase heatmap
sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')

##checking data normality
describe=df.describe()

def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)
df_norm=norm_func(df.loc[:, df.columns!=" Class variable"])

###taken column names
colnames=df.columns
###unique values in the columns
df[" Class variable"].unique()
##count the unique values
df[" Class variable"].value_counts()
df = pd.get_dummies(df, columns = [" Class variable"], drop_first = True)

###saperated predictors and target
predictors=df_norm.iloc[:,:8]
target=df[" Class variable_YES"]

###given data to train and test
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.33)
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn import tree
model = DT(criterion = 'entropy',max_depth=5)
model.fit(x_train,y_train)


plt.figure(figsize=(20,20))
tree.plot_tree(model,filled=True)

###predection
test_predict=model.predict(x_test)
pd.crosstab(test_predict,y_test,rownames=['actual'],colnames=['predict'])

np.mean(test_predict==y_test)
###pre train

train_predict=model.predict(x_train)
pd.crosstab(train_predict,y_train,rownames=['actual'],colnames=['predict'])
np.mean(train_predict==y_train)


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=200)

rf_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, rf_clf.predict(x_test))
accuracy_score(y_test, rf_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, rf_clf.predict(x_train))
accuracy_score(y_train, rf_clf.predict(x_train))



######
# GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=12)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, cv_rf_clf_grid.predict(x_test))
accuracy_score(y_test, cv_rf_clf_grid.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, cv_rf_clf_grid.predict(x_train))
accuracy_score(y_train, cv_rf_clf_grid.predict(x_train))
