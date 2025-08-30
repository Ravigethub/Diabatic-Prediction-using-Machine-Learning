🩺 Diabetes Prediction using Machine Learning
📌 Problem Statement

We aim to predict whether a patient has diabetes based on health parameters such as glucose level, blood pressure, BMI, insulin level, etc.
We trained Decision Tree and Random Forest classifiers on the dataset to classify patients as Diabetic or Non-Diabetic.


| Feature                      | Type        | Description                    |
| ---------------------------- | ----------- | ------------------------------ |
| Pregnancies                  | Discrete    | Number of times pregnant       |
| Plasma glucose concentration | Continuous  | Glucose concentration          |
| Diastolic blood pressure     | Continuous  | Patient blood pressure         |
| Triceps skin fold thickness  | Continuous  | Skin fold thickness            |
| 2-Hour serum insulin         | Continuous  | Insulin level in 2 hours       |
| Body mass index (BMI)        | Continuous  | Weight / Height²               |
| Diabetes pedigree function   | Continuous  | Genetic likelihood of diabetes |
| Age                          | Continuous  | Patient age                    |
| Class (Target)               | Categorical | 1 = Diabetic, 0 = Non-Diabetic |


🔎 Data Preprocessing

Checked for null/NaN/duplicates

Normalized features using the Norm function

Detected outliers with boxplots → handled using Winsorization (5th & 95th percentile)

Converted target column (Class) into dummy variables

Split dataset → 70% training / 30% testing


📈 Exploratory Data Analysis (EDA)

Univariate Analysis:

Histograms, Boxplots, Distplots → observed skewness & outliers

Bivariate Analysis:

Correlation heatmap to check feature relationships

Found the dataset was imbalanced (more non-diabetic patients)

🤖 Model Building
1️⃣ Decision Tree

Max Depth = 5

Training Accuracy = 81%

Testing Accuracy = 74%

2️⃣ Random Forest

Training Accuracy = 100% (Overfitting)

Testing Accuracy = 81%

3️⃣ Random Forest + GridSearchCV

Tuned 500 estimators

Training Accuracy = 95%

Testing Accuracy = 81%

✅ Conclusion

Random Forest performed better than Decision Tree but had slight overfitting.

Achieved 81% accuracy on test data.

Patients with high glucose & BMI showed higher chances of diabetes.

Predictions can help identify patients at risk of diabetes early.

💡 Benefits

Helps healthcare providers predict potential diabetes cases.

Early diagnosis = better prevention & treatment.

Patients benefit by taking preventive steps before the condition worsens.

📌 Technologies Used

Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-learn (DecisionTree, RandomForest, GridSearchCV)

Jupyter Notebook
