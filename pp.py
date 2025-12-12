import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

 

df = pd.read_csv(r"C:/Users/piyus\Downloads/heart_disease_risk_dataset_earlymed.csv") # 1) Data Load & Preprocess
print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())
duplicates_exist = df.duplicated().any()
print(duplicates_exist)

duplicate_rows = df[df.duplicated()]
print(duplicate_rows)
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

df.drop_duplicates(inplace=True)  # remove all duplicate rows
# After removing
print(df.duplicated().any())  # Should return False
print(f"Number of duplicate rows now: {df.duplicated().sum()}") 
df.columns = df.columns.str.strip()
df.columns = df.columns.str.strip()  # remove extra spaces
df = df.drop_duplicates()
print("Dataset shape after removing duplicates:", df.shape)



print("Heart_Risk class distribution:")
print(df['Heart_Risk'].value_counts())
print("\nNormalized distribution:")
print(df['Heart_Risk'].value_counts(normalize=True))




num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows in dataset: {num_duplicates}")


if num_duplicates > 0:
    print(df[df.duplicated()])


corr_target = df.corr()['Heart_Risk'].sort_values(ascending=False)
print("\nFeature correlation with Heart_Risk:")
print(corr_target)




X = df.drop("Heart_Risk", axis=1)  # Features 2)split and scaling
y = df['Heart_Risk']             # Target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% of data for testing, 80% for training
    random_state=100, # Ensures reproducibility
    shuffle=True
)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


scaler = StandardScaler() #fit
X_train_scaled = scaler.fit_transform(X_train)#fit
X_test_scaled = scaler.transform(X_test)#transforme


lr_model = LogisticRegression(max_iter=1000,C=0.5, random_state=100) #3) model train
lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)          # Predicted labels
y_pred_prob = lr_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for ROC-AUC

rf_model = RandomForestClassifier(           
    n_estimators=100,     # Number of trees
    random_state=100,
    max_depth=6)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)            # Predicted labels
y_pred_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]  # Probability for ROC-AUC
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

xgb_model = XGBClassifier(
    n_estimators=100,         # Number of trees
    eval_metric='logloss',     # Required for classification
    random_state=100,
    max_depth=4
)  
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)            # Predicted labels
y_pred_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]  # Probability for ROC-AUC







print("Accuracy:", accuracy_score(y_test, y_pred)) #4)Evaluate the Model #ACCURACY

cm = confusion_matrix(y_test, y_pred)#confusion matrix
ConfusionMatrixDisplay(cm).plot()

roc_auc = roc_auc_score(y_test, y_pred_prob)#ROC-AUC SCORE
print("ROC-AUC Score:", roc_auc)


cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()


print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(cm_rf).plot()
roc_auc_rf = roc_auc_score(y_test, y_pred_prob_rf)
print("Random Forest ROC-AUC Score:", roc_auc_rf)


print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
ConfusionMatrixDisplay(cm_xgb).plot()
roc_auc_xgb = roc_auc_score(y_test, y_pred_prob_xgb)
print("XGBoost ROC-AUC Score:", roc_auc_xgb)


lr_model = LogisticRegression(max_iter=1000,C=0.5, random_state=100) #5)votimg ensemble
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=100)
xgb_model = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=100,max_depth=4)

voting_model = VotingClassifier(
    estimators=[('lr', lr_model), ('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'  # Use predicted probabilities
)

voting_model.fit(X_train_scaled, y_train)

y_pred_vote = voting_model.predict(X_test_scaled)
y_pred_prob_vote = voting_model.predict_proba(X_test_scaled)[:, 1]


print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_vote))# Accuracy


print("Classification Report:\n", classification_report(y_test, y_pred_vote))# Classification Report


roc_auc_vote = roc_auc_score(y_test, y_pred_prob_vote)# ROC-AUC Score
print("Voting Classifier ROC-AUC Score:", roc_auc_vote)

import joblib

joblib.dump({
    "model": voting_model,
    "scaler": scaler
}, "model.pkl")




models = {
    'Logistic Regression': LogisticRegression(max_iter=1000,C=0.5, random_state=100),  #6)Cross-Validation
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=100),
    'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=100,max_depth=4)
}

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=50)
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='accuracy')
    print(f"{name} - CV Accuracy Scores: {cv_scores}")
    print(f"{name} - Mean CV Accuracy: {cv_scores.mean():.4f}\n")



for name, model in models.items():
    model.fit(X_train_scaled, y_train)



plt.figure(figsize=(15,6)) # visualization
sns.boxplot(data=df.drop("Heart_Risk", axis=1))
plt.xticks(rotation=45)
plt.title("Feature Distribution and Outliers")
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='Age', y='Heart_Risk', data=df, hue='Heart_Risk', palette='Set1', alpha=0.6)

plt.title('Scatter Plot of Age vs Heart Risk (Outlier Visualization)')
plt.xlabel('Age')
plt.ylabel('Heart Risk')
plt.grid(True)
plt.show()



features = X.columns.tolist() #real time data
input_data = {}
print("\nEnter patient details for Heart Risk Prediction:")
for feature in features:
    if feature == "Age":
        value = int(input(f"{feature}: "))
    else:
        value = int(input(f"{feature} (0=No, 1=Yes): "))
    input_data[feature] = value

patient_df = pd.DataFrame([input_data])
patient_scaled = scaler.transform(patient_df)

prediction = voting_model.predict(patient_scaled)[0]
probability = voting_model.predict_proba(patient_scaled)[:,1][0]

if prediction == 1:
    print(f"⚠️ High risk of heart disease! Probability: {probability:.2f}")
else:
    print(f"✅ Low risk of heart disease. Probability: {probability:.2f}")



