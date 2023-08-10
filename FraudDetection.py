# -*- coding: utf-8 -*-
"""CapstoneFraudDetection.ipynb

# Part 0: Summary of Fraud Detection
- Data is highly imbalanced
- Features of interval_after_signup and time related raw and aggregates are highly predictive of fraud
- Made actionable operation recommendations/proposal for business

# Part 1: Import Data
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

!pip install -U imbalanced-learn
# !pip install pandas-profiling
!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

# Load the Drive helper and mount
from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/My Drive/fraudData"

ipToCountry = pd.read_csv('drive/My Drive/fraudData/IpAddress_to_Country.csv')
fraud_data = pd.read_csv('drive/My Drive/fraudData/imbalancedFraudDF.csv')

fraud_data.head()

"""# Part 2: Data exploration"""

#Distribution of the label column
fraud_data['class'].value_counts()

!pip install --upgrade pip
!pip install --upgrade Pillow

import pandas_profiling

#Inline summary report without saving report as object
pandas_profiling.ProfileReport(fraud_data)

#simpler version without installing pandas_profiling
# fraud_data.describe().transpose()

# will give warnings on missing, correlation, constant value(0 variance), etc, see http://nbviewer.jupyter.org/github/JosPolfliet/pandas-profiling/blob/master/examples/meteorites.ipynb

"""### Check Missing values"""

# count of NaN in each column
fraud_data.isna().sum()
#fraud_data.isnull().sum(axis = 0)

"""### Identify country info based on ip_address

"""

ipToCountry.head()

#start = time.time()

countries = []
for i in range(len(fraud_data)):
    ip_address = fraud_data.loc[i, 'ip_address']#number
    #check which interval does ip_address falls into
    #below [] is list of T/F, only when this ip_address falls into the correct internal row does the index generate a True
    #tmp is a df of shape n * 3, where n is 1 if found a match (ip_address falls in range) or 0 if no match
    tmp = ipToCountry[(ipToCountry['lower_bound_ip_address'] <= ip_address) &
                    (ipToCountry['upper_bound_ip_address'] >= ip_address)]
    if len(tmp) == 1:#found match
        countries.append(tmp['country'].values[0])
    else:#no match
        countries.append('NA')

fraud_data['country'] = countries
#runtime = time.time() - start
#print("Lookup took", runtime, "seconds.")

fraud_data.head()

print(fraud_data.user_id.nunique())#138376
print(len(fraud_data.index))#138376
#all of the user_id has only the first 1 transaction, no way to do time based aggregates,
#e.g. amount/counts in past 1 day for this user

"""### Feature Engineering"""

#time related features: can be done before split, as they has no interaction between other rows, solely based on other columns of the same row
fraud_data['interval_after_signup'] = (pd.to_datetime(fraud_data['purchase_time']) - pd.to_datetime(
        fraud_data['signup_time'])).dt.total_seconds()

fraud_data['signup_days_of_year'] = pd.DatetimeIndex(fraud_data['signup_time']).dayofyear

#bed time operation
fraud_data['signup_seconds_of_day'] = pd.DatetimeIndex(fraud_data['signup_time']).second + 60 * pd.DatetimeIndex(
      fraud_data['signup_time']).minute + 3600 * pd.DatetimeIndex(fraud_data['signup_time']).hour

fraud_data['purchase_days_of_year'] = pd.DatetimeIndex(fraud_data['purchase_time']).dayofyear
fraud_data['purchase_seconds_of_day'] = pd.DatetimeIndex(fraud_data['purchase_time']).second + 60 * pd.DatetimeIndex(
    fraud_data['purchase_time']).minute + 3600 * pd.DatetimeIndex(fraud_data['purchase_time']).hour

fraud_data = fraud_data.drop(['user_id','signup_time','purchase_time'], axis=1)

fraud_data.head()
#note there are NAs in country

print(fraud_data.source.value_counts())

"""### Train/Valid/Test Data Split"""

y = fraud_data['class']
X = fraud_data.drop(['class'], axis=1)

#split into train/test
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
print("X_train.shape:", X_train.shape)
print("X_valid.shape:", X_valid.shape)
print("X_test.shape:", X_test.shape)
print("y_test.shape:", y_test.shape)
print("y_train.shape:", y_train.shape)
print("y_valid.shape:", y_valid.shape)

X_train['country'].value_counts(ascending=True)
#drawback: collision in the same bucket(no differentiation for these countries)

X_train.head()

fraud_purchase_total = fraud_data.loc[fraud_data['class'] == 1, 'purchase_value'].sum()
fraud_purchase_total

"""### Feature Engineering

Convert categorical features with high cadinality to numericals
"""

#converting needs to be done after split
X_train = pd.get_dummies(X_train, columns=['source', 'browser'])
X_train['sex'] = (X_train.sex == 'M').astype(int)

# the more a device is shared, the more suspicious
# if device_id abc occurred 100 times in X_train, then replace all abc in device_id col in X_train by 100
X_train_device_id_mapping = X_train.device_id.value_counts(dropna=False)
X_train['n_dev_shared'] = X_train.device_id.map(X_train_device_id_mapping)# number of times device_id occurred in train data

# the more a ip is shared, the more suspicious
X_train_ip_address_mapping = X_train.ip_address.value_counts(dropna=False)
X_train['n_ip_shared'] = X_train.ip_address.map(X_train_ip_address_mapping)

# the less visit from a country, the more suspicious
X_train_country_mapping = X_train.country.value_counts(dropna=False)#include counts of NaN
X_train['n_country_shared'] = X_train.country.map(X_train_country_mapping)#lots of NAs in country column, #without dropna=False will produce nan in this col


X_train = X_train.drop(['device_id','ip_address','country'], axis=1)

############## Feature Engineering to avoid Data Leakage   ###############

X_valid = pd.get_dummies(X_valid, columns=['source', 'browser'])
X_valid['sex'] = (X_valid.sex == 'M').astype(int)


X_test = pd.get_dummies(X_test, columns=['source', 'browser'])
X_test['sex'] = (X_test.sex == 'M').astype(int)


# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/target-encoding.html
# the more a device is shared, the more suspicious
# X_test['n_dev_shared'] = X_test.device_id.map(X_train_device_id_mapping)

# the more a ip is shared, the more suspicious
# X_test['n_ip_shared'] = X_test.ip_address.map(X_train_ip_address_mapping)

# the less visit from a country, the more suspicious
# X_test['n_country_shared'] = X_test.country.map(X_train_country_mapping)

#but here device_id, ip_address has few overlap between train and test, if we apply the above
# 3 X_train_mappings (generated from X_train) on X_test, then most of the X_test will be NaN, as we can not find the keys in the train mapping


# if apply train mapping, most of the levels in test does not occur in train, so most are null after converting, so redo mapping on test data, which we should not

# the more a device is shared, the more suspicious
X_valid['n_dev_shared'] = X_valid.device_id.map(X_valid.device_id.value_counts(dropna=False))

# the more a ip is shared, the more suspicious
X_valid['n_ip_shared'] = X_valid.ip_address.map(X_valid.ip_address.value_counts(dropna=False))

# the less visit from a country, the more suspicious
X_valid['n_country_shared'] = X_valid.country.map(X_valid.country.value_counts(dropna=False))

X_valid = X_valid.drop(['device_id','ip_address','country'], axis=1)


# the more a device is shared, the more suspicious
X_test['n_dev_shared'] = X_test.device_id.map(X_test.device_id.value_counts(dropna=False))

# the more a ip is shared, the more suspicious
X_test['n_ip_shared'] = X_test.ip_address.map(X_test.ip_address.value_counts(dropna=False))

# the less visit from a country, the more suspicious
X_test['n_country_shared'] = X_test.country.map(X_test.country.value_counts(dropna=False))

X_test = X_test.drop(['device_id','ip_address','country'], axis=1)

# if the levels/values/mapping keys of the column in train and test data are pretty much the same(lots of overlap), e.g. country,
# then we should apply the above 3 X_train_mappings (generated from X_train) on X_test(like below),
# rather than using the new mapping generated from X_test

X_train.head()

"""### Normalization vs Standardization"""

# # normalize (min-max) to [0,1], standardize(StandardScaler) to normal, mu=0,var = 1 can < 0, so we do normalize here

# needs to be brought to the same scale for models like LR with regularization(that are not tree based)

#Compute the train minimum and maximum to be used for later scaling:
scaler = preprocessing.MinMaxScaler().fit(X_train[['n_dev_shared', 'n_ip_shared', 'n_country_shared']])
#print(scaler.data_max_)

#transform the training data and use them for the model training
X_train[['n_dev_shared', 'n_ip_shared', 'n_country_shared']] = scaler.transform(X_train[['n_dev_shared', 'n_ip_shared', 'n_country_shared']])

#before the prediction of the test data, apply the same scaler obtained from above, on X_valid, not fitting a brandnew scaler on test
X_valid[['n_dev_shared', 'n_ip_shared', 'n_country_shared']] = scaler.transform(X_valid[['n_dev_shared', 'n_ip_shared', 'n_country_shared']])

#before the prediction of the test data, apply the same scaler obtained from above, on X_test, not fitting a brandnew scaler on test
X_test[['n_dev_shared', 'n_ip_shared', 'n_country_shared']] = scaler.transform(X_test[['n_dev_shared', 'n_ip_shared', 'n_country_shared']])

X_train.n_dev_shared.value_counts(dropna=False)

X_valid.n_dev_shared.value_counts(dropna=False)

X_test.n_dev_shared.value_counts(dropna=False)

"""# Part 5: Model Training

"""

def performance_metrics(confusion_matrix):
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1], "Confusion matrix should be square."

    num_classes = confusion_matrix.shape[0]

    true_positives = np.diag(confusion_matrix)
    false_positives = np.sum(confusion_matrix, axis=0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
    true_negatives = np.sum(confusion_matrix) - (true_positives + false_positives + false_negatives)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("precision:", precision[1])
    print("recall:", recall[1])
    print("f1", f1_score[1])
    print("specificity:", specificity[1])
    return precision, recall, specificity, f1_score

"""#### XGBoost"""

import xgboost as xgb

####xg_class = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#####              colsample_bynode=1, colsample_bytree=1, gamma=0,
#####              importance_type='gain', learning_rate=0.01, max_depth= 10,
#####              min_child_weight=1, n_estimators=100, n_jobs= 5, num_parallel_tree=1, random_state=0,
#####              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1)

xg_class = xgb.XGBClassifier()
xg_class.fit(X_train,y_train)
xg_preds_train = xg_class.predict(X_train)
xg_preds_valid = xg_class.predict(X_valid)

xg_cm_train = metrics.confusion_matrix(y_train, xg_preds_train)
xg_cmDF_train = pd.DataFrame(xg_cm_train, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(xg_cmDF_train)

xg_cm_valid = metrics.confusion_matrix(y_valid, xg_preds_valid)
xg_cmDF_valid = pd.DataFrame(xg_cm_valid, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(xg_cmDF_valid)

#@title Hide for later(XGBoost)
xg_preds_test = xg_class.predict(X_test)
xg_cm_test = metrics.confusion_matrix(y_test, xg_preds_test)
xg_cmDF_test = pd.DataFrame(xg_cm_test, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(xg_cmDF_test)

performance_metrics(xg_cm_test)

accuracy_score(y_test, xg_preds_test)

#### print the performance metrics ######
performance_metrics(xg_cm_train)

performance_metrics(xg_cm_valid)

from sklearn.calibration import calibration_curve

# Assuming you have a binary classification model with predicted probabilities called 'y_proba' and true labels called 'y_true'
y_true = y_valid
# Calculate calibration curve
xg_y_pred_proba = xg_class.predict_proba(X_valid)
xg_prob_true, xg_prob_pred = calibration_curve(y_true, xg_y_pred_proba[:, 1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(xg_prob_pred, xg_prob_true, marker='o', linestyle='--', label='Uncorrected')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color='gray')
ax.set_title('XGBoost')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_pred_prob, model_name, title=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if title:
        plt.title(f"{model_name} - {title} ROC Curve")
    else:
        plt.title(f"{model_name} ROC Curve")

    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

y_pred_prob_xg = xg_preds_valid
xg_roc_auc = plot_roc_curve(y_true, y_pred_prob_xg, "XGBoost", title = "Without Fixing Class Imbalance")
print("AUC:", xg_roc_auc)

plot_roc_curve(y_test, xg_preds_test, "XGBoost", title = "Without Fixing Class Imbalance")

"""Simple KNN model"""

from sklearn.neighbors import KNeighborsClassifier
knnModel = KNeighborsClassifier()
knnModel.fit(X_train, y_train)
knn_y_pred_train = knnModel.predict(X_train)
knn_y_pred_valid = knnModel.predict(X_valid)

knn_cm_train = metrics.confusion_matrix(y_train, knn_y_pred_train)
knn_cmDF_train = pd.DataFrame(knn_cm_train, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(knn_cmDF_train)

knn_cm_valid = metrics.confusion_matrix(y_valid, knn_y_pred_valid)
knn_cmDF_valid = pd.DataFrame(knn_cm_valid, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(knn_cmDF_valid)

#@title Hide for later(kNN)
knn_y_pred_test = knnModel.predict(X_test)
knn_cm_test = metrics.confusion_matrix(y_test, knn_y_pred_test)
knn_cmDF_test = pd.DataFrame(knn_cm_test, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(knn_cmDF_test)

performance_metrics(knn_cm_test)

performance_metrics(knn_cm_train)

performance_metrics(knn_cm_valid)

"""#### Calibration Curve and its intercept"""

y_true = y_valid
knn_y_pred_proba = knnModel.predict_proba(X_valid)
knn_prob_true, knn_prob_pred = calibration_curve(y_true, knn_y_pred_proba[:, 1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(knn_prob_pred, knn_prob_true, marker='o', linestyle='--', label='Uncorrected')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color='gray')
ax.set_title('kNN')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

y_pred_prob_knn = knn_y_pred_valid
knn_roc_auc = plot_roc_curve(y_true, y_pred_prob_knn, "kNN", title = "Without Fixing Class Imbalance")
print("AUC:", knn_roc_auc)

plot_roc_curve(y_test, knn_y_pred_test, "kNN", title = "Without Fixing Class Imbalance")

"""Simple LogisticRegression model"""

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

# predict on test
logreg_y_pred_train = logreg.predict(X_train)
logreg_y_pred_valid = logreg.predict(X_valid)
logreg_y_proba_valid = logreg.predict_proba(X_valid)

logreg_cm_train = metrics.confusion_matrix(y_train, logreg_y_pred_train)
logreg_cmDF_train = pd.DataFrame(logreg_cm_train, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(logreg_cmDF_train)

logreg_cm_valid = metrics.confusion_matrix(y_valid, logreg_y_pred_valid)
logreg_cmDF_valid = pd.DataFrame(logreg_cm_valid, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(logreg_cmDF_valid)

logreg_y_pred_test = logreg.predict(X_test)
logreg_cm_test= metrics.confusion_matrix(y_test, logreg_y_pred_test)
logreg_cmDF_test = pd.DataFrame(logreg_cm_test, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(logreg_cmDF_test)

performance_metrics(logreg_cm_train)

performance_metrics(logreg_cm_valid)

performance_metrics(logreg_cm_test)

y_true = y_valid
y_pred_prob_logreg = logreg_y_pred_valid
logreg_roc_auc = plot_roc_curve(y_true, y_pred_prob_logreg, "Logistic", title = "Without Fixing Class Imbalance")
print("AUC:", logreg_roc_auc)

plot_roc_curve(y_test, logreg_y_pred_test , "Logistic", title = "Without Fixing Class Imbalance")

# Calculate calibration curve
log_prob_true, log_prob_pred = calibration_curve(y_true, logreg_y_proba_valid[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(log_prob_pred, log_prob_true, marker='o', linestyle='--', label='Uncorrected')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('Logistic Regression')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

"""Simple RF model"""

classifier_RF = RandomForestClassifier()

classifier_RF.fit(X_train, y_train)

# generate class raw probabilities
rf_proba = classifier_RF.predict_proba(X_valid)

# predict class labels 0/1 for the test set
rf_predicted_train = classifier_RF.predict(X_train)
rf_predicted_valid = classifier_RF.predict(X_valid)
# generate evaluation metrics
print("confusion_matrix for training set is: ")
rf_cm_train = confusion_matrix(y_train, rf_predicted_train)
rf_cmDF_train = pd.DataFrame(rf_cm_train, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(rf_cmDF_train)

print("confusion_matrix for validation set is: ")
rf_cm_valid = confusion_matrix(y_valid, rf_predicted_valid)
rf_cmDF_valid = pd.DataFrame(rf_cm_valid, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(rf_cmDF_valid)

rf_predicted_test = classifier_RF.predict(X_test)
rf_cm_test = confusion_matrix(y_test, rf_predicted_test)
rf_cmDF_test = pd.DataFrame(rf_cm_test, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(rf_cmDF_test)

performance_metrics(rf_cm_train)

performance_metrics(rf_cm_valid)

performance_metrics(rf_cm_test)

# Calculate calibration curve
rf_prob_true, rf_prob_pred = calibration_curve(y_true, rf_proba[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(rf_prob_pred, rf_prob_true, marker='o', linestyle='--', label='Uncorrected')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('Random Forest')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

plot_roc_curve(y_true, rf_predicted_valid, "Random Forest", title = "Without Fixing Class Imbalance")

plot_roc_curve(y_test, rf_predicted_test, "Random Forest", title = "Without Fixing Class Imbalance")

"""#### From the simple logistic regression we can notice that the result are all predicted into non-fraud type where we suspect the performance of logistic regression was influenced by the class imbalance of the dataset

#### Accuracy here may not be a good metric to eavaluate model performance

### Address Class Imbalance

#### Random Under Sampling
"""

from imblearn.under_sampling import RandomUnderSampler
under_sampler = RandomUnderSampler()
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)

unique, counts = np.unique(y_train_under, return_counts=True)

print(np.asarray((unique, counts)).T)

logreg_under = LogisticRegression()

# fit the model with data
logreg_under.fit(X_train_under,y_train_under)

# predict on test
logreg_y_under_pred = logreg_under.predict(X_valid)
logreg_y_under_proba = logreg_under.predict_proba(X_valid)

logreg_cm_under = metrics.confusion_matrix(y_valid, logreg_y_under_pred)
logreg_cmDF_under = pd.DataFrame(logreg_cm_under, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(logreg_cmDF_under)

# Calculate calibration curve
log_under_prob_true, log_under_prob_pred = calibration_curve(y_true, logreg_y_under_proba[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(log_under_prob_pred, log_under_prob_true, marker='o', linestyle='--', label='RUS')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('Logistic Regression')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

performance_metrics(logreg_cm_under)

"""kNN"""

knnModel_under = KNeighborsClassifier()
knnModel_under.fit(X_train_under, y_train_under)
#knn_y_pred_train = knnModel.predict(X_train)
knn_y_pred_under = knnModel_under.predict(X_valid)
knn_y_pred_under_proba = knnModel_under.predict_proba(X_valid)

knn_cm_under = metrics.confusion_matrix(y_valid, knn_y_pred_under)
knn_cmDF_under = pd.DataFrame(knn_cm_under, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(knn_cmDF_under)

performance_metrics(knn_cm_under)

# Calculate calibration curve
knn_under_prob_true, knn_under_prob_pred = calibration_curve(y_true, knn_y_pred_under_proba[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(knn_under_prob_pred, knn_under_prob_true, marker='o', linestyle='--', label='RUS')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('kNN')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

"""Random Forest"""

#RF on RUS training data
classifier_RF_under = RandomForestClassifier()

classifier_RF_under.fit(X_train_under, y_train_under)

# predict class labels for the test set
rf_predicted_under = classifier_RF_under.predict(X_valid)

# generate class probabilities
rf_proba_under = classifier_RF_under.predict_proba(X_valid)

# generate evaluation metrics
print ("confusion_matrix_over is: ")
rf_cm_under = confusion_matrix(y_valid, rf_predicted_under)
rf_cmDF_under = pd.DataFrame(rf_cm_under, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(rf_cmDF_under)

performance_metrics(rf_cm_under)

# Calculate calibration curve
rf_under_prob_true, rf_under_prob_pred = calibration_curve(y_true, rf_proba_under[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(rf_under_prob_pred, rf_under_prob_true, marker='o', linestyle='--', label='RUS')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('Random Forest')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

"""XGBoost"""

xg_class_under = xgb.XGBClassifier()
xg_class_under.fit(X_train_under,y_train_under)
xg_preds_under = xg_class_under.predict(X_valid)
xg_preds_proba_under = xg_class_under.predict_proba(X_valid)

xg_cm_under= metrics.confusion_matrix(y_valid, xg_preds_under)
xg_cmDF_under = pd.DataFrame(xg_cm_under, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(xg_cmDF_under)

performance_metrics(xg_cm_under)

# Calculate calibration curve
xg_under_prob_true, xg_under_prob_pred = calibration_curve(y_true, xg_preds_proba_under[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(rf_under_prob_pred, rf_under_prob_true, marker='o', linestyle='--', label='RUS')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('XGBoost')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

"""####  Random Over Sampling"""

from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler()
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)

unique, counts = np.unique(y_train_over, return_counts=True)

print(np.asarray((unique, counts)).T)

"""Logistic Regression"""

logreg_over = LogisticRegression()

# fit the model with data
logreg_over.fit(X_train_over,y_train_over)

# predict on test
logreg_y_over_pred = logreg_over.predict(X_valid)
logreg_y_over_proba = logreg_over.predict_proba(X_valid)

logreg_cm_over = metrics.confusion_matrix(y_valid, logreg_y_over_pred)
logreg_cmDF_over_valid = pd.DataFrame(logreg_cm_over, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(logreg_cmDF_over_valid)

performance_metrics(logreg_cm_over)

# Calculate calibration curve
log_over_prob_true, log_over_prob_pred = calibration_curve(y_true, logreg_y_over_proba[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(log_under_prob_pred, log_under_prob_true, marker='o', linestyle='--', label='ROS')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('Logistic Regression')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

knnModel_over = KNeighborsClassifier()
knnModel_over.fit(X_train_over, y_train_over)
#knn_y_pred_train = knnModel.predict(X_train)
knn_y_pred_over = knnModel_over.predict(X_valid)
knn_y_pred_over_proba = knnModel_over.predict_proba(X_valid)

knn_cm_over = metrics.confusion_matrix(y_valid, knn_y_pred_over)
knn_cmDF_over = pd.DataFrame(knn_cm_over, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(knn_cmDF_over)

performance_metrics(knn_cm_over)

# Calculate calibration curve
knn_over_prob_true, knn_over_prob_pred = calibration_curve(y_true, knn_y_pred_over_proba[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(knn_over_prob_pred, knn_over_prob_true, marker='o', linestyle='--', label='ROS')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = 'grey')
ax.set_title('kNN')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

"""Random Forest"""

#RF on ROS training data
classifier_RF_over = RandomForestClassifier()

classifier_RF_over.fit(X_train_over, y_train_over)

# predict class labels for the test set
rf_predicted_over = classifier_RF_over.predict(X_valid)

# generate class probabilities
rf_proba_over = classifier_RF_over.predict_proba(X_valid)

# generate evaluation metrics
print ("confusion_matrix_over is: ")
rf_cm_over = confusion_matrix(y_valid, rf_predicted_over)
rf_cmDF_over = pd.DataFrame(rf_cm_over, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(rf_cmDF_over)

performance_metrics(rf_cm_over)

# Calculate calibration curve
rf_over_prob_true, rf_over_prob_pred = calibration_curve(y_true, rf_proba_over[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(rf_over_prob_pred, rf_over_prob_true, marker='o', linestyle='--', label='ROS')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('Random Forest')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

"""XGBoost"""

xg_class_over = xgb.XGBClassifier()
xg_class_over.fit(X_train_over, y_train_over)
xg_preds_over = xg_class_over.predict(X_valid)
xg_preds_proba_over = xg_class_over.predict_proba(X_valid)

xg_cm_over= metrics.confusion_matrix(y_valid, xg_preds_over)
xg_cmDF_over = pd.DataFrame(xg_cm_over, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(xg_cmDF_over)

performance_metrics(xg_cm_over)

# Calculate calibration curve
xg_over_prob_true, xg_over_prob_pred = calibration_curve(y_true, xg_preds_proba_over[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(xg_over_prob_pred, xg_over_prob_true, marker='o', linestyle='--', label='ROS')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('XGBoost')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

"""#### SMOTE sampling"""

#Wrong way to sampling: smote before split train/test, then test data does not reflect true distribution in reality,
#and “blend” information from the test set into the training of the model. overfit! think about the case of simple
#oversampling (where I just duplicate observations). If I upsample a dataset before splitting it into a train and
#validation set, I could end up with the same observation in both datasets

#https://imbalanced-learn.org/en/stable/install.html

# Install
# imbalanced-learn is currently available on the PyPi’s reporitories and you can install it via pip:

# pip install -U imbalanced-learn

#oversampling on only the training data, the right way!
#sampling_strategy = number of samples in the majority class over the number of samples in the minority class after resampling

smote = SMOTE()
x_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

unique, counts = np.unique(y_train_sm, return_counts=True)

print(np.asarray((unique, counts)).T)

"""Logistic Regerssion"""

# instantiate the model (using the default parameters)
logreg_sm = LogisticRegression()

# fit the model with data
logreg_sm.fit(x_train_sm,y_train_sm)

# predict on test
logreg_sm_y_pred = logreg_sm.predict(X_valid)
logreg_sm_y_proba = logreg_sm.predict_proba(X_valid)

logit_cm_sm = metrics.confusion_matrix(y_valid, logreg_sm_y_pred)
logit_cmDF_sm = pd.DataFrame(logit_cm_sm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(logit_cmDF_sm)

performance_metrics(logit_cm_sm)

log_sm_prob_true, log_sm_prob_pred = calibration_curve(y_true, logreg_sm_y_proba[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(log_sm_prob_pred, log_sm_prob_true, marker='o', linestyle='--', label='SMOTE')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = 'grey')
ax.set_title("Logistic Regression")
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

knnModel = KNeighborsClassifier()
knnModel.fit(x_train_sm, y_train_sm)
#knn_y_pred_train = knnModel.predict(X_train)
knn_y_pred_valid_sm = knnModel.predict(X_valid)
knn_y_proba_valid_sm = knnModel.predict_proba(X_valid)

knn_cm_valid_sm = metrics.confusion_matrix(y_valid, knn_y_pred_valid_sm)
knn_cmDF_valid_sm = pd.DataFrame(knn_cm_valid_sm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(knn_cmDF_valid_sm)

performance_metrics(knn_cm_valid_sm)

knn_sm_prob_true, knn_sm_prob_pred = calibration_curve(y_true, knn_y_proba_valid_sm[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(knn_sm_prob_pred, knn_sm_prob_true, marker='o', linestyle='--', label='SMOTE')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = 'grey')
ax.set_title("kNN")
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

"""Random Forest"""

#RF on smoted training data
classifier_RF_sm = RandomForestClassifier()

classifier_RF_sm.fit(x_train_sm, y_train_sm)

# predict class labels for the test set
rf_predicted_sm = classifier_RF_sm.predict(X_valid)

# generate class probabilities
rf_proba_sm = classifier_RF_sm.predict_proba(X_valid)

# generate evaluation metrics
print ("confusion_matrix_sm is: ")
rf_cm_sm = confusion_matrix(y_test, rf_predicted_sm)
rf_cmDF_sm = pd.DataFrame(rf_cm_sm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(rf_cmDF_sm)

performance_metrics(rf_cm_sm)

# Calculate calibration curve
rf_sm_prob_true, rf_sm_prob_pred = calibration_curve(y_true, rf_proba_sm[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(rf_sm_prob_pred, rf_sm_prob_true, marker='o', linestyle='--', label='SMOTE')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('Random Forest')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

"""XGBoost"""

xg_class_sm = xgb.XGBClassifier()
xg_class_sm.fit(x_train_sm, y_train_sm)
xg_preds_sm = xg_class_sm.predict(X_valid)
xg_preds_proba_sm = xg_class_sm.predict_proba(X_valid)

xg_cm_sm = metrics.confusion_matrix(y_valid, xg_preds_sm)
xg_cmDF_sm = pd.DataFrame(xg_cm_sm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(xg_cmDF_sm)

performance_metrics(xg_cm_sm)

# Calculate calibration curve
xg_sm_prob_true, xg_sm_prob_pred = calibration_curve(y_true, xg_preds_proba_sm[:,1], n_bins=10)

# Plot calibration curve
fig, ax = plt.subplots()
ax.plot(xg_sm_prob_pred, xg_sm_prob_true, marker='o', linestyle='--', label='SMOTE')
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color = "grey")
ax.set_title('XGBoost')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability')
ax.legend()
plt.show()

"""ACCURACY SCORE CALCULATION

Uncorrected
"""

logreg_accuracy_uncorrect = accuracy_score(y_valid, logreg_y_pred_valid)
knn_accuracy_uncorrect = accuracy_score(y_valid, knn_y_pred_valid)
rf_accuracy_uncorrect = accuracy_score(y_valid, rf_predicted_valid)
xg_acuracy_uncorrect = accuracy_score(y_valid, xg_preds_valid)
print("Accuracy score without correct class imbalance for logistic regression: ", logreg_accuracy_uncorrect)
print("Accuracy score without correct class imbalance for kNN: ", knn_accuracy_uncorrect)
print("Accuracy score without correct class imbalance for random forest: ", rf_accuracy_uncorrect)
print("Accuracy score without correct class imbalance for XGBoost: ", xg_acuracy_uncorrect)

"""Random Under Sampling"""

logreg_accuracy_under = accuracy_score(y_valid, logreg_y_under_pred)
knn_accuracy_under = accuracy_score(y_valid, knn_y_pred_under)
rf_accuracy_under = accuracy_score(y_valid, rf_predicted_under)
xg_acuracy_under = accuracy_score(y_valid, xg_preds_under)
print("Accuracy score after RUS for logistic regression: ", logreg_accuracy_under)
print("Accuracy score after RUS for kNN: ", knn_accuracy_under)
print("Accuracy score after RUS for random forest: ", rf_accuracy_under)
print("Accuracy score after RUS for XGBoost: ", xg_acuracy_under)

"""Random Over Sampling"""

logreg_accuracy_over = accuracy_score(y_valid, logreg_y_over_pred)
knn_accuracy_over = accuracy_score(y_valid, knn_y_pred_over)
rf_accuracy_over = accuracy_score(y_valid, rf_predicted_over)
xg_acuracy_over = accuracy_score(y_valid, xg_preds_over)
print("Accuracy score after ROS for logistic regression: ", logreg_accuracy_over)
print("Accuracy score after ROS for kNN: ", knn_accuracy_over)
print("Accuracy score after ROS for random forest: ", rf_accuracy_over)
print("Accuracy score after ROS for XGBoost: ", xg_acuracy_over)

"""SMOTE"""

logreg_accuracy_sm = accuracy_score(y_valid, logreg_sm_y_pred)
knn_accuracy_sm = accuracy_score(y_valid, knn_y_pred_valid_sm)
rf_accuracy_sm = accuracy_score(y_valid, rf_predicted_sm)
xg_acuracy_sm = accuracy_score(y_valid, xg_preds_sm)
print("Accuracy score after SMOTE for logistic regression: ", logreg_accuracy_sm)
print("Accuracy score after SMOTE for kNN: ", knn_accuracy_sm)
print("Accuracy score after SMOTE for random forest: ", rf_accuracy_sm)
print("Accuracy score after SMOTE for XGBoost: ", xg_acuracy_sm)

"""### Calibration Intercept Calculation"""

from scipy.stats import t
def calculate_calibration_intercept_ci(prob_true, prob_pred, threshold=0.5, alpha=0.05):
    # Convert probabilities to binary labels using a threshold
    y_true = np.where(prob_true >= threshold, 1, 0)

    # Check if only one class is present in the data
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        print("Error: Only one class present in the data.")
        return None, None

    # Fit logistic regression model with offset
    reg = LogisticRegression(solver='liblinear', max_iter=1000)
    reg.fit(prob_pred.reshape(-1, 1), y_true)

    # Extract calibration intercept
    calibration_intercept = reg.intercept_[0]

    # Calculate variance
    n_samples = len(y_true)
    y_pred = reg.predict_proba(prob_pred.reshape(-1, 1))[:, 1]
    variance = np.mean(np.abs(y_pred - prob_true)) / n_samples

    # Calculate standard error
    se_intercept = np.sqrt(variance)

    # Calculate confidence interval
    dof = n_samples - 2  # degrees of freedom
    t_critical = t.ppf(1 - alpha / 2, dof)
    ci_lower = calibration_intercept - t_critical * se_intercept
    ci_upper = calibration_intercept + t_critical * se_intercept

    return calibration_intercept, (ci_lower, ci_upper)

"""Uncorrected"""

logit_calibrate_prob_true = log_prob_true
logit_calibrate_prob_pred = log_prob_pred
logit_calibration_intercept, logit_ci = calculate_calibration_intercept_ci(logit_calibrate_prob_true, logit_calibrate_prob_pred)

print("Calibration intercept for Uncorrected Logistic Regression:", logit_calibration_intercept)
print("95% Confidence Interval of the intercept:", logit_ci)

knn_calibrate_prob_true = knn_prob_true
knn_calibrate_prob_pred = knn_prob_pred
knn_calibration_intercept, knn_ci = calculate_calibration_intercept_ci(knn_calibrate_prob_true, knn_calibrate_prob_pred)

print("Calibration intercept for Uncorrected kNN:", knn_calibration_intercept)
print("95% Confidence Interval of the intercept:", knn_ci)

rf_calibrate_prob_true = rf_prob_true
rf_calibrate_prob_pred = rf_prob_pred
rf_calibration_intercept, rf_ci = calculate_calibration_intercept_ci(rf_calibrate_prob_true, rf_calibrate_prob_pred)

print("Calibration intercept for Uncorrected Random Forest:", rf_calibration_intercept)
print("95% Confidence Interval of the intercept:", rf_ci)

xg_calibrate_prob_true = xg_prob_true
xg_calibrate_prob_pred = xg_prob_pred
xg_calibration_intercept, xg_ci = calculate_calibration_intercept_ci(xg_calibrate_prob_true, xg_calibrate_prob_pred)

print("Calibration intercept for Uncorrected XGBoost:", xg_calibration_intercept)
print("95% Confidence Interval of the intercept:", xg_ci)

"""RUS"""

logit_calibrate_prob_true_under = log_under_prob_true
logit_calibrate_prob_pred_under = log_under_prob_pred
logit_calibration_intercept_under, logit_ci_under = calculate_calibration_intercept_ci(logit_calibrate_prob_true_under, logit_calibrate_prob_pred_under)

print("Calibration intercept for RUS Logistic Regression:", logit_calibration_intercept_under)
print("95% Confidence Interval of the intercept:", logit_ci_under)

knn_calibrate_prob_true_under = knn_under_prob_true
knn_calibrate_prob_pred_under = knn_under_prob_pred
knn_calibration_intercept_under, knn_ci_under = calculate_calibration_intercept_ci(knn_calibrate_prob_true_under, knn_calibrate_prob_pred_under)

print("Calibration intercept for RUS kNN:", knn_calibration_intercept_under)
print("95% Confidence Interval of the intercept:", knn_ci_under)

rf_calibrate_prob_true_under = rf_under_prob_true
rf_calibrate_prob_pred_under = rf_under_prob_pred
rf_calibration_intercept_under, rf_ci_under = calculate_calibration_intercept_ci(rf_calibrate_prob_true_under, rf_calibrate_prob_pred_under)

print("Calibration intercept for Uncorrected Random Forest:", rf_calibration_intercept_under)
print("95% Confidence Interval of the intercept:", rf_ci_under)

xg_calibrate_prob_true_under = xg_under_prob_true
xg_calibrate_prob_pred_under = xg_under_prob_pred
xg_calibration_intercept_under, xg_ci_under = calculate_calibration_intercept_ci(xg_calibrate_prob_true_under, xg_calibrate_prob_pred_under)

print("Calibration intercept for Uncorrected XGBoost:", xg_calibration_intercept_under)
print("95% Confidence Interval of the intercept:", xg_ci_under)

"""ROS"""

logit_calibrate_prob_true_over = log_over_prob_true
logit_calibrate_prob_pred_over = log_over_prob_pred
logit_calibration_intercept_over, logit_ci_over = calculate_calibration_intercept_ci(logit_calibrate_prob_true_over, logit_calibrate_prob_pred_over)

print("Calibration intercept for ROS Logistic Regression:", logit_calibration_intercept_over)
print("95% Confidence Interval of the intercept:", logit_ci_over)

knn_calibrate_prob_true_over = knn_over_prob_true
knn_calibrate_prob_pred_over = knn_over_prob_pred
knn_calibration_intercept_over, knn_ci_iver = calculate_calibration_intercept_ci(knn_calibrate_prob_true_over, knn_calibrate_prob_pred_over)

print("Calibration intercept for Uncorrected kNN:", knn_calibration_intercept)
print("95% Confidence Interval of the intercept:", knn_ci)

rf_calibrate_prob_true_over = rf_over_prob_true
rf_calibrate_prob_pred_over = rf_over_prob_pred
rf_calibration_intercept_over, rf_ci_over = calculate_calibration_intercept_ci(rf_calibrate_prob_true_over, rf_calibrate_prob_pred_over)

print("Calibration intercept for Uncorrected Random Forest:", rf_calibration_intercept_over)
print("95% Confidence Interval of the intercept:", rf_ci_over)

xg_calibrate_prob_true_over = xg_over_prob_true
xg_calibrate_prob_pred_over = xg_over_prob_pred
xg_calibration_intercept_over, xg_ci_over = calculate_calibration_intercept_ci(xg_calibrate_prob_true_over, xg_calibrate_prob_pred_over)

print("Calibration intercept for Uncorrected XGBoost:", xg_calibration_intercept_over)
print("95% Confidence Interval of the intercept:", xg_ci_over)

"""SMOTE"""

logit_calibrate_prob_true_sm = log_sm_prob_true
logit_calibrate_prob_pred_sm = log_sm_prob_pred
logit_calibration_intercept_sm, logit_ci_sm = calculate_calibration_intercept_ci(logit_calibrate_prob_true_sm, logit_calibrate_prob_pred_sm)

print("Calibration intercept for Uncorrected Logistic Regression:", logit_calibration_intercept_sm)
print("95% Confidence Interval of the intercept:", logit_ci_sm)

knn_calibrate_prob_true_sm = knn_sm_prob_true
knn_calibrate_prob_pred_sm = knn_sm_prob_pred
knn_calibration_intercept_sm, knn_ci_sm = calculate_calibration_intercept_ci(knn_calibrate_prob_true_sm, knn_calibrate_prob_pred_sm)

print("Calibration intercept for Uncorrected kNN:", knn_calibration_intercept_sm)
print("95% Confidence Interval of the intercept:", knn_ci_sm)

rf_calibrate_prob_true_sm = rf_sm_prob_true
rf_calibrate_prob_pred_sm = rf_sm_prob_pred
rf_calibration_intercept_sm, rf_ci_sm = calculate_calibration_intercept_ci(rf_calibrate_prob_true_sm, rf_calibrate_prob_pred_sm)

print("Calibration intercept for Uncorrected Random Forest:", rf_calibration_intercept_sm)
print("95% Confidence Interval of the intercept:", rf_ci_sm)

xg_calibrate_prob_true_sm = xg_sm_prob_true
xg_calibrate_prob_pred_sm = xg_sm_prob_pred
xg_calibration_intercept_sm, xg_ci_sm = calculate_calibration_intercept_ci(xg_calibrate_prob_true_sm, xg_calibrate_prob_pred_sm)

print("Calibration intercept for Uncorrected XGBoost:", xg_calibration_intercept_sm)
print("95% Confidence Interval of the intercept:", xg_ci_sm)

"""# Part 6: Parameter tuning by GridSearchCV

Eval metrics for GridSearchCV over all fits upon combination of parameters and cv
"""

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score, pos_label=1),
    'roc_auc_score': make_scorer(roc_auc_score)
}

def grid_search_wrapper(model, parameters, refitscore):
    """
    Fits a GridSearchCV classifier optimizing for F1 score while considering sensitivity and specificity.
    Prints classifier performance metrics.
    """

    grid_search = GridSearchCV(model, parameters, scoring=scorers, refit= refitscore,
                               cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)

    # Make the predictions
    y_pred = grid_search.predict(X_valid)
    y_prob = grid_search.predict_proba(X_valid)[:, 1]

    print('Best params for', 'refitscore:')
    print(grid_search.best_params_)

    # Confusion matrix on the test data
    print('\nConfusion matrix of {} optimized for', refitscore, 'on the test data:'.format(model.__class__.__name__))
    cm = confusion_matrix(y_valid, y_pred)
    cmDF = pd.DataFrame(cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
    print(cmDF)

    print("\t%s: %r" % ("ROC AUC Score is: ", roc_auc_score(y_valid, y_prob)))
    print("\t%s: %r" % ("F1 Score is: ", f1_score(y_valid, y_pred)))

    sensitivity = float(cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    specificity = float(cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    print('Sensitivity (Recall) =', sensitivity)
    print('Specificity =', specificity)

    return grid_search

"""Optimizing on f1_score on LR"""

# C: inverse of regularization strength, smaller values specify stronger regularization
LRGrid = {"C" : np.logspace(-2,2,5), "penalty":["l1"]}# l1 lasso l2 ridge
#param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
logRegModel = LogisticRegression(random_state=0, solver='liblinear')

grid_search_LR_f1 = grid_search_wrapper(logRegModel, LRGrid, 'f1_score')

# C: inverse of regularization strength, smaller values specify stronger regularization
#LRGrid = {"C" : np.logspace(-2,2,5), "penalty":["l1"]}# l1 lasso l2 ridge
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
logRegModel = LogisticRegression(random_state=0, solver='liblinear')

grid_search_LR_f1 = grid_search_wrapper(logRegModel, param_grid, 'f1_score')

"""Optimizing on f1_score on kNN"""

knn_grid_model = KNeighborsClassifier()
knn_params = {'n_neighbors': np.arange(1,11)}
knn_grid_search = grid_search_wrapper(knn_grid_model, knn_params, 'f1_score')

"""Optimizing on f1_score on RF"""

rf_parameters = {
    'max_depth': [None, 5, 15],
    'n_estimators': [10, 30, 50, 100],
    'class_weight': [{0: 1, 1: w} for w in [0.2, 1, 10]],
}

clf = RandomForestClassifier(random_state=0)

grid_search_rf_f1 = grid_search_wrapper(clf, rf_parameters, 'f1_score')

best_rf_model_f1 = grid_search_rf_f1.best_estimator_

print(best_rf_model_f1)

"""Optimizing on F1 socre on XGBoost"""

xgb_parameters = {
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
}

xgb_grid_model = xgb.XGBClassifier(random_state=0)
xgb_grid_search = grid_search_wrapper(xgb_grid_model, xgb_parameters, 'f1_score')

"""Four Models by optimizing Recall(Sensitivity)

L1 regularization
"""

grid_search_LR_recall_lasso = grid_search_wrapper(logRegModel, LRGrid, 'recall_score')

best_logit_model_recall_lasso = grid_search_LR_recall_lasso.best_estimator_

best_logit_y_pred = best_logit_model_recall_lasso.predict(X_valid)
best_logit_cm = confusion_matrix(y_valid, best_logit_y_pred)
best_logit_cmDF = pd.DataFrame(best_logit_cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(best_logit_cmDF)

performance_metrics(best_logit_cm)

best_logit_y_pred_test = best_logit_model_recall_lasso.predict(X_test)
best_logit_cm_test = confusion_matrix(y_test, best_logit_y_pred_test)
best_logit_cmDF_test = pd.DataFrame(best_logit_cm_test, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(best_logit_cmDF_test)

performance_metrics(best_logit_cm_test)

"""L2 regularization"""

grid_search_LR_recall_ridge = grid_search_wrapper(logRegModel, param_grid, 'recall_score')

best_logit_model_recall_ridge = grid_search_LR_recall_ridge.best_estimator_

best_logit2_y_pred = best_logit_model_recall_ridge.predict(X_valid)
best_logit2_cm = confusion_matrix(y_valid, best_logit2_y_pred)
best_logit2_cmDF = pd.DataFrame(best_logit2_cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(best_logit2_cmDF)

best_logit2_y_pred_test = best_logit_model_recall_ridge.predict(X_test)
best_logit2_cm_test = confusion_matrix(y_test, best_logit2_y_pred_test)
best_logit2_cmDF_test = pd.DataFrame(best_logit2_cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(best_logit2_cmDF_test)

performance_metrics(best_logit2_cm)

performance_metrics(best_logit2_cm_test)

knn_grid_search_recall = grid_search_wrapper(knn_grid_model, knn_params, 'recall_score')

best_knn_model_recall = knn_grid_search_recall.best_estimator_

best_knn_y_pred = best_knn_model_recall.predict(X_valid)
best_knn_cm = confusion_matrix(y_valid, best_knn_y_pred)
best_knn_cmDF = pd.DataFrame(best_knn_cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(best_knn_cmDF)

performance_metrics(best_knn_cm)

best_knn_y_pred_test = best_knn_model_recall.predict(X_test)
best_knn_cm_test = confusion_matrix(y_test, best_knn_y_pred_test)
best_knn_cmDF_test = pd.DataFrame(best_knn_cm_test, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(best_knn_cmDF_test)

performance_metrics(best_knn_cm_test)

plot_roc_curve(y_test, best_knn_y_pred_test, "kNN", title = "")

grid_search_rf_recall = grid_search_wrapper(clf, rf_parameters, 'recall_score')

best_rf_model_recall = grid_search_rf_recall.best_estimator_

best_rf_predicted = best_rf_model_recall .predict(X_valid)
best_rf_cm = confusion_matrix(y_valid, best_rf_predicted)
best_rf_cmDF = pd.DataFrame(best_rf_cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(best_rf_cmDF)

best_rf_predicted_test = best_rf_model_recall.predict(X_test)
best_rf_cm_test = confusion_matrix(y_test, best_rf_predicted_test)
best_rf_cmDF_test = pd.DataFrame(best_rf_cm_test, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(best_rf_cmDF_test)

performance_metrics(best_rf_cm)

performance_metrics(best_rf_cm_test)

plot_roc_curve(y_test, best_rf_predicted_test, "Random Forest", title = "")

xgb_grid_search_recall = grid_search_wrapper(xgb_grid_model, xgb_parameters, 'recall_score')

best_xg_model_recall = xgb_grid_search_recall.best_estimator_

best_xg_predicted = best_xg_model_recall.predict(X_valid)
best_xg_cm = confusion_matrix(y_valid, best_xg_predicted)
best_xg_cmDF = pd.DataFrame(best_xg_cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(best_xg_cmDF)

best_xg_predicted_test = best_rf_model_recall.predict(X_test)
best_xg_cm_test = confusion_matrix(y_test, best_xg_predicted_test)
best_xg_cmDF_test = pd.DataFrame(best_xg_cm_test, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(best_xg_cmDF_test)

performance_metrics(best_xg_cm)

performance_metrics(best_xg_cm_test)

plot_roc_curve(y_test, best_xg_predicted_test, "XGBoost", title = "")

"""Print the best model after cross validation and grid search"""

#Var Importance
pd.DataFrame(best_rf_model_recall.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

pd.DataFrame(best_xg_model_recall.feature_importances_, index = X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
