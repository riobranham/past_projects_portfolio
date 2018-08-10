# =============================================================================
# Title: Predict Default Customers
# Author: Rio Branham
# =============================================================================


# %% Setup

import vslr as vs
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report, roc_curve,
                             confusion_matrix)

# %% Data

full = vs.run_sql(vs.ch, './predict_default/bin/full.sql')

sample = full[['billing_account',
               'aging_days',
               'age_group',
               'case_number',
               'production_charges',
               'sales_tax',
               'other_charges',
               'total_due',
               'bucket_current',
               'current_sales_office_id',
               'current_sales_rep_id',
               'customer_fico_score',
               'service_state',
               'service_city',
               'service_latitude',
               'service_longitude',
               'transferred_flag',
               'fund_name',
               'credit_range',
               'solar_utility_provider',
               'opty_contract_type']]
sample.index = sample.billing_account

sample = sample.assign(
        default=[
                1 if result is not None else 0 for result in sample.case_number
                ])

sample = sample.drop(['billing_account', 'case_number'], 1)

sample = sample.sample(50000)

category_vars = ['age_group', 'current_sales_office_id',
                 'current_sales_rep_id', 'service_state', 'service_city',
                 'transferred_flag', 'fund_name', 'credit_range',
                 'solar_utility_provider', 'opty_contract_type']
category_dummies = vs.pd.get_dummies(sample[category_vars])

sample = sample.drop(category_vars, 1)
sample = sample.join(category_dummies)
sample = sample.dropna()

X_trainval, X_test, y_trainval, y_test = train_test_split(
    sample.drop('default', 1), sample.default)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval)

# %% Model

forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=2,
                                class_weight='balanced_subsample')
forest.fit(X_train, y_train)
roc_auc_score(y_val, forest.predict_proba(X_val)[:, 1])

# Plot ROC curve
roc = roc_curve(y_val, forest.predict_proba(X_val)[:, 1])

plt.close()
plt.plot(roc[0], roc[1])
default_threshold = np.argmin(np.abs(roc[2] - .5))
best_op = np.argmin(np.abs(roc[1] - .756407))
plt.plot(roc[0][default_threshold], roc[1][default_threshold], 'o',
         markersize=10, label='Default Threshold', fillstyle='none')
plt.plot(roc[0][best_op], roc[1][best_op], 'o',
         markersize=10, fillstyle='none',
         label='Best Operating Point: ths={:.3f}'.format(roc[2][best_op]))
plt.legend(loc='best')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Random Forest ROC Curve for Classifying Default Customers')

# Predict using Threshold .126
y_pred = forest.predict_proba(X_test)[:, 1] > .126
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])
forest.score(X_test, y_test)

# %% Final Results 2018-05-12

# Used Validation set to select threshold then calculated metrics on Test set

# Features:
#     [['billing_account',
#       'aging_days',
#       'age_group',
#       'case_number',
#       'production_charges',
#       'sales_tax',
#       'other_charges',
#       'total_due',
#       'bucket_current',
#       'current_sales_office_id',
#       'current_sales_rep_id',
#       'customer_fico_score',
#       'service_state',
#       'service_city',
#       'service_latitude',
#       'service_longitude',
#       'transferred_flag',
#       'fund_name',
#       'credit_range',
#       'solar_utility_provider',
#       'opty_contract_type']]

# Sample Size:
#     50000

# Model:
#     RandomForest(n_estimators=1000, class_weight='balanced_subsample')

# Threshold:
#     .126

# Accuracy:
#     .99209

# AUC:
#     .91359

# Default Class Precision:
#     .57

# Defaul Class Recall:
#     .79

# Default Class F1-Score:
#     .66

# Confusion Matrix:
#     [[11419   125]
#      [   44   163]]
