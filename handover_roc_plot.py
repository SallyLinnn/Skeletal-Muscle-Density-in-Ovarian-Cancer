# Import the necessary packages
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC                                                                                                                                                                             
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

import random
random.seed(33)

# Load the data from an Excel file into a DataFrame
data = pd.read_excel('G:\\My Drive\\ym\\about_lab\\SMD_ML\\SMD_analysis_1226\\MMH_OVA_SMD provide20230706_model_1.xlsx')
# Filter the dataset for training where 'Year_1' is 0
trainingSet = data[data['Year_1'] == 0]
# Filter the dataset for validation where 'Year_1' is 1
inValidatingSet = data[data['Year_1'] == 1]
# Extract the target variable for training set
trainY = trainingSet['SMD_cate5']
# Extract the target variable for validation set
inValidY = inValidatingSet['SMD_cate5']
# Drop unnecessary columns from the training set
trainingSet = trainingSet.drop(columns=['Year_1', 'Enroll', 'Height', 'BW_1', 'BW_2',
                                        'Post—BMI', 'Post-albumin', 'Post-NLR', 'Post-PNI',
                                        'Post-PLR', 'SMD_cate5', 'Pre-PNI', 'PNI change'])
# Set 'ID' as the index for the training set
trainingSet = trainingSet.set_index('ID')
# Drop unnecessary columns from the validation set
inValidatingSet = inValidatingSet.drop(columns=['Year_1', 'Enroll', 'Height', 'BW_1', 'BW_2',
                                                'SMD_cate5', 'Post—BMI', 'Post-albumin',
                                                'Post-NLR', 'Post-PNI', 'Post-PLR', 'Pre-PNI', 'PNI change'])
# Set 'ID' as the index for the validation set
inValidatingSet = inValidatingSet.set_index('ID')

# z-score
# Standardize the training set
z = StandardScaler()
trainingSetZ = z.fit_transform(trainingSet)
trainingSetZ = pd.DataFrame(trainingSetZ, columns=trainingSet.columns, index=trainingSet.index)

# Standardize the internal validation set using the mean and standard deviation of the training set
inValidatingSetZ = z.transform(inValidatingSet)
inValidatingSetZ = pd.DataFrame(inValidatingSetZ, columns=inValidatingSet.columns, index=inValidatingSet.index)

# Build the model
cf_RF = RandomForestClassifier()
cf_SVM = SVC(probability=True)
cf_XGB = XGBClassifier()
cf_CAT = CatBoostClassifier()

# Bootstrap process definition
N = 500 # Bootstrap iteration count
n_sample = 416 # Total number of training set

# ROC plots
"""### ROC curve (RF)"""

# Initialize lists to store true positive rates and AUC values for Random Forest
tprs_RF = []
aucs_RF = []
# Create a base false positive rate array for interpolation
base_fpr = np.linspace(0, 1, 501)
# Number of bootstrap iterations
N = 100
# Number of samples in the dataset (replace this with the actual number of samples)
n_sample = len(trainingSetZ)
# Random Forest classifier (assume it's defined elsewhere in your code)
cf_RF = RandomForestClassifier()
# Perform bootstrap resampling and model evaluation
for _ in range(N):
    # Generate bootstrap samples
    indices = np.random.choice(range(n_sample), size=291, replace=True)
    X_bootstrap, y_bootstrap = trainingSetZ.iloc[indices], trainY.iloc[indices]

    # Apply BorderlineSMOTE for oversampling
    smote = BorderlineSMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_bootstrap, y_bootstrap)

    # Fit the Random Forest model
    cf_RF.fit(X_resampled, y_resampled)
    
    # Identify the samples not included in the bootstrap sample (out-of-bag samples)
    indices_set = set(indices)
    all_numbers = set(range(n_sample))
    missing_numbers = all_numbers - indices_set
    missing_numbers_list = np.array(sorted(list(missing_numbers)))

    # Create the test set from out-of-bag samples
    test_data, test_y = trainingSetZ.iloc[missing_numbers_list], trainY.iloc[missing_numbers_list]
    
    # Predict probabilities for the test set
    predicted_RF_p = cf_RF.predict_proba(test_data)[:, 1]
    
    # Compute the ROC curve and AUC for the test set
    fpr, tpr, _ = roc_curve(test_y, predicted_RF_p)
    roc_auc = auc(fpr, tpr)
    aucs_RF.append(roc_auc)
    
    # Interpolate the true positive rates at the base false positive rates
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs_RF.append(tpr)

# Convert list of TPRs to a NumPy array
tprs_RF = np.array(tprs_RF)

# Compute the mean and standard deviation of the TPRs
mean_tprs = tprs_RF.mean(axis=0)
std = tprs_RF.std(axis=0)

# Compute the mean AUC
mean_auc = auc(base_fpr, mean_tprs)

# Compute the upper and lower bounds for the TPRs
tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

# Plot the mean ROC curve with shaded areas for the standard deviation
plt.figure(dpi=100)
plt.plot(base_fpr, 
         mean_tprs, 
         color="deeppink",
         linewidth=1,
         label='RF (AUC = %0.3f)' % (mean_auc))

"""### ROC curve (SVM)"""

tprs_SVM = []
aucs_SVM = []
base_fpr = np.linspace(0, 1, 501)

for _ in range(N):
    indices = np.random.choice(range(n_sample), size=291, replace=True)
    X_bootstrap, y_bootstrap = trainingSetZ.iloc[indices], trainY.iloc[indices]

    smote = BorderlineSMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_bootstrap, y_bootstrap) # type: ignore

    cf_SVM.fit(X_resampled, y_resampled)
    
    # test
    indices_set = set(indices)
    all_numbers = set(range(n_sample))
    missing_numbers = all_numbers - indices_set
    missing_numbers = all_numbers - indices_set
    missing_numbers_list = np.array(sorted(list(missing_numbers)))

    test_data, test_y = trainingSetZ.iloc[missing_numbers_list], trainY.iloc[missing_numbers_list]
    
    predicted_SVM_p =cf_SVM.predict_proba(test_data)[:,1]
    fpr, tpr, _ = roc_curve(test_y, predicted_SVM_p)
    
    roc_auc = auc(fpr, tpr)
    aucs_SVM.append(roc_auc)
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs_SVM.append(tpr)
    
tprs_SVM = np.array(tprs_SVM)
mean_tprs = tprs_SVM.mean(axis=0)
std = tprs_SVM.std(axis=0)

mean_auc = auc(base_fpr, mean_tprs)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, 
         mean_tprs, 
         color="aqua",
         linewidth=1, 
         label='SVM (AUC = %0.3f)' % (mean_auc),)

"""### ROC curve (XGB)"""

tprs_XGB = []
aucs_XGB = []
base_fpr = np.linspace(0, 1, 501)

for _ in range(N):
    indices = np.random.choice(range(n_sample), size=291, replace=True)
    X_bootstrap, y_bootstrap = trainingSetZ.iloc[indices], trainY.iloc[indices]

    smote = BorderlineSMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_bootstrap, y_bootstrap) # type: ignore

    cf_XGB.fit(X_resampled, y_resampled)
    
    # test
    indices_set = set(indices)
    all_numbers = set(range(n_sample))
    missing_numbers = all_numbers - indices_set
    missing_numbers = all_numbers - indices_set
    missing_numbers_list = np.array(sorted(list(missing_numbers)))

    test_data, test_y = trainingSetZ.iloc[missing_numbers_list], trainY.iloc[missing_numbers_list]
    
    predicted_XGB_p =cf_XGB.predict_proba(test_data)[:,1]
    fpr, tpr, _ = roc_curve(test_y, predicted_XGB_p)
    
    roc_auc = auc(fpr, tpr)
    aucs_XGB.append(roc_auc)
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs_XGB.append(tpr)
    
tprs_XGB = np.array(tprs_XGB)
mean_tprs = tprs_XGB.mean(axis=0)
std = tprs_XGB.std(axis=0)

mean_auc = auc(base_fpr, mean_tprs)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, 
         mean_tprs, 
         color="darkorange",
         linewidth=1,  
         label='XGB (AUC = %0.3f)' % (mean_auc),)

"""### ROC curve (Cat)"""

tprs_CAT = []
aucs_CAT = []
base_fpr = np.linspace(0, 1, 501)

for _ in range(N):
    indices = np.random.choice(range(n_sample), size=291, replace=True)
    X_bootstrap, y_bootstrap = trainingSetZ.iloc[indices], trainY.iloc[indices]

    smote = BorderlineSMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_bootstrap, y_bootstrap) # type: ignore

    cf_CAT.fit(X_resampled, y_resampled)
    
    # test
    indices_set = set(indices)
    all_numbers = set(range(n_sample))
    missing_numbers = all_numbers - indices_set
    missing_numbers = all_numbers - indices_set
    missing_numbers_list = np.array(sorted(list(missing_numbers)))

    test_data, test_y = trainingSetZ.iloc[missing_numbers_list], trainY.iloc[missing_numbers_list]
    
    predicted_CAT_p =cf_CAT.predict_proba(test_data)[:,1]
    fpr, tpr, _ = roc_curve(test_y, predicted_CAT_p)
    
    roc_auc = auc(fpr, tpr)
    aucs_CAT.append(roc_auc)
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs_CAT.append(tpr)
    
tprs_CAT = np.array(tprs_CAT)
mean_tprs = tprs_CAT.mean(axis=0)
std = tprs_CAT.std(axis=0)

mean_auc = auc(base_fpr, mean_tprs)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, 
         mean_tprs, 
         color="cornflowerblue",
         linewidth=1,  
         label='CatBoost (AUC = %0.3f)' % (mean_auc),)


plt.plot([0, 1], [0, 1], linestyle = '--', lw = 1.5, color = 'r')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend(loc="lower right")
plt.title('Training set')
plt.grid(True)
plt.savefig('boostrap_tr.png')
plt.show()