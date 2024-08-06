# Import the necessary packages
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from scipy.stats import t 
from imblearn.over_sampling import BorderlineSMOTE
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

# Check if the number of pos and neg samples is correct
# print('label = 1:', sum(trainY))
# print('label = 0:', len(trainY) - sum(trainY))

# Bootstrap process definition
N = 500 # Bootstrap iteration count
n_sample = 416 # Total number of training set

def boostrapTraining(N, model, data, y, n_sample):
      test_precision = []
      test_sensitivity = []
      test_specificity = []
      test_F1 = []
      test_auc = []
      test_acc = []
      # Rank the best models based on AUC
      model_auc = []

      for _ in range(N):
            # Set the random seed for reproducibility
            random.seed(33)
            
            # Randomly sample 70% of the training set (e.g., 491 * 0.7 = 291)
            indices = np.random.choice(range(n_sample), size=291, replace=True)
            X_bootstrap, y_bootstrap = data.iloc[indices], y.iloc[indices]
            
            # Apply BorderlineSMOTE to increase the positive samples to achieve a 1:1 ratio
            smote = BorderlineSMOTE()
            X_bor, y_bor = smote.fit_resample(X_bootstrap, y_bootstrap)

            # Check if the number of pos and neg samples is correct
            # print('sample label = 1:', sum(y_bor))
            # print('sample label = 0:', len(y_bor) - sum(y_bor))

            # Fit the model with the resampled data
            model.fit(X_bor, y_bor)

            # Identify the samples not included in the bootstrap sample (out-of-bag samples)
            indices_set = set(indices)
            all_numbers = set(range(n_sample))
            missing_numbers = all_numbers - indices_set
            missing_numbers_list = np.array(sorted(list(missing_numbers)))

            # Use the remaining 30% of the data for testing
            test_data, test_y = data.iloc[missing_numbers_list], y.iloc[missing_numbers_list]

            # Predict the test set
            test_pred = model.predict(test_data)
            
            # print the results
            print(metrics.classification_report(test_y, test_pred))
            Pre = metrics.precision_score(test_y, test_pred)
            print('Precision_predict:',Pre)
            Sen = metrics.recall_score(test_y, test_pred)
            print('Sensitivity_predict:',Sen)
            Spe = metrics.recall_score(test_y, test_pred, pos_label=0)
            print('Specificity_predict:',Spe)
            F1 = metrics.f1_score(test_y, test_pred)
            print('F1_predict:',F1)
            Acc = metrics.accuracy_score(test_y, test_pred)
            print('accuracy_predict:',Acc)
            predicted_p = model.predict_proba(test_data)[:,1]
            AUC = roc_auc_score(test_y, predicted_p)
            print('auc_predict:',roc_auc_score(test_y, predicted_p),"\n")
                  
            # Save each result
            model_auc.append((model, AUC))

            test_acc.append(Acc)
            test_auc.append(AUC)
            test_F1.append(F1)
            test_sensitivity.append(Sen)
            test_precision.append(Pre)
            test_specificity.append(Spe)

            all_pre_m = round(np.mean(test_precision), 3)
            all_sen_m = round(np.mean(test_sensitivity), 3)
            all_spe_m = round(np.mean(test_specificity), 3)
            all_f1_m = round(np.mean(test_F1), 3)
            all_acc_m = round(np.mean(test_acc), 3)
            all_auc_m = round(np.mean(test_auc), 3)
                  
            pre_m = np.mean(test_precision)
            sen_m = np.mean(test_sensitivity)
            spe_m = np.mean(test_specificity)
            f1_m = np.mean(test_F1)
            acc_m = np.mean(test_acc)
            auc_m = np.mean(test_auc)
                  
            pre_s = np.std(test_precision)
            sen_s = np.std(test_sensitivity)
            spe_s = np.std(test_specificity)
            f1_s = np.std(test_F1)
            acc_s = np.std(test_acc)
            auc_s = np.std(test_auc)
            
            # Calculate the 95% confidence interval
            dof = len(test_precision)-1
            confidence = 0.95
            t_crit = np.abs(t.ppf((1-confidence)/2,dof))
                  
            pre_CI = (round(pre_m-pre_s*t_crit/np.sqrt(len(test_precision)), 3), round(pre_m+pre_s*t_crit/np.sqrt(len(test_precision)), 3)) 
            sen_CI = (round(sen_m-sen_s*t_crit/np.sqrt(len(test_sensitivity)), 3), round(sen_m+sen_s*t_crit/np.sqrt(len(test_sensitivity)), 3))
            spe_CI = (round(spe_m-spe_s*t_crit/np.sqrt(len(test_specificity)), 3), round(spe_m+spe_s*t_crit/np.sqrt(len(test_specificity)), 3))
            f1_CI = (round(f1_m-f1_s*t_crit/np.sqrt(len(test_F1)), 3), round(f1_m+f1_s*t_crit/np.sqrt(len(test_F1)), 3))
            acc_CI = (round(acc_m-acc_s*t_crit/np.sqrt(len(test_acc)), 3), round(acc_m+acc_s*t_crit/np.sqrt(len(test_acc)), 3))
            auc_CI = (round(auc_m-auc_s*t_crit/np.sqrt(len(test_auc)), 3), round(auc_m+auc_s*t_crit/np.sqrt(len(test_auc)), 3))

            # Final model performance
            print(f'Precision: {all_pre_m} {pre_CI}')
            print(f'Sensitivity: {all_sen_m} {sen_CI}')
            print(f'Specificity: {all_spe_m} {spe_CI}')
            print(f'F1 score: {all_f1_m} {f1_CI}')
            print(f'Accuracy: {all_acc_m} {acc_CI}')
            print(f'AUC: {all_auc_m} {auc_CI}')
                
      return sorted(model_auc, reverse=True) # Rank the best models based on AUC (The best model is at index 0)

model_RF = boostrapTraining(N, cf_RF, trainingSetZ, trainY, n_sample)
model_SVM = boostrapTraining(N, cf_SVM, trainingSetZ, trainY, n_sample)
model_CAT = boostrapTraining(N, cf_CAT, trainingSetZ, trainY, n_sample)
model_XGB = boostrapTraining(N, cf_XGB, trainingSetZ, trainY, n_sample)

# best model (Access the best model out of the 500 training iterations)
best_RF = model_RF[0][0]
best_SVM = model_SVM[0][0]
best_CAT = model_CAT[0][0]
best_XGB = model_XGB[0][0]

# internal val
predicted = best_RF.predict(inValidatingSetZ)
print(metrics.classification_report(inValidY, predicted))

Pre = metrics.precision_score(inValidY, predicted)
print('Precision_predict:',Pre)
Sen = metrics.recall_score(inValidY, predicted)
print('Sensitivity_predict:',Sen)
Spe = metrics.recall_score(inValidY, predicted, pos_label=0)
print('Specificity_predict:',Spe)
F1 = metrics.f1_score(inValidY, predicted)
print('F1_predict:',F1)
Acc = metrics.accuracy_score(inValidY, predicted)
print('accuracy_predict:',Acc)
yPred = best_RF.predict_proba(inValidatingSetZ)[:,1]
print('auc_predict:',roc_auc_score(inValidY, yPred),"\n")

# external test
exdata = pd.read_excel('G:\\My Drive\\ym\\about_lab\\SMD_ML\\CCH_OVA_2023data_anony_provide20230808.xlsx')
dataX = exdata[exdata['En']==0]

yX = dataX['SMD_cate5'].tolist()

dataX = dataX.drop(columns=['En','SMD_cate5','SMD_1','SMD_2','aSMD_change','rSMD_change','Post-BMI',
                            'Post-albumin','Post-NLR','Post-PNI','Post-PLR','Pre-PNI','PNI change'])
dataX = dataX.set_index('ID_C')

# Standardize the external test set using the mean and standard deviation of the training set
dataXZ = z.transform(dataX)
dataXZ = pd.DataFrame(dataXZ, columns=dataX.columns)

predicted = best_RF.predict(dataXZ)
print(metrics.classification_report(yX, predicted))

Pre = metrics.precision_score(yX, predicted)
print('Precision_predict:',Pre)
Sen = metrics.recall_score(yX, predicted)
print('Sensitivity_predict:',Sen)
Spe = metrics.recall_score(yX, predicted, pos_label=0)
print('Specificity_predict:',Spe)
F1 = metrics.f1_score(yX, predicted)
print('F1_predict:',F1)
Acc = metrics.accuracy_score(yX, predicted)
print('accuracy_predict:',Acc)
yPred = best_RF.predict_proba(dataXZ)[:,1]
print('auc_predict:',roc_auc_score(yX, yPred),"\n")

# Import the SHAP library for model explanation
import shap
shap.initjs()

# Assuming 'best_SVM' and 'best_RF' are already defined models
# If your model is kernel-based (e.g., SVM)
model = best_SVM
explainer = shap.KernelExplainer(model.predict, inValidatingSetZ)
shapValues = explainer.shap_values(inValidatingSetZ)

# If your model is tree-based (e.g., Random Forest)
model = best_RF
explainer = shap.TreeExplainer(model)
shapValues = explainer.shap_values(inValidatingSetZ)

# Generate SHAP summary plot
shap.summary_plot(shapValues, inValidatingSetZ)

# Generate SHAP bar plot for top 10 features
shap.summary_plot(shapValues, inValidatingSetZ, plot_type='bar', max_display=10, sort=True)

# Define target features for dependence plots
tg = ["Residual disease",
      "Ascites",
      "Pre-albumin",
      "Albumin change",
      "Pre-NLR",
      "Pre-PLR"
    ]

# Generate SHAP dependence plots for specified features
for i in range(3):
    shap.dependence_plot("Residual disease", 
                         shapValues,
                         inValidatingSet, 
                         feature_names=inValidatingSet.columns,
                         interaction_index=tg[0], 
                         alpha=0.8,
                         show=False)                       

# Import math library for log odds conversion
import math

# List of sample IDs to be plotted in force plots
list_id = [790, 783, 771, 681, 629, 612, 748, 728, 679]

# Function to convert log odds to probabilities
def logodds_to_prob(logit):
        odds = math.exp(logit)
        return odds / (1 + odds)

# Function to generate and save SHAP force plots
def shap_plot_save(model):
    for j in range(0, 1000): 
        if trainingSetZ.index[j] in list_id:
            print(str(trainingSetZ.index[j]))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(trainingSetZ)
            shap_values = np.round(shap_values, 4)

            shap.force_plot(explainer.expected_value, 
                            shap_values[j, :], 
                            features=round(trainingSet.iloc[j, :], 2), 
                            matplotlib=True,
                            show=True)
            plt.tight_layout()

# Call the function to generate and save SHAP force plots for the given model
shap_plot_save(model)
