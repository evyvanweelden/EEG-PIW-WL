#!/usr/bin/env python
# coding: utf-8

# In[2]:


#code created by Carl van Beek
#last updated on 16-10-2023
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score


# In[4]:


#load data
#Data is in long format with columns for EEG spectral power values (baseline-corrected or absolute), Order, Subject, Trial number, and Condition.

Alpha_norm_table = pd.read_csv("C:/Users/.../Table_alpha_baselinecorrectedvalues.csv")
Beta_norm_table = pd.read_csv("C:/Users/.../Table_beta_baselinecorrectedvalues.csv")
Theta_norm_table = pd.read_csv("C:/Users/.../Table_theta_baselinecorrectedvalues.csv")
Engagement_norm_table = pd.read_csv("C:/Users/.../Table_engagement_baselinecorrectedvalues.csv")

Alpha_absolute_table = pd.read_csv("C:/Users/.../Table_alpha_absolutevalues.csv")
Beta_absolute_table = pd.read_csv("C:/Users/.../Table_beta_absolutevalues.csv")
Theta_absolute_table = pd.read_csv("C:/Users/.../Table_theta_absolutevalues.csv")
Engagement_absolute_table = pd.read_csv("C:/Users/.../Table_engagement_absolutevalues.csv")


# In[5]:


#create tables containing only features
#new_duty_aggr_table = duty_aggr_table.drop(columns = ['Subject','Trial', 'Condition'])

new_Alpha_norm_table = Alpha_norm_table.drop(columns = ['Order','Subject','Trial', 'Condition'])
new_Beta_norm_table = Beta_norm_table.drop(columns = ['Order','Subject','Trial', 'Condition'])
new_Theta_norm_table = Theta_norm_table.drop(columns = ['Order','Subject','Trial', 'Condition'])
new_Engagement_norm_table = Engagement_norm_table.drop(columns = ['Order','Subject','Trial', 'Condition'])

new_Alpha_absolute_table = Alpha_absolute_table.drop(columns = ['Order','Subject','Trial', 'Condition'])
new_Beta_absolute_table = Beta_absolute_table.drop(columns = ['Order','Subject','Trial', 'Condition'])
new_Theta_absolute_table = Theta_absolute_table.drop(columns = ['Order','Subject','Trial', 'Condition'])
new_Engagement_absolute_table = Engagement_absolute_table.drop(columns = ['Order','Subject','Trial', 'Condition'])

#concatenate all features into a single feature table, making a distinction between a table with and without stick data
full_EEG_norm_table = pd.concat([new_Alpha_norm_table,new_Beta_norm_table,new_Theta_norm_table,new_Engagement_norm_table], axis=1)
#full_norm_table = pd.concat([full_EEG_norm_table,new_duty_aggr_table], axis = 1 )

full_EEG_absolute_table = pd.concat([new_Alpha_absolute_table,new_Beta_absolute_table,new_Theta_absolute_table,new_Engagement_absolute_table], axis=1)
#full_absolute_table = pd.concat([full_EEG_absolute_table,new_duty_aggr_table], axis = 1 )

#get features
y = Alpha_norm_table['Condition'] # prediction labels, they are the same in all tables, alpha_norm is just randomly used

function_test_table = pd.concat([full_EEG_absolute_table,y], axis = 1)
print(function_test_table)


# In[6]:


#Dataframe should have columns: 'Condition' and Alpha, Beta, Theta and Engagement values
#for all channels. In the case of 12 x 6 trials and 32 channels, this would be a 72 rows x 129 columns dataframe
#Order, Subject and Trial are not used in this function, but could be used in other applications
def workload_predictor(df):
    X_EEG = df.drop(columns = ['Condition'])
    #Make sure there is a column called 'Condition'. Could be replaced by an integer if there is a standard for that.
    y = df['Condition']
    #first create the RFE algorithm
    SVM = SVC(kernel = 'linear',C = 1,gamma = 0.1) #Choose a classifier to perform RFE
    SVM_RFE = RFE(SVM,n_features_to_select = 10) #Perform RFE
    SVM_RFE.fit(X_EEG,y) 

    #View which features got selected
    selected_EEG = X_EEG.columns[SVM_RFE.support_]
    print("Selected features: ", selected_EEG)


    #Transform the original X-data to only retain the selected features
    X_RFE_EEG = SVM_RFE.transform(X_EEG)

    #global value to keep track of the accuracy of each iteration
    all_accuracies_EEG = []
    all_f1_EEG = []
    all_precision_EEG = []
    all_recall_EEG = []

    #perform the training and testing 10 times, to rule out chance due to the small dataset
    for i in range(1,11):
        #split into 70-30 training and testing for RFE
        X_RFE_train_EEG, X_RFE_test_EEG, y_train, y_test = train_test_split(X_RFE_EEG, y, test_size = 0.3)

        #gridsearch for all models of the first level in the stacking classifier
        rf_params = {'n_estimators': [10, 25, 50, 100,200], 'max_depth': [None, 5, 10, 15]}
        RF_EEG = RandomForestClassifier() #Check literature
        rf_grid_EEG = GridSearchCV(RF_EEG, rf_params, cv=5, n_jobs=-1)
        rf_grid_EEG.fit(X_RFE_train_EEG,y_train)

        svm_params = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10, 100]}
        svm_EEG = SVC()
        svm_grid_EEG = GridSearchCV(svm_EEG, svm_params, cv=5, n_jobs=-1)
        svm_grid_EEG.fit(X_RFE_train_EEG,y_train)

        lr_params = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['none','l2']}
        lr_EEG = LogisticRegression()
        lr_grid_EEG = GridSearchCV(lr_EEG, lr_params, cv=5, n_jobs=-1)
        lr_grid_EEG.fit(X_RFE_train_EEG,y_train)

        #variable to input as the first level in the stacking classifier
        estimators_EEG = [('rf', rf_grid_EEG),('svc', svm_grid_EEG),('lr', lr_grid_EEG)]

        #gridsearch for the second level of the stacking model
        second_level_params_EEG = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10, 100]}
        second_level_model_EEG = SVC()
        second_level_grid_EEG = GridSearchCV(second_level_model_EEG, second_level_params_EEG, cv=5, n_jobs=-1)
        second_level_grid_EEG.fit(X_RFE_train_EEG,y_train)

        #Create the stacking classifier
        sclf_EEG = StackingClassifier(estimators=estimators_EEG, final_estimator=second_level_grid_EEG)
        sclf_EEG.fit(X_RFE_train_EEG, y_train)

        prediction = sclf_EEG.predict(X_RFE_test_EEG)
        #get model perfromance
        accuracy_EEG = accuracy_score(y_test, prediction)
        f1_EEG = f1_score(y_test, prediction)
        precision_EEG = precision_score(y_test, prediction)
        recall_EEG = recall_score(y_test, prediction)
        #add the performance metrics to their global list to obtain averages later
        all_accuracies_EEG.append(accuracy_EEG)
        all_f1_EEG.append(f1_EEG)
        all_precision_EEG.append(precision_EEG)
        all_recall_EEG.append(recall_EEG)
        #print out the metrics for the current iteration
        print("Accuracy test number ", i, ": ", accuracy_EEG)
        print("F1 score test number ", i, ": ", f1_EEG)
        print("Precision test number ", i, ": ", precision_EEG)
        print("Recall test number ", i, ": ", recall_EEG)
        #obtain the chosen hyperparameters
        print("Chosen parameters for the random forest model in the first level of the stacking classifier for iteration number ",i,": ",rf_grid_EEG.best_params_)
        print("Chosen parameters for the SVM model in the first level of the stacking classifier for iteration number ",i,": ",svm_grid_EEG.best_params_)
        print("Chosen parameters for the Logistic regression model in the first level of the stacking classifier for iteration number ",i,": ",lr_grid_EEG.best_params_)
        print("Chosen parameters for the SVM model in the second level of the stacking classifier for iteration number ",i,": ",second_level_grid_EEG.best_params_)

    #print out the average model evaluation scores of all iterations
    print("Average accuracy score", np.mean(all_accuracies_EEG))
    print("Average f1 score", np.mean(all_f1_EEG))
    print("Average precision", np.mean(all_precision_EEG))
    print("Average recall", np.mean(all_recall_EEG))

    


# In[7]:


workload_predictor(function_test_table)


# In[ ]:




