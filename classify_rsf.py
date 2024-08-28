# This is the main code used for training and testing random survival forest models for the paper
# "Machine learning predicts phenoconversion from poly-somnography in isolated REM sleep behavior disorder"

#%% Necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import brier_score, integrated_brier_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
import seaborn as sns
from feature_engine.selection import SmartCorrelatedSelection
from utils import kfold_indices
import argparse
import os

#%% Read the input params
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fName')
parser.add_argument('-c', '--controls')
args = parser.parse_args()
fName = args.fName
controls = args.controls
#%% Load the database
database = ## Path to the database of the features
database.dropna(inplace=True)

outcomes = database[["conversion","time"]]
features = database.copy()
features.drop(labels = ["id","group","time","conversion"],axis = 1,inplace=True)

#%% Define the random states and the hyperparameters
k = 4
random_states = np.arange(1,11)
n_estimators = 100
min_samples_split = 6
min_samples_leaf = 3
AUC_points = np.linspace(24,10*12,num=9)

n_features = np.arange(5,features.shape[1]+1,step=5)

# Prepare the arrays where the test performances will be saved 
c_index_Harrel = np.empty((random_states.size,k,n_features.size))
c_index_Harrel[:] = np.nan

c_index_Uno = np.empty((random_states.size,k,n_features.size))
c_index_Uno[:] = np.nan

mean_AUC = np.empty((random_states.size,k,n_features.size))
mean_AUC[:] = np.nan

AUC_values = np.empty((random_states.size,k,n_features.size,AUC_points.size))
AUC_values[:] = np.nan

integrated_brier = np.empty((random_states.size,k,n_features.size))
integrated_brier[:] = np.nan

mean_importance = np.empty((features.shape[1],random_states.size,k,n_features.size))
mean_importance[:] = np.nan

#%% Loop
for rs in random_states:
    print('Random state ', rs)
    
    skf = StratifiedKFold(n_splits=k,shuffle=True,random_state=rs)
    this_k = -1
    
    for train_index, test_index in skf.split(features, outcomes['conversion']):
        this_k = this_k+1;
        X_train = features.copy()
        X_test = features.copy()
        X_train.drop(X_train.index[test_index],inplace=True)
        X_test.drop(X_test.index[train_index],inplace=True)
        y_train = outcomes.copy()
        y_test = outcomes.copy()
        y_train.drop(y_train.index[test_index],inplace=True)
        y_test.drop(y_test.index[train_index],inplace=True)
        
        
        # Prepare output for the training and test
        y_train2 = y_train.to_numpy() # Convert to array
        y_train3 = np.core.records.fromarrays(y_train2.transpose(), 
                                                     names='conversion, time',formats='bool,int64')
        y_test2 = y_test.to_numpy() # Convert to array
        y_test3 = np.core.records.fromarrays(y_test2.transpose(), 
                                                     names='conversion, time',formats='bool,int64')
        
        # Remove correlated features in test set
        scs = SmartCorrelatedSelection(threshold=0.8,method='spearman',
                                       selection_method='variance')
        scs.fit_transform(X_train)
        features_to_remove = scs.features_to_drop_
        
        X_train_reduced = X_train.drop(features_to_remove,axis=1)
        X_test_reduced = X_test.drop(features_to_remove,axis=1)
        
        names_features_all = list(X_train.columns.values)
        names_features_reduced = list(X_train_reduced.columns.values)
        both = list(set(names_features_all).intersection( names_features_reduced))
        idx_kept = [names_features_all.index(x) for x in both]
        idx_kept= np.sort(idx_kept)
      
        # Use the training set to order the features according to their importance   
        coxph = CoxPHSurvivalAnalysis(alpha=1e-2).fit(X_train_reduced, y_train3)
        sorted_indices = np.argsort(-np.abs(coxph.coef_)) #Make negative to sort as descending
        
        this_n_features = np.arange(5,X_train_reduced.shape[1]+1,step=5)
        
        for nf in this_n_features:
            #print('Number of features ', nf)
            #Get a smaller subset of features
            X_train_subset = X_train_reduced.iloc[:,sorted_indices[0:nf]]
            X_test_subset = X_test_reduced.iloc[:,sorted_indices[0:nf]]
        
            # Train the RSF
            rsf = RandomSurvivalForest(n_estimators = n_estimators, 
                                       min_samples_leaf=min_samples_leaf,
                                       min_samples_split=min_samples_split,
                                       random_state=rs,
                                       n_jobs=-1)
            this_rsf_trained = rsf.fit(X_train_subset,y_train3)
    
            #Get the metrics 
            c_index_Harrel[rs-1,this_k,np.where(n_features==nf)] = this_rsf_trained.score(X_test_subset,y_test3)
            c_index_Uno[rs-1,this_k,np.where(n_features==nf)] = concordance_index_ipcw(y_train3, y_test3, this_rsf_trained.predict(X_test_subset),tau=np.max(outcomes['time']))[0]
            
            prob = this_rsf_trained.predict_survival_function(X_test_subset,return_array=True)
            times = this_rsf_trained.unique_times_
            max_test_time = np.max(y_test3['time'])-1
            min_test_time = np.min(y_test3['time'])+1
            keep_times = np.logical_and(times<=max_test_time,times>=min_test_time) 
            times_c = times.copy()
            prob_c = prob.copy()
            times_c = times_c[keep_times]
            prob_c = prob_c[:,keep_times]
            integrated_brier[rs-1,this_k,np.where(n_features==nf)] = integrated_brier_score(y_train3,y_test3,prob_c,times_c)
            
            auc = []
            for a in enumerate(AUC_points):
                if (a[1]>=np.min(y_test3['time']) and a[1]<np.max(y_test3['time'])):
                    aa = cumulative_dynamic_auc(y_train3, y_test3, this_rsf_trained.predict(X_test_subset), a[1])
                    auc.append(aa[1])
                else:
                    auc.append(np.nan)                  
                    
            mean_AUC[rs-1,this_k,np.where(n_features==nf)] = np.nanmean(auc)
            AUC_values[rs-1,this_k,np.where(n_features==nf),:] = auc
        
            #Get the feature importance
            result_importance = permutation_importance(
                this_rsf_trained, X_test_subset, y_test3, n_repeats=15, random_state=rs)
            mean_importance[idx_kept[sorted_indices[0:nf]],rs-1,this_k,np.where(n_features==nf)] = result_importance['importances_mean']
    
    
    
#%% Make plots 
#Reshape the results to have the random states and the k of the cross-validation mixed
c_index_Harrel =  np.reshape(c_index_Harrel,(len(random_states)*k,len(n_features)))
c_index_Uno = np.reshape(c_index_Uno,(len(random_states)*k,len(n_features)))
integrated_brier = np.reshape(integrated_brier,(len(random_states)*k,len(n_features)))
mean_AUC = np.reshape(mean_AUC,(len(random_states)*k,len(n_features)))
cm = 1/2.54  # centimeters in inches
f, axes = plt.subplots(4,1)
f.set_size_inches(15*cm,30*cm)
ax1 = sns.pointplot(data=c_index_Harrel,ax=axes[0])
ax1.set_xticklabels(n_features)
ax1.set_xlabel('Number of features')
ax1.set_ylabel('C-index (Harrel)')
ax2 = sns.pointplot(data=c_index_Uno,ax=axes[1])
ax2.set_xticklabels(n_features)
ax2.set_xlabel('Number of features')
ax2.set_ylabel('C-index (Uno)')
ax3 = sns.pointplot(data=mean_AUC,ax=axes[2])
ax3.set_xticklabels(n_features)
ax3.set_xlabel('Number of features')
ax3.set_ylabel('Mean AUC')    
ax4 = sns.pointplot(data=integrated_brier,ax=axes[3])
ax4.set_xticklabels(n_features)
ax4.set_xlabel('Number of features')
ax4.set_ylabel('Integrated Brier score')   
# Get the number of features leading to the best aveage C index value
avg_c_index_Harrel =  np.mean(np.reshape(c_index_Harrel,(len(random_states)*k,len(n_features))),axis=0)  
avg_c_index_Uno = np.mean(np.reshape(c_index_Uno,(len(random_states)*k,len(n_features))),axis=0)
best_index = np.nanargmax((avg_c_index_Uno+avg_c_index_Harrel)/2)   
      
#For the best values, make the AUC plot
best_AUC_values = np.squeeze(AUC_values[:,:,best_index,:])
best_AUC_values = np.reshape(best_AUC_values,(len(random_states)*k,len(AUC_points)))
f,axes = plt.subplots(1,1)
ax1 = sns.pointplot(data=best_AUC_values)
ax1.set_xticklabels(np.round(AUC_points/12,decimals = 1))
ax1.set_xlabel('Years')
ax1.set_ylabel('AUC')
   
#%% Feature importance
# We take the feature importance for the best models 
feature_importance = np.squeeze(mean_importance[:,:,:,best_index])
feature_importance = np.reshape(feature_importance,(np.shape(mean_importance)[0],len(random_states)*k))
mean_feature_importance = np.empty((feature_importance.shape[0],1))
for n in np.arange(feature_importance.shape[0]):
    thisRow = feature_importance[n,:]
    howManyReal = np.count_nonzero(~np.isnan(thisRow))
    mean_feature_importance[n] = np.nanmean(thisRow)*(100*howManyReal/thisRow.size)

   
indices_sorted = np.argsort(np.squeeze(mean_feature_importance))
sorted_mean_feature_importance = mean_feature_importance[indices_sorted]
sorted_features_columns = features.columns[indices_sorted]
#Find nans and remove them
idx_na = np.where(np.isnan(sorted_mean_feature_importance))
sorted_mean_feature_importance = np.delete(sorted_mean_feature_importance,idx_na)
sorted_features_columns = np.delete(sorted_features_columns,idx_na)

plt.subplots(figsize=(15*cm, 35*cm))

plt.barh(y=np.arange(0,np.squeeze(sorted_mean_feature_importance).size),width=np.squeeze(sorted_mean_feature_importance),tick_label=sorted_features_columns)
  

#%% Save the outputs for the reproduction of the graphs
import pickle
x = 'outputs_' + fName + '_no_cont.pkl'

with open(x, 'wb') as f:  
    pickle.dump([random_states,n_estimators,n_features,
    c_index_Harrel,c_index_Uno,integrated_brier,AUC_points,AUC_values,mean_AUC,mean_importance,
    features,outcomes,k],f)
