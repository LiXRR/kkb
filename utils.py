#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from KNN_KDE_Bagging_Imputer import KKBImputer
import numpy as np
import random
from knnxkde import KNNxKDE

# In[2]:


def select_param_rmse_b(complete_data,miss_data,paramB,params,paramk,paramh,paramK):
    #5-fold-corss validation,the last column must be the prediction!
    origi_data = complete_data.copy()
    mis_data = miss_data.copy()
    cv = KFold(n_splits=5, shuffle=True,random_state=42)
    best_rmse = float('inf') 
    result = []
    best_hyperparameters = None
    n,_ = complete_data.shape
    for param1 in paramB:
        print('B:',param1,end=' => ')
        
        for param2 in params:
            #print('s:',param2,end=' => ')
            
            for param3 in paramk:
                #print('k:',param3,end=' => ')
                
                for param4 in paramh:
                    #print('h:',param4,end='\r', flush=True)
                    
                    for param5 in paramK:
                        
                        rmse_list = []
                        for train_idx, val_idx in cv.split(origi_data):
                            train_fold_o, val_fold = origi_data[train_idx].copy(), origi_data[val_idx].copy()
                            imputer = KKBImputer(B=param1, s_ratio=param2, n_neighbors_ratio=param3, h=param4)
                            whole_data = np.vstack((train_fold_o,mis_data))
                            imp_data = imputer.impute_pattern(whole_data)
                            
                            k = int(round(param5*(imp_data.shape[0])))
                            knn_regressor = KNeighborsRegressor(n_neighbors=k)
                            knn_regressor.fit(imp_data[:,:-1],imp_data[:,-1])
                            y_pred = knn_regressor.predict(val_fold[:,:-1])
                            
                            rmse_fold = np.sqrt(mean_squared_error(val_fold[:,-1], y_pred))
                            rmse_list.append(rmse_fold)
                    # Calculate average MSE across folds for the current k
                        avg_rmse = np.mean(rmse_list)
                    # Update best_k and best_rmse if the current k has a lower MSE
                        if avg_rmse < best_rmse:
                            best_hyperparameters = {'B': param1, 's': param2, 'k': param3, 'h': param4,'K':param5}
                            best_rmse = avg_rmse
    print(best_hyperparameters,best_rmse)                    
    return best_hyperparameters,best_rmse    


# In[3]:


def knnxkde_param_rmse(complete_data,missing_data,paramt,paramh,paramK):
    mis_data = missing_data.copy()
    origi_data = complete_data.copy()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    best_rmse = float('inf') 
    result = []
    best_hyperparameters = None
    for param1 in paramt:
        print('tau:',param1,end=' => ')
        
        for param4 in paramh:

            for param5 in paramK:
                
                rmse_list = []
                for train_idx, val_idx in cv.split(origi_data):
                    train_fold_0, val_fold = origi_data[train_idx].copy(), origi_data[val_idx].copy()
                    train_fold = np.vstack((train_fold_0,mis_data))
                    
                    knnxkde = KNNxKDE(h=param4, tau=1.0/param1, metric='nan_std_eucl')
                    norm_imputed_samples = knnxkde.impute_samples(train_fold, nb_draws=1)#we only sample 1
                    
                    for (row, col), value in norm_imputed_samples.items():
                        train_fold[row, col] = value
                    
                    k = int(round(param5*(train_fold.shape[0])))
                    knn_regressor = KNeighborsRegressor(n_neighbors=k)
                    knn_regressor.fit(train_fold[:,:-1],train_fold[:,-1])
                    y_pred = knn_regressor.predict(val_fold[:,:-1])
                    
                    # Calculate mean squared error for this fold
                    rmse_fold = np.sqrt(mean_squared_error(val_fold[:,-1], y_pred))
                    rmse_list.append(rmse_fold)
            # Calculate average MSE across folds for the current k
                avg_rmse = np.mean(rmse_list)
            # Update best_k and best_rmse if the current k has a lower MSE
                if avg_rmse < best_rmse:
                    best_hyperparameters = {'tau': param1,  'h': param4,'K':param5}
                    best_rmse = avg_rmse
    return (best_hyperparameters,best_rmse) 


# In[4]:


def normalization(data, parameters=None):
    _, dim = data.shape
    norm_data = data.copy()
    
    if parameters is None:
  
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
    
        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
        # Return norm_parameters for renormalization
        norm_parameters = {"min_val": min_val, "max_val": max_val}

    else:
        min_val = parameters["min_val"]
        max_val = parameters["max_val"]
    
        # For each dimension
        for i in range(dim):
            norm_data[:,i] = norm_data[:,i] - min_val[i]
            norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
        norm_parameters = parameters
    
    return norm_data, norm_parameters



def renormalization(norm_data, norm_parameters):
    min_val = norm_parameters["min_val"]
    max_val = norm_parameters["max_val"]
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    
    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
    return renorm_data


# In[5]:


def introduce_miss(original_data,remain_rate):#we only consider MCAR
    n,d = original_data.shape
    select_index = np.random.choice(n,size = int(n*remain_rate),replace=False)
    com_data = original_data[select_index].copy()
    m_data = original_data[np.delete(np.arange(n),select_index)].copy()
    for i in range(m_data.shape[0]):
        j = np.random.randint(1,d-1)#impossible all features are 0
        miss_index = np.random.choice(d,j,replace=False)
        m_data[i,miss_index] = np.nan
      
    return com_data,m_data


# In[ ]:



