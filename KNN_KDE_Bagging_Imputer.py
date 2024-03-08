#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
from sklearn.neighbors import KDTree

# Select the rows matching the missing pattern
def select_receivers(miss_data, current_miss_pattern):
    (n, d) = miss_data.shape
    final_filter = np.ones(n).astype('bool')
    for i in range(d):
        cur_filter = (np.isnan(miss_data[:, i]) == current_miss_pattern[i])
        final_filter = np.logical_and(final_filter, cur_filter)
    id_receivers = np.where(final_filter)[0]
    return id_receivers

class KKBImputer:
    def __init__(self, B=10, s_ratio=0.6, n_neighbors_ratio=0.005, h=0.1, leaf_size=30, 
                 metric='euclidean', **kwargs):
        self.n_neighbors_ratio = n_neighbors_ratio
        self.B = B
        self.s_ratio = s_ratio
        self.h = h
        self.leaf_size = leaf_size
        self.metric = metric
        self.kwargs = kwargs
     
    def impute_pattern(self,X):
        # Split the array into two parts: complete data and missing data
        self.complete_data = X[~np.isnan(X).any(axis=1)].copy()
        self.missing_data = X[np.isnan(X).any(axis=1)].copy()
        self.s = int(round((self.s_ratio*self.complete_data.shape[0])))
        self.n_neighbors = int(round((self.n_neighbors_ratio*self.complete_data.shape[0])))
        # Find the missing patterns
        self.miss_patterns = np.unique(np.isnan(self.missing_data), axis=0)
        
        # Subsample complete data for B rounds
        self.subsampled_complete_data = [self.complete_data[
            np.random.choice(len(self.complete_data), int(self.s), replace=False)] 
                                         for _ in range(self.B)]
        
        for n, current_miss_pattern in enumerate(self.miss_patterns):
            if not np.logical_or.reduce(current_miss_pattern):  # if there is no missing value
                continue  # do nothing
            if np.logical_and.reduce(current_miss_pattern):  # if there are only missing values
                continue  # do nothing
                
            # Select the missing rows and columns with the missing pattern
            miss_pattern_rows = select_receivers(self.missing_data, current_miss_pattern)
            missing_columns = np.where(current_miss_pattern)[0]
            complete_columns = np.where(~current_miss_pattern)[0]
            
            kd_tree_dict = {}
            # Iterate over each subsample to build kd-tree
            for i, array in enumerate(self.subsampled_complete_data):
                # Build a KD-tree for the current sub-array
                kd_tree = KDTree(array[:,complete_columns],leaf_size=self.leaf_size, metric=self.metric, **self.kwargs)
                kd_tree_dict[f'kd_tree_{i}'] = kd_tree  
            for i in miss_pattern_rows:
                missing_row = self.missing_data[i,complete_columns]
                

                # Randomly choose a round for missing pattern
                round_idx = np.random.randint(self.B)
                
                round_kdtree = kd_tree_dict[f'kd_tree_{round_idx}']
                dists, indices = round_kdtree.query(missing_row.reshape(1,-1), k=self.n_neighbors)
                cov = np.diag(np.std(self.subsampled_complete_data[round_idx][indices[0]][:,missing_columns],axis = 0)**2*self.h)
                epsilon = np.random.multivariate_normal(np.zeros(len(missing_columns)),cov)
                
                K = np.random.randint(1,self.n_neighbors)
                kth_neighbor = self.subsampled_complete_data[round_idx][indices[0][K]][missing_columns].copy()
                
                self.missing_data[i,missing_columns] = epsilon+kth_neighbor
        return np.vstack((self.complete_data,self.missing_data))
    
    
    
    

    def impute_row(self,X):
        # Split the array into two parts: complete data and missing data
        self.complete_data = X[~np.isnan(X).any(axis=1)].copy()
        self.missing_data = X[np.isnan(X).any(axis=1)].copy()
        n,_ = self.missing_data.shape
        self.s = int(round((self.s_ratio*self.complete_data.shape[0])))
        self.n_neighbors = int(round((self.n_neighbors_ratio*self.complete_data.shape[0])))
        # Subsample complete data for B rounds
        self.subsampled_complete_data = [self.complete_data[
            np.random.choice(len(self.complete_data), int(self.s), replace=False)] 
                                         for _ in range(self.B)]
        
        for i in range(n):
            missing_row = self.missing_data[i]
            
            missing_columns = np.where(np.isnan(missing_row))[0]
            complete_columns = np.where(~np.isnan(missing_row))[0]
            
            # Randomly choose a round for missing pattern
            round_idx = np.random.randint(self.B)
            round_complete_data = self.subsampled_complete_data[round_idx].copy()
            
            # Build KD-tree for the round complete data using only non-nan columns
            round_kdtree = KDTree(round_complete_data[:,complete_columns], leaf_size=self.leaf_size, metric=self.metric, **self.kwargs)
            
            # Impute
            
            dists, indices = round_kdtree.query(missing_row[complete_columns].reshape(1,-1), k = int(self.n_neighbors))
            cov = np.diag(np.std(round_complete_data[indices[0]][:,missing_columns],axis = 0)**2*self.h)
            epsilon = np.random.multivariate_normal(np.zeros(len(missing_columns)),cov)
            
            K = np.random.randint(1,self.n_neighbors)
            kth_neighbor = round_complete_data[indices[0][K]][missing_columns].copy()
            
            self.missing_data[i,missing_columns] = epsilon+kth_neighbor
        return np.vstack((self.complete_data,self.missing_data))
    
    
#     def impute_brute_row(self,X):
#         # Split the array into two parts: complete data and missing data
#         self.complete_data = X[~np.isnan(X).any(axis=1)].copy()
#         self.missing_data = X[np.isnan(X).any(axis=1)].copy()
        
        
#         # Subsample complete data for B rounds
#         self.subsampled_complete_data = [self.complete_data[
#             np.random.choice(len(self.complete_data), int(self.s), replace=False)] 
#                                          for _ in range(self.B)]
        
#         print('B',self.subsampled_complete_data)
#         n,_ = self.missing_data.shape
#         for i in range(n):
#             missing_row = self.missing_data[i]
#             print(missing_row)
#             miss_columns = np.where(np.isnan(missing_row))[0]
#             complete_columns = np.where(~np.isnan(missing_row))[0]
       
#             # Randomly choose a round for missing pattern
#             round_idx = np.random.randint(self.B)
#             round_complete_data = self.subsampled_complete_data[round_idx].copy()
#             print('chosen round:',round_idx)
#             # Caculate distances
#             dis = ((round_complete_data[:,complete_columns]-missing_row[complete_columns])**2).sum(axis = 1)
#             print('distance:',dis)
#             r = np.random.randint(1,self.n_neighbors)
#             print('k:',r)
#             bnn = np.where(dis == np.sort(dis)[r-1])[0][0]#if same diatances,we only choose the first one
#             print('neighbour index:',bnn)
#             cov = np.diag(self.h[miss_columns])
#             epsilon = np.random.multivariate_normal(np.zeros(len(miss_columns)),cov)
            
#             # IMpute
#             self.missing_data[i,miss_columns] = round_complete_data[bnn,miss_columns]+epsilon
            
           
        
#         return np.vstack((self.complete_data,self.missing_data))
    
        
        


# In[ ]:




