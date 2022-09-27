import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle
import os
import datetime
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
import gc
import argparse
import utils
import seaborn as  sns

gc.collect()

parser = argparse.ArgumentParser()
parser.add_argument("--use_saved_params", type=int)
parser.add_argument("--saved_params", type=str, default='')
parser.add_argument("--result", type=str)
args = parser.parse_args()

#dataset = "dataset_M.csv"
#dataset = "dataset_Z.csv"
dataset = "dataset_Z+M.csv"
best_params = "best_params"
if args.saved_params:
    best_params_path = best_params + '/' + args.saved_params
    print(best_params_path)
    
output = "output"
name = "/plots"
test_ratio = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

if not os.path.exists(output + name):
    os.makedirs(output + name)
test_size = 0.2
random_state = 42 # 132

# load dataset
df = pd.read_csv(dataset)

# preprocess only if
df = df.rename(columns = lambda x:re.sub('-1.0', 'm1.0', x))
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# split data to x(features) and y(targets)
not_dft = [i for i in df.columns if i != 'k_dft' and i != 'g_dft']
x = df[not_dft]
x = x[x.columns[1:]]
y_k = df[['k_dft']]
y_g = df[['g_dft']]

y_k_sorted = y_k.sort_values(by=['k_dft']).reset_index(drop=True)
y_g_sorted = y_g.sort_values(by=['g_dft']).reset_index(drop=True)

k_binsplits = np.linspace(y_k_sorted.k_dft[8],y_k_sorted.k_dft[863],5)
g_binsplits = np.linspace(y_g_sorted.g_dft[8],y_g_sorted.g_dft[863],5)

k_categorized = np.digitize(y_k, bins=k_binsplits)
g_categorized = np.digitize(y_g, bins=g_binsplits)

# # train test split
# x_train, x_test, y_train_k, y_test_k = train_test_split(x, y_k, stratify=k_categorized, test_size=test_size, random_state=random_state)
# _, _, y_train_g, y_test_g = train_test_split(x, y_g, stratify=g_categorized, test_size=test_size, random_state=random_state)

if args.use_saved_params:
    with open(best_params_path + '/saved_best_params_k.pkl', 'rb') as f:
           loaded_params_k = pickle.load(f)
    with open(best_params_path + '/saved_best_params_g.pkl', 'rb') as f:
           loaded_params_g = pickle.load(f)
else:
    loaded_params_k = {'boosting_type': 'gbdt',
                        'learning_rate': 0.1,
                        'max_bin': 70,
                        'max_depth': 5,
                        'min_data_in_leaf': 10,
                        'num_leaves': 10,
                        'objective': 'regression'}
    
    loaded_params_g = {'max_depth': [1],
                        'num_leaves': [2],
                        'min_data_in_leaf': [35],
                        'feature_fraction': [0.6],
                        'learning_rate': [0.1],
                        'max_bin': [15],}


# predict and save the result
k_mean = []
k_std = []
g_mean = []
g_std = []

for test_size in test_ratio:
    
    x_train, x_test, y_train_k, y_test_k = train_test_split(x, y_k, test_size=test_size, random_state=random_state)
    #skf = utils.StratifiedKFoldReg(n_splits=5)
    
    reg = lgb.LGBMRegressor(**loaded_params_k)
#     cv = ShuffleSplit(n_splits=30, test_size=test_size, random_state=random_state)
    scores = cross_val_score(reg, x_train, y_train_k.values.ravel(), scoring='neg_mean_squared_error') #, cv=skf)
    
    rmse = np.sqrt(-scores)
    k_mean.append(rmse.mean())
    k_std.append(rmse.std())

    
    x_train, x_test, y_train_g, y_test_g = train_test_split(x, y_g, test_size=test_size, random_state=random_state)
    
    reg = lgb.LGBMRegressor(**loaded_params_g)
    scores = cross_val_score(reg, x_train, y_train_g.values.ravel(), scoring='neg_mean_squared_error') #, cv=skf) 

    rmse = np.sqrt(-scores)
    g_mean.append(rmse.mean())
    g_std.append(rmse.std())       
        
# plot
for to_plot, y_label in zip([k_mean, k_std, g_mean, g_std], 
                   ["k mean", "k std", "g mean", "g std"]):
    utils.save_plot_mean_std(to_plot, y_label, test_ratio, output, name)
  
k_pred = []
k_true = []
g_pred = []
g_true = []
    
for test_size in test_ratio:
    x_train, x_test, y_train_k, y_test_k = train_test_split(x, y_k, test_size=test_size, random_state=random_state)
    reg = lgb.LGBMRegressor(**loaded_params_k)
    reg.fit(x_train, y_train_k)
    
    y_pred = reg.predict(x_test)
    
    yx = np.array(y_test_k).reshape(y_test_k.shape[0], )
    
    r2 = r2_score(yx, y_pred)
    utils.save_plot_r2(y_pred, yx, 'k', test_size, r2, output, name)
    
    if test_size == 0.1:
        k_pred.append(y_pred)
        k_true.append(yx)
        
    x_train, x_test, y_train_g, y_test_g = train_test_split(x, y_g, test_size=test_size, random_state=random_state)
    reg = lgb.LGBMRegressor(**loaded_params_g)
    reg.fit(x_train, y_train_g)
         
    y_pred = reg.predict(x_test)
    
    yx = np.array(y_test_g).reshape(y_test_g.shape[0], )
    
    r2 = r2_score(yx, y_pred)
    utils.save_plot_r2(y_pred, yx, 'g', test_size, r2, output, name)
    
    if test_size == 0.1:
        g_pred.append(y_pred)
        g_true.append(yx)

sns.reset_orig()    

# plot feature importance
for test_size in test_ratio:
    x_train, x_test, y_train_k, y_test_k = train_test_split(x, y_k, test_size=test_size, random_state=random_state)
    reg = lgb.LGBMRegressor(**loaded_params_k)
    reg.fit(x_train, y_train_k)
    
    # feature importance plot
    if test_size == 0.3:
        feature_imp_k = pd.DataFrame({'Value':reg.feature_importances_,'Feature':reg.feature_name_})
        plt.figure(figsize=(60, 40))
        sns.set(font_scale = 3)
        sns.barplot(x="Value", y="Feature", data=feature_imp_k.sort_values(by="Value", ascending=False)[0:49])
        plt.title('LightGBM Features')
        plt.savefig('feature_import-K.png')

    x_train, x_test, y_train_g, y_test_g = train_test_split(x, y_g, test_size=test_size, random_state=random_state)
    reg = lgb.LGBMRegressor(**loaded_params_g)
    reg.fit(x_train, y_train_g)
    
     # feature importance plot
    if test_size == 0.7:
        feature_imp_g = pd.DataFrame({'Value':reg.feature_importances_,'Feature':reg.feature_name_})
        plt.figure(figsize=(40, 20))
        sns.set(font_scale = 1)
        sns.barplot(x="Value", y="Feature", data=feature_imp_g.sort_values(by="Value", ascending=False)[0:49])
        plt.title('LightGBM Features')
        plt.savefig('feature_import-G.png')
        
sorted_feature_imp_k = feature_imp_k.sort_values(by="Value",ascending=False)
sorted_feature_imp_g = feature_imp_g.sort_values(by="Value",ascending=False)

sorted_feature_imp_k.to_excel('sorted_fea_imp_k.xlsx')
sorted_feature_imp_g.to_excel('sorted_fea_imp_g.xlsx')
