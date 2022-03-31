
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.base import clone
from sklearn.metrics import r2_score 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score 
import random
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance


def Spearman_correl(x,y):

    x=np.array(x)
    y=np.array(y)
    x=np.array(pd.Series(x).rank())
    y=np.array(pd.Series(y).rank())
    xu=np.mean(x)
    yu=np.mean(y)
    Num= np.sum((x-xu)*(y-yu))
    Den= np.sqrt(np.sum((x-xu)**2)*np.sum((y-yu)**2))
    corr=Num/Den
      
    return "{:.6f}".format(corr)


def df_corr(df):
    df=df.dropna()
    variables=df.columns
    correl=[]
    for x in variables:
        for y in variables:
            df1 = df.dropna(subset=[x,y]).copy()
            correl.append((x,y,Spearman_correl(df1[x],df1[y])))
    new_df=pd.DataFrame(correl,columns=['1','2','3'])
    new_df=new_df.pivot_table(index='1', columns='2', values='3',aggfunc=np.sum)
    return new_df


def basic_PCA(X,k):
    cols=X.columns
    Xu=X - np.mean(X)
    total_var = []
    covmtrx = np.cov(Xu , rowvar = False)
    evalues , evectors = np.linalg.eigh(covmtrx)
    sort_idx = np.argsort(evalues)[::-1] 
    eval_sort = evalues[sort_idx]
    evec_sort = evectors[:,sort_idx]
    total_var= [(cols[i],evalues[i] / np.sum(evalues)) for i in range(len(evalues))]
    ncomps = k
    changed_evec = evec_sort[:,0:ncomps]
    Xred=np.dot(changed_evec.transpose(),Xu.transpose()).transpose()
    return Xred,sorted(total_var, key=lambda x: x[1],reverse=True)

def mrmr(X,y,num_select):
    F = pd.Series(f_regression(X, y)[0], index = X.columns)
    corr = X.corr(method="spearman").abs().clip(.00001) 

    sel = []
    ns = list(X.columns)
    for i in range(num_select):
        if i > 0:
            ls = sel[-1]
            corr.loc[ns, ls] = X[ns].corrwith(X[ls]).abs().clip(.00001)

        val = F.loc[ns] / corr.loc[ns, sel].mean(axis = 1).fillna(.00001)
        best_score = val.index[val.argmax()]
        sel.append(best_score)
        ns.remove(best_score)
    return sel


def dropcol_importances(model,metric,X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    baseline = metric(y_valid, model.predict(X_valid))
    imp = []
    X_train=pd.DataFrame(X_train)
    X_valid=pd.DataFrame(X_valid)
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1)
        X_valid_ = X_valid.drop(col, axis=1)
        model_ = clone(model)
        model_.fit(X_train_, y_train)
        m = metric(y_valid, model_.predict(X_valid_))
        imp.append((col,baseline - m))
    return imp


def permutation_importances(model, metric,X_valid, y_valid):
    baseline = metric(y_valid, model.predict(X_valid))
    imp = []
    X_valid=pd.DataFrame(X_valid)
    for col in X_valid.columns:
        save = X_valid[col].copy()
        X_valid[col] = np.random.permutation(X_valid[col])
        m = metric(y_valid, model.predict(X_valid))
        X_valid[col] = save
        imp.append((col,baseline - m))
    return imp


def rsquared(actual,predict):
    corr_matrix = np.corrcoef(actual, predict)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    return R_sq
 

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def automatic_feature_select(model,metric,method_imp,X_train, y_train, X_valid, y_valid):
    X_train=np.array(X_train)
    columns_dataframe=pd.DataFrame(X_train).columns
    model.fit(X_train, y_train)
    baseline = metric(y_valid, model.predict(X_valid))
    scoring = 'neg_mean_absolute_percentage_error'
    if method_imp=='permutation':
        r_multi = permutation_importance(model, X_valid, y_valid, n_repeats=30, random_state=0, scoring=scoring)['importances_mean']
    if method_imp=='drop':
        r_multi= dropcol_importances(model, metric,X_train, y_train,X_valid, y_valid)[1]  
    r_multi=np.vstack((np.arange(X_train.shape[1]), r_multi)).T
    r_multisort=r_multi.copy()
    r_multisort=r_multi[r_multi[:, 1].argsort()]
    rem_X_train=X_train.copy()
    rem_X_valid=X_valid.copy()
    dropped_indexes=[]
    for k in range(0,len(columns_dataframe)):
        print(rem_X_train.shape)
        rem_X_train=np.delete(np.array(rem_X_train),0 , 1)
        rem_X_valid=np.delete(np.array(rem_X_valid), 0, 1)
        model.fit(rem_X_train, y_train)
        new_val_metric = metric(y_valid, model.predict(rem_X_valid))
        if (abs(baseline-new_val_metric)*1.00/baseline)>0.05:
            break
        else:
            dropped_indexes.append(r_multisort[0][0])
            r_multi=np.delete(r_multi,np.where(r_multi[:,0]==r_multisort[0][0]),0)
            if method_imp=='permutation':
                r_multi_new= permutation_importance(model, rem_X_valid, y_valid, n_repeats=30, random_state=0, scoring=scoring)['importances_mean']
                r_multi_new=np.vstack((r_multi[:,0], r_multi_new)).T
                
            if method_imp=='drop':
                r_multi_new= dropcol_importances(model, metric,X_train, y_train,X_valid, y_valid)[1]
                r_multi_new=np.vstack((r_multi[:,0], r_multi_new)).T
                
            r_multisort=r_multi_new.copy()
            r_multisort=r_multisort[r_multisort[:, 1].argsort()]
    return r_multisort,dropped_indexes
        


def var_fi(data, times):
    varimps=[]
    for i in range(0,times):
        print(i)
        bootsp_data_X=pd.DataFrame(np.hstack((data.data,data.target.reshape(-1,1)))).sample(frac=0.5, replace=False)
        X_train, X_val, y_train, y_val = train_test_split(bootsp_data_X.iloc[:,:-1], bootsp_data_X.iloc[:,-1])
        imp=pd.DataFrame(permutation_importances(model, rsquared,X_val, y_val),columns=['Variable','Importance'])
        #dropcol_importances(model, rsquared,X_train, y_train,X_val, y_val)
        sc = MinMaxScaler(feature_range=(0, 1))
        data_norm = sc.fit_transform(np.array(imp)[:,1].reshape(-1,1))
        imp=np.hstack((np.array(imp)[:,0].reshape(-1,1),data_norm))
        varimps.append(imp)
    cols=bootsp_data_X.shape[1]-1 
    var_df=pd.DataFrame(np.array(varimps).reshape(cols*times,2))
    var_df.columns=['col','val_vi']
    result = var_df.groupby(['col'], as_index=False).agg({'val_vi':['mean','std','count']})
    result['relative_mean']=(result['val_vi']['mean']*1.00)/sum(result['val_vi']['mean'])
    result['max_CI']=(result['relative_mean']+(2*result['val_vi']['std']/np.sqrt(result['val_vi']['count'])))/2
    result['min_CI']=(result['relative_mean']-(2*result['val_vi']['std']/np.sqrt(result['val_vi']['count'])))/2
    return var_df,result
    

def pvalue_fi(data,runs):
    null_imp_df = pd.DataFrame()
    nb_runs = runs
    start = time.time()
    data_X=pd.DataFrame(np.hstack((data.data,data.target.reshape(-1,1))))
    add_counts=np.zeros((data_X.shape[1])-1)
    X_train, X_val, y_train, y_val = train_test_split(data_X.iloc[:,:-1], data_X.iloc[:,-1])
    baseline_imp_df = dropcol_importances(model, rsquared,X_train, y_train,X_val, y_val)
    for i in range(nb_runs):
        y_train=y_train.sample(frac=1, replace=False)
        upd_imp_df = dropcol_importances(model, rsquared,X_train, y_train,X_val, y_val)
        conditn=np.where(((np.array(upd_imp_df)[:,1]-np.array(baseline_imp_df)[:,1]))>=0,1,0)
        gain_loss=pd.DataFrame(((np.array(upd_imp_df)[:,1])))
        add_counts=add_counts+conditn
        intm_df=pd.concat([pd.DataFrame(np.arange(X_train.shape[1])), pd.DataFrame(conditn),gain_loss], axis=1)
        intm_df.columns=['col','count','gain_loss']
        intm_df['run'] = i + 1 
        null_imp_df = pd.concat([null_imp_df, intm_df], axis=0)
        #if i%10==0:
            #print(i)

    add_counts=(add_counts*1.00)/(nb_runs)
    thresh=np.where(np.array(add_counts)>=0.05,0,1)
    baseline_imp_df=pd.DataFrame(baseline_imp_df,columns=['col','gain_loss'])
    return add_counts,thresh,null_imp_df,baseline_imp_df



def plot_selc(X,k):
    results = []   
    for i in range(1,k+1):
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = boston.target
        filtered_data=X.iloc[:,selected[:,0]]
        X=filtered_data[filtered_data.columns[:i]]
        models = []
        #models.append(('Ridge', Ridge(1e-5)))
        models.append(('OLS', LinearRegression()))
        models.append(('RF', RandomForestRegressor()))
        models.append(('XGB', xgb.XGBRegressor()))
        names = []
        scoring = 'neg_mean_absolute_error'
        for name, model in models:
            kfold = ms.KFold(n_splits=10)
            cv_results = ms.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
            results.append((name,i,cv_results.mean()))
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

    return results
