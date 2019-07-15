#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from math import log
import re


class qcut_stable():
    def __init__(self,data,features,target,bin_pct):
        #method in( 'iv','et') et stands for entropy
        self.data=data.query('oot2<=1')
        self.data_oot=data
        self.features=features
        self.target=self.data[target]
        self.bin_pct=bin_pct
        self.set_obs=self.data.__len__()
        self.tbbinresult={}
        self.trenddf={}

    def features_std(self,exile,classvar=[]):
        oldfeatures=self.data.dtypes.astype('str')
        tabvarlist=self.data.columns.tolist()
        classvar=[f for f in classvar if f in tabvarlist]
        if classvar.__len__()==0:
            classvar=[]
        newfield={}
        for x,y in oldfeatures.iteritems():
            if x not in exile and x not in classvar and str(y)!='object':
                newfield[x]=str(y)
        self.features=newfield

    def calcu_entropy(self,df):
        sub_tot =df.sum(axis=1)
        tot=sub_tot.sum()
        sub_tot_rt=sub_tot*1.0/tot
        df_rt=df.T.apply(lambda x:x*1.0/x.sum()).T
        entropy=(df_rt.apply(lambda x:-x*np.log(x)).sum(axis=1)/np.log(2)*sub_tot_rt).sum()
        return entropy

    def calcu_iv(self,df):
        df_rt = df.iloc[:, 0:2].applymap(lambda x: 1 if x == 0 else x).apply(lambda x: x / x.sum())
        x = np.array(df_rt.T)
        iv = ((x[0] - x[1]) * np.log(x[0] / x[1])).sum()
        return iv

    def var_num_basebin(self,vardata,bin_pct=0.05):
        bin_sum=vardata.round(3).value_counts().sort_index()
        num_var_rt=bin_sum*1.0/bin_sum.sum()
        if num_var_rt.iloc[0]>=bin_pct:
            num_var_rt.iloc[0]=0
        var_base=np.ceil(num_var_rt.cumsum()/bin_pct)
        bins=[vardata.min()-0.5]
        loc=0
        for bin_id,idx in var_base.value_counts().sort_index().cumsum().iteritems():
            bins.append(var_base.index.tolist()[idx-1])
            loc+=1
        return bins

    def stable_cacule(self):
        tmp_flag=self.target.name
        tmp_varindex=[]
        tmp_varbin=[]
        var_list=list(self.features.keys())
        var_list = self.features
        #         for var in self.features.keys():
        for var in var_list:
        # for var in self.features.keys():
            # print(var)
            tmp_oot_dt=self.data_oot[[var,tmp_flag,'oot2']].fillna(-999)
            if self.data[var].fillna(-999).value_counts().__len__()==1:
                var_list.remove(var)
                continue
            bins=self.var_num_basebin(self.data[var].fillna(-999),self.bin_pct)
            bins=[tmp_oot_dt[var].min()-0.5]+bins[1:-1] +[tmp_oot_dt[var].max()]
            #print(bins)
            tmp_newvar=pd.cut(tmp_oot_dt[var],bins=bins)
            trenddf=tmp_oot_dt.groupby([tmp_newvar,'oot2'])[tmp_flag].mean().unstack(level=1)
            #corr
            corr=np.prod(trenddf.corr(method='spearman').iloc[0:1,1:].values)
            corr_p = np.prod(trenddf.corr(method='pearson').iloc[0:1, 1:].values)
            trenddf['varname']=var
            trenddf.index.name='binrange'
            trenddf.reset_index(inplace=True)
            trenddf['binrange']=trenddf['binrange'].astype('str')
            #iv
            df=pd.crosstab(pd.cut(self.data[var].fillna(-999),bins=bins),self.target)
            iv=self.calcu_iv(df=df)
            et=self.calcu_entropy(df=df)
            df_rt=df.apply(lambda x: x/x.sum())
            df['row_rt']=df.sum(axis=1)/df.values.sum()
            df=df.join(df_rt,lsuffix='cnt',rsuffix='rt')
            df['varname']=var
            df.index.name='binrange'
            df.reset_index(inplace=True)
            df['binrange']=df['binrange'].astype('str')
            tmp_varbin.append(df)
            #psi
            cross_cnt=tmp_oot_dt.groupby([tmp_newvar,'oot2']).size().unstack(level=1)
            stb_et=self.calcu_entropy(df=cross_cnt)
            cross_cnt=cross_cnt.applymap(lambda x: 1 if x==0 else x)
            psi=[]
            for x in range(1,cross_cnt.shape[1]):
                psi.append(self.calcu_iv(df=cross_cnt.iloc[:,[0,x]]))
            psi.append(max(psi))
            #
            null_pct=self.data[var].isnull().sum()/self.set_obs
            bin_max_pct=cross_cnt.apply(lambda x: x/x.sum()).max().values[0]
            tmp_varindex.append([iv,et,corr,corr_p,bin_max_pct,bins.__len__()-1,null_pct,stb_et]+psi)

            self.trenddf[var]=trenddf
        self.tbbinresult['varbin']=pd.concat(tmp_varbin,keys=var_list,names=['varname','bin'])
        self.tbbinresult['varindex']=pd.DataFrame(tmp_varindex,index=var_list,columns=['e_value','et', 'corr','corr_p','bin_max_pct','bins','null_pct','stb_et']+['psi'+str(x) for x in list(range(0,psi.__len__()))])
        self.tbbinresult['varindex'].index.name='varname'
        self.tbbinresult['badtrend']=pd.concat(list(self.trenddf.values()))

    def var_desc(self):
        catvar=[]
        numvar=[]
        for var,type in self.features.items():
            if type=='object':
                catvar.append(var)
            else:
                numvar.append(var)
        if numvar.__len__()>0:
            numvar_desc= self.data[numvar].describe().T
            numvar_desc['null_cnt']=self.set_obs- numvar_desc['count']
            self.tbbinresult['numvar_desc']=numvar_desc
        if catvar.__len__()>0:
            catvar_desc =self.data[catvar].describe().T
            catvar_desc['null_cnt']=self.set_obs- catvar_desc['count']
            self.tbbinresult['catvar_desc']=catvar_desc

    def save_result(self,outpath):
        writer=pd.ExcelWriter(outpath)
        for x,y in self.tbbinresult.items():
            #print(x)
            y.to_excel(writer,x)
        writer.save()
