import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif
import copy

class CorrelatedRemoval():
    
    def __init__(self, X, y, method_xy, corr_thr=0.8):
        self.X = X
        self.y = y
        self.method_xy = method_xy
        self.corr_thr = corr_thr
        
    def get_xy_corr(self):
        
        res = pd.DataFrame()
        
        for col in self.X.columns:

            if self.method_xy == 'mutual_info_classif':
                xy_corr = abs(mutual_info_classif(X=self.X[col].values.reshape(-1, 1), y=self.y)[0])
                
            if self.method_xy == 'f_classif':
                xy_corr, pval = f_classif(X=self.X[col].values.reshape(-1, 1), y=self.y)
                xy_corr = xy_corr[0]
                
            xy_corr_df = pd.DataFrame([{'feature': col, 'xy_corr': xy_corr}])
            res = pd.concat([res, xy_corr_df])
            
        self.xy_corr = res
                
    def evaluate(self):
        
        print('Calculate features correlation ... ')
        # features correlation
        df_cor = self.X.corr(method='spearman')
        df_corr_unstacked = df_cor.unstack().reset_index()
        df_corr_unstacked = df_corr_unstacked.loc[df_corr_unstacked['level_0'] != df_corr_unstacked['level_1'], ]
        df_corr_unstacked[0] = abs(df_corr_unstacked[0])
        df_corr_unstacked = df_corr_unstacked[df_corr_unstacked[0] > self.corr_thr]
        
        print('Calculate correlation with target ... ')
        # correlation with target
        self.get_xy_corr()
        
        print('Sequentially exclude correlated variables ... ')
        # merge in one dataset
        xy_corr = self.xy_corr
        df_all = pd.merge(df_corr_unstacked, xy_corr, left_on='level_0', right_on='feature', how="left")
        df_all.drop(['feature'], axis=1, inplace=True)
        df_all.columns = ['level_0', 'level_1', 0, 'level_0_xy_corr']
        df_all = pd.merge(df_all, xy_corr, left_on='level_1', right_on='feature', how="left")
        df_all.drop(['feature'], axis=1, inplace=True)
        df_all.columns = ['level_0', 'level_1', 0, 'level_0_xy_corr', 'level_1_xy_corr']
        df_all.sort_values(by=0, ascending=False, inplace=True)
        df_all.reset_index(drop=True, inplace=True)        
        
        
        # sequentially remove correlated
        corr_feat = df_all['level_0'].unique().tolist()

        all_corr_feats = []
        corr_replace_d = {}

        df_init = df_all.copy(deep=True)

        for current_feat in corr_feat:

            if current_feat in df_all['level_0'].unique():
                current_data = df_all[df_all['level_0'] == current_feat]
                p1 = current_data['level_0_xy_corr'].unique()[0]
                idxmax = current_data.agg({'level_1_xy_corr': 'idxmax'})['level_1_xy_corr']
                feat2_best = current_data.loc[idxmax, 'level_1']
                p2 = current_data.loc[idxmax, 'level_1_xy_corr']

                if p1 < p2:
                    cor_list  = current_data['level_1'].unique().tolist()
                    cor_list.remove(feat2_best)
                    cor_list.append(current_feat)
                    corr_replace_d[feat2_best] = cor_list
                else:
                    cor_list  = current_data['level_1'].unique().tolist()
                    corr_replace_d[current_feat] = cor_list

                df_all = df_all[~df_all['level_0'].isin(cor_list)]
                df_all = df_all[~df_all['level_1'].isin(cor_list)]
                all_corr_feats.extend(cor_list)  
                
        self.corr_replace_d = corr_replace_d
        self.all_corr_feats = all_corr_feats
    