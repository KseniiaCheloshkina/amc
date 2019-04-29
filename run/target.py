import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt


class CreateTarget():
    
    """
    Класс для разметки фаз ходьбы.
    Основная функция - evaluate(self, d, plot=False), остальные вспомогательные и используются основной функцией.
    Задание параметров происходит при инициализации __init__(self, win=10, der_thr=0.1, tibia_delta=0.1):
        - win - ширина окна для усреднения при подсчете производной
        - der_thr - максимально допустимое значение колебания производной около нуля в абсолютном измерении 
        - tibia_delta - максимально допустимое значение превышения координаты высоты tibia одной ноги над другой, 
        которое считается незначительным и не приводит к смене single support 
        
    Функции:
    - get_derivative(self, d, col, win, thr) - считает сглаженную производную
    - smooth_phases(self, l_ones, min_v, max_v) - заполняет пустоты, если размеры пустоты находятся в заданных границах 
    - map_one_leg(self, d, base_foot, plot=False) - размечает фазы 'toe off' / 'heel strike' для заданной ноги base_foot
    - evaluate(self, d, plot=False) - осуществляет полную разметку фаз: по отдельности для каждой ноги (phases_r, phases_l) и 
    в целом тип опоры (support)
    
    """
    
    def __init__(self, win=10, der_thr=0.1, tibia_delta=0.1, smooth_max_v=20):
        
        self.win = win
        self.der_thr = der_thr
        self.tibia_delta = tibia_delta
        self.smooth_max_v = smooth_max_v
        
        
    def get_derivative(self, d, col, win, thr):

        df = pd.DataFrame(d[col])
        d_final = d.copy(deep=True)

        for i in range(1, win + 1):
            df['diff_' + str(i)] = df[col].diff(i)
        cols = df.columns.tolist()
        cols.remove(col)
        df['med_diff'] = df[cols].median(axis=1)
        df['monoton'] = 0
        df.loc[df['med_diff'] > thr, 'monoton'] = 1
        df.loc[df['med_diff'] < -thr, 'monoton'] = -1

        # smooth
        df['monoton_shift_1'] = df['monoton'].shift(1)
        df['monoton_shift_-1'] = df['monoton'].shift(-1)
        df['monoton_corr'] = df['monoton']
        df.loc[df['monoton_shift_1'] == df['monoton_shift_-1'], 'monoton_corr'] = \
            df.loc[df['monoton_shift_1'] == df['monoton_shift_-1'], 'monoton_shift_-1']

        d_final[col + '_monot'] = df['monoton_corr']

        return d_final

    
    def smooth_phases(self, l_ones, min_v, max_v):
        
        # fill in holes
        all_frames = []
        prev = [l_ones[0]]
        l_diff = [l_ones[i] - l_ones[i - 1] for i in range(1, len(l_ones))]

        for i in range(len(l_diff)):

            el = l_diff[i]
            if el > min_v and el < max_v:
                new_vals = list(np.arange(l_ones[i], l_ones[i + 1] + 1)) 
                prev.extend(new_vals)
            elif el == min_v:
                prev.append(l_ones[i + 1])
            else:
                all_frames.extend(prev)
                prev = [l_ones[i + 1]]        
        all_frames.extend(prev)
        
        # correct ouliers
        all_int = []
        prev = all_frames[0]
        cur_int = [prev]

        for i in range(1, len(all_frames)):
            el = all_frames[i]
            if el - prev == 1:
                cur_int.append(el)
            else:
                all_int.append(cur_int)
                cur_int = [el]
            prev = el
        all_int.append(cur_int)

        final_frames = []

        for interval in all_int:
            if len(interval) > 5:
                final_frames.extend(interval)        
        
        return final_frames
    
    
    def map_one_leg(self, d, base_foot, plot=False):
        
        d_final = d.copy(deep=True)
        
        if base_foot == 'l':
            base_foot_col = 'lfoot_coord_1'
            base_toes_col = 'ltoes_coord_1'
            base_tibia_col = 'ltibia_coord_1'
        else:
            base_foot_col = 'rfoot_coord_1'
            base_toes_col = 'rtoes_coord_1'  
            base_tibia_col = 'rtibia_coord_1'          
            
        d_final = self.get_derivative(d_final, col=base_foot_col, win=self.win, thr=self.der_thr)
        d_final = self.get_derivative(d_final, col=base_toes_col, win=self.win, thr=self.der_thr)
        d_final = self.get_derivative(d_final, col=base_tibia_col, win=self.win, thr=self.der_thr)
        
        # heel strike
        d_final['heel_strike_' + base_foot] = 0
        d_final.loc[
            (d_final[base_toes_col] > d_final[base_foot_col]) &
            (d_final[base_toes_col +  "_monot"] == -1) & 
            (d_final[base_foot_col +  "_monot"] == -1), 
        'heel_strike_' + base_foot] = 1

        # toe off
        d_final['toe_off_' + base_foot] = 0
        d_final.loc[
            (d_final[base_tibia_col] > d_final[base_foot_col]) &
            (d_final[base_toes_col +  "_monot"] >= 0) & 
            (d_final[base_foot_col +  "_monot"] >= 0) &
            (d_final[base_tibia_col +  "_monot"] == 1) , 
        'toe_off_' + base_foot] = 1
        
        # smooth
        for col in ['toe_off_' + base_foot, 'heel_strike_' + base_foot]:            
            l_ones = d_final[d_final[col] == 1].frame.tolist()
            fr_all = self.smooth_phases(l_ones, min_v=1, max_v=self.smooth_max_v)
            d_final[col] = 0
            d_final.loc[d_final['frame'].isin(fr_all), col] = 1
        
        d_final['phases_' + base_foot] = None
        d_final.loc[d_final['toe_off_' + base_foot] == 1, 'phases_' + base_foot] = 'toe_off'
        d_final.loc[d_final['heel_strike_' + base_foot] == 1, 'phases_' + base_foot] = 'heel_strike'
        
        
        if plot:
            plt.figure(figsize=(15, 9))
            plt.plot(d_final['frame'], d_final[base_foot_col]/0.45*2.54, label='foot')
            plt.plot(d_final['frame'], d_final[base_foot_col +  "_monot"], label='foot monot')
            plt.plot(d_final['frame'], d_final[base_toes_col]/0.45*2.54, label='toes')
            plt.plot(d_final['frame'], d_final[base_toes_col +  "_monot"] - 5, label='toes monot')
            plt.plot(d_final['frame'], d_final[base_tibia_col]/0.45*2.54, label='tibia')
            plt.plot(d_final['frame'], d_final[base_tibia_col +  "_monot"] - 10, label='tibia monot')
            plt.plot(d_final['frame'], d_final['heel_strike_' + base_foot] * (-15), label='heel_strike')
            plt.plot(d_final['frame'], d_final['toe_off_' + base_foot] * (-20), label='toe_off')
            plt.title('Mapping for {} leg'.format(base_foot))
            plt.legend(loc=2)
            plt.show()
                
        return d_final
    
    
    def evaluate(self, d, plot=False, savefig=None):
        
        # get heel strike | toe off labels for each leg
        base_foot = 'l'
        d_l = self.map_one_leg(d, base_foot=base_foot, plot=plot)
        d_l = d_l[['heel_strike_' + base_foot, 'toe_off_' + base_foot, 'phases_' + base_foot, 'frame']]
        d = pd.merge(d, d_l, left_on='frame', right_on='frame')
        
        base_foot = 'r'
        d_r = self.map_one_leg(d, base_foot=base_foot, plot=plot)
        d_r = d_r[['heel_strike_' + base_foot, 'toe_off_' + base_foot, 'phases_' + base_foot, 'frame']]
        d = pd.merge(d, d_r, left_on='frame', right_on='frame')  
        
        
        # get single support
        d['support'] = None
        d.loc[d['ltibia_coord_1'] > d['rtibia_coord_1'], 'support'] = 'rss'
        d.loc[d['ltibia_coord_1'] < d['rtibia_coord_1'], 'support'] = 'lss'
        # correct single support
        d.loc[(d['rtibia_coord_1'] - d['ltibia_coord_1'] < self.tibia_delta) &
              (d['rtibia_coord_1'] - d['ltibia_coord_1'] > 0), 'support'] = 'rss'
        d.loc[(d['ltibia_coord_1'] - d['rtibia_coord_1'] < self.tibia_delta) &
              (d['ltibia_coord_1'] - d['rtibia_coord_1'] > 0), 'support'] = 'lss'
        d.loc[~d['phases_r'].isnull(), 'support'] = 'ds'
        d.loc[~d['phases_l'].isnull(), 'support'] = 'ds'
        
        # smooth
        d['shift_support_1'] = d['support'].shift(1)
        d['shift_support_-1'] = d['support'].shift(-1)
        d['support_corr'] = d['support']            
        d.loc[(d['shift_support_-1'] == 'ds') & (d['shift_support_1'] != d['support']), 'support_corr'] = 'ds'
        d.loc[(d['shift_support_1'] == 'ds') & (d['shift_support_-1'] != d['support']), 'support_corr'] = 'ds'

        prev_support = d.loc[0, 'support']
        for idx, row in d.iterrows():
            if (row['support'] != 'ds') & (row['support'] != prev_support) & (prev_support != 'ds'):
                d.loc[idx, 'support_corr'] = prev_support
            else:
                prev_support = row['support']
        
        d['support'] = d['support_corr']         
        d.drop(['shift_support_1', 'shift_support_-1', 'support_corr'], axis=1, inplace=True)
    
        # for drawing
        d.loc[d['support'] == 'ds', 'support_cat'] = 0
        d.loc[d['support'] == 'lss', 'support_cat'] = 1
        d.loc[d['support'] == 'rss', 'support_cat'] = -1
      
        
        plt.figure(figsize=(15, 6))
        plt.plot(d['frame'], d['ltoes_coord_1']/0.45*2.54, label='toes l')
        plt.plot(d['frame'], d['ltibia_coord_1']/0.45*2.54, label='tibia l')
        plt.plot(d['frame'], d['rtoes_coord_1']/0.45*2.54, label='toes r')
        plt.plot(d['frame'], d['rtibia_coord_1']/0.45*2.54, label='tibia r')
        plt.plot(d['frame'], d['support_cat'] , label='support')
        plt.title('Support: 0 - double support, 1 - left leg single support, -1 - right leg single support')
        plt.legend(loc=2);
        
        if plot:
            plt.show()
        
        if savefig is not None:
            plt.savefig(savefig)
            plt.close()

        return d
    