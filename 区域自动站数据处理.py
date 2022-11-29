import numpy as np
import csv  
import pandas as pd
import matplotlib.pyplot as plt

data_csv = pd.read_csv('.\\jupyter_data\\dexing\\dms.csv', header=None)
data_csv.columns = ['id','time','t_ave','t_max','t_min','wind_speed_2min','rain','wind_speed_max']

df = data_csv.drop([0])

df['time'] = pd.to_datetime(df['time']) 

df_mon6 = df.loc[(df['time'].dt.month == 6)]
df_mon7 = df.loc[(df['time'].dt.month == 7)]
df_mon8 = df.loc[(df['time'].dt.month == 8)]

#风
w_mean = np.array(df.wind_speed_2min,dtype=np.float)
w_mean_mon6 = np.array(df_mon6.wind_speed_2min,dtype=np.float)
w_mean_mon7 = np.array(df_mon7.wind_speed_2min,dtype=np.float)
w_mean_mon8 = np.array(df_mon8.wind_speed_2min,dtype=np.float)

w_mean[w_mean==999999.0]=np.nan
w_mean_mon6[w_mean_mon6==999999.0]=np.nan
w_mean_mon7[w_mean_mon7==999999.0]=np.nan
w_mean_mon8[w_mean_mon8==999999.0]=np.nan

wmean = np.nanmean(w_mean)
wmean_mon6 = np.nanmean(w_mean_mon6)
wmean_mon7 = np.nanmean(w_mean_mon7)
wmean_mon8 = np.nanmean(w_mean_mon8)

wmean_06 = np.zeros(30)
for day in np.arange(30):
    day_data = df_mon6.loc[(df_mon6['time'].dt.day == day+1)]
    tday_mean = np.array(day_data.wind_speed_2min,dtype=np.float)
    tday_mean[tday_mean==999999.0]=np.nan
    wmean_06[day] = np.nanmean(tday_mean)
    
wmean_07 = np.zeros(31)
for day in np.arange(31):
    day_data = df_mon7.loc[(df_mon7['time'].dt.day == day+1)]
    tday_mean = np.array(day_data.wind_speed_2min,dtype=np.float)
    tday_mean[tday_mean==999999.0]=np.nan
    wmean_07[day] = np.nanmean(tday_mean)
    
wmean_08 = np.zeros(31)
for day in np.arange(31):
    day_data = df_mon8.loc[(df_mon8['time'].dt.day == day+1)]
    tday_mean = np.array(day_data.wind_speed_2min,dtype=np.float)
    tday_mean[tday_mean==999999.0]=np.nan
    wmean_08[day] = np.nanmean(tday_mean)
    
wmean_day = np.concatenate([wmean_06,wmean_07,wmean_08])

date = np.arange(len(wmean_day))
labels =['6月1日','6月8日','6月15日','6月22日','6月29日','7月6日','7月13日','7月20日','7月27日',
        '8月3日','8月10日','8月17日','8月24日','8月31日']
        
fig, ax = plt.subplots()

l1 = ax.plot(date,wmean_day)
plt.rcParams['font.sans-serif']=['SimHei']
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width , box.height])
#ax.legend(loc='center left', bbox_to_anchor=(-0.01,0.35),ncol=4)

ax.set_ylabel('风速（m/s）')
plt.xticks(date[::7],labels,rotation=40)

plt.savefig('.\\jupyter_data\\dexing\\风速日平均.png')
