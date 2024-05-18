# -*- coding: utf-8 -*-
# Article https://towardsdatascience.com/the-anatomy-of-a-stock-market-downturn-6527e31406f0


# from https://github.com/yiuhyuk/market_decline/blob/master/market_crash.ipynb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wget
import sys
import datetime
#from scipy.optimize import minimize  # not used so commented out - 05/18/2024
#import statsmodels.api as sm  # not used so commented out - 05/18/2024
#from scipy.stats import norm  # not used so commented out - 05/18/2024


def fix_shiller_date(date):
    date=str(date)
    if len(date) == 6:
        date += '0'
    date_ = date.split('.')
    return(date_[0] + '-' + date_[1] +'-01')

# Calculate monthly Total Returns for the S&P500 (includes dividends)
# stock_ret = pd.read_csv('data_csv.csv')  # commented out on 05/18/2024

# get latest data from original source
# http://www.econ.yale.edu/~shiller/data.htm
# Adjusted to remove inflation - https://www.multpl.com/s-p-500-historical-prices/table/by-year
# Note: during college break, data is not updated

dateStr = datetime.datetime.now().strftime('%Y%m%d')
filename = 'shiller_ie_data_' + dateStr + '.xls'
try:
    url = 'http://www.econ.yale.edu/~shiller/data/ie_data.xls'   # from http://www.econ.yale.edu/~shiller/data.htm
    filename = wget.download(url, out=filename, bar=None)
except:
    print('ERROR: not able to retrieve', url)
    sys.exit()


stock_ret = pd.read_excel(filename, sheet_name='Data', skiprows=7)
# only use one column so let's realign name to original article
stock_ret.columns = ['SP500' if c == 'P' else c for c in stock_ret.columns]
stock_ret = stock_ret[~stock_ret.Date.isnull()]  # drop any rows that do not have a date defined

# I want annual data so I keep only January of each year
mth_df = stock_ret.copy()
# Convert date string to datetime format to accomodate multiple date formats in CSV file - 05/18/2024
mth_df.Date = mth_df.Date.apply(lambda d: fix_shiller_date(d))
mth_df.Date = pd.to_datetime(mth_df.Date, format='mixed')

#mth_df['year'] = [int(i.split('-')[0]) for i in mth_df['Date']]  # commented out - 05/18/2024
mth_df['year'] = mth_df.Date.dt.year  # retrieve year from converted date format - 05/18/2024
#mth_df['mth'] = [int(i.split('-')[1]) for i in mth_df['Date']]  # commented out - 05/18/2024
mth_df['mth'] = mth_df.Date.dt.month  # retrieve year from converted date format - 05/18/2024

mth_ret = mth_df['SP500']/mth_df['SP500'].shift(1) - 1
mth_df['mth_return'] = mth_ret #+ (mth_df['Dividend']*(1/12))/mth_df['SP500']

# Focus on post World War II period - THIS IS THE DATAFRAME THAT HOLDS MOST OF THE ANALYSIS DATA
#modern_df = mth_df.loc[948:].copy()   # replaced with code below - 05/18/2024
modern_df = mth_df[mth_df.Date >= '1948-01-01'].copy()  # use to be >= 1950-01-01  -- 05/18/2024
modern_df['cum_return'] = np.cumprod(modern_df['mth_return']+1)

# Calculate the negative runs in the S&P 500 (from previous peak to the next peak)
# I use this to plot the declines graph (the red lined chart below)
neg_run = []
max_so_far = modern_df['cum_return'].iloc[0]                 # this variable stores the max return observed so far
for i, val in enumerate(modern_df['mth_return']):
    if i == 0:
        if val < 0:
            neg_run.append(val)
        else:
            neg_run.append(0)
    else:
        if modern_df['cum_return'].iloc[i] < max_so_far:     # if we have not yet regained the previous all time high
            neg_run.append((1 + neg_run[i-1])*(1 + val) - 1) # return then keep tallying the loss
        else:
            neg_run.append(0)                                # otherwise stop tallying the loss
            max_so_far = modern_df['cum_return'].iloc[i]

modern_df['neg_run'] = neg_run
# Convert date string to datetime format
#modern_df['Date'] = pd.to_datetime(modern_df['Date'])  # commented out because this was done above - 05/18/2024

# Plot the cumulative declines from the previous all time high
fig, ax = plt.subplots(figsize=(12,8))
ax = sns.lineplot(x='Date', y='neg_run', data=modern_df, color='red');
ax.set_xlabel("Date", fontsize=16)
ax.set_ylabel("% Decline from Previous Peak for S&P 500", fontsize=16)

plt.savefig(fname='decline_plot', dpi=150)

# Tag each cumulative decline (neg_run) with a integer label, for example the 5th neg_run would be tagged 5
# This label will be used later on in groupby's to investigate the characteristics of each decline
# neg_run goes from previous all time high to next all time high
label = 1
decline_num = []
first_seen = False
for i in modern_df['neg_run']:
    if i < 0:
        decline_num.append(label)
        first_seen = True
    else:
        decline_num.append(0)
        if first_seen:
            label += 1
            first_seen = False
            
modern_df['decline_num'] = decline_num

# Tag each peak to trough decline (neg_run) with a integer label
# pt_num identifies just the period from the previous all time high to the declines trough (lowest point)
label = 1
pt_num = []
found_min = False
for index, val in enumerate(modern_df['decline_num']):
    min_decline = min(modern_df[modern_df['decline_num']==val]['neg_run'])
    if val > 0:
        if found_min:
            pt_num.append(0)
        else:
            if modern_df.iloc[index]['neg_run'] == min_decline:
                found_min = True
                pt_num.append(val)
            else:
                pt_num.append(val)
    else:
        found_min = False
        pt_num.append(val)
            
modern_df['pt_num'] = pt_num

# Groupby's to check out the durations and maximum loss of each market decline identified
run_len = modern_df[modern_df['decline_num']>0].groupby(by='decline_num').count()['neg_run']
peak_decline = modern_df[modern_df['decline_num']>0].groupby(by='decline_num').min()['neg_run']
peak_trough_dur = modern_df[modern_df['pt_num']>0].groupby(by='pt_num').count()['neg_run']

# plt.plot(run_len.sort_values(ascending=False).reset_index(drop=True));
# Store groupby results in a new dataframe
declines_df = pd.DataFrame()

declines_df['run_len'] = run_len
declines_df['peak_decline'] = peak_decline
declines_df['peak_trough_dur'] = peak_trough_dur

# Bucket declines by magnitude
decline_bin = []
for i in peak_decline:
    if i >= 0.00:
        decline_bin.append(0)
    elif i >= -0.05:
        decline_bin.append(1)
    elif i >= -0.10:
        decline_bin.append(2)
    elif i >= -0.20:
        decline_bin.append(3)
    elif i >= -0.30:
        decline_bin.append(4)
    else:
        decline_bin.append(5)

declines_df['decline_bin'] = decline_bin

# Overall means for decline metrics
print('Overall mean for decline: ', np.mean(declines_df), '\n')

# Count the number of declines in each magnitude bucket
declines_df.groupby(by='decline_bin').count()['run_len']

# Plot the number of declines in each magnitude bucket as a probability
prob_bucket = declines_df.groupby(by='decline_bin').count()['run_len']/sum(declines_df.groupby(by='decline_bin').count()['run_len'])
fig, ax = plt.subplots(figsize=(10,6))
bin_names = ['-5% or Better','-5% to -10%','-10% to -20%','-20% to -30%','-30% or Worse']
sns.barplot(x=prob_bucket, y=bin_names);
ax.set_xlabel("Probability",fontsize=14)
ax.set_ylabel("Market Decline Bin",fontsize=14)
ax.set_xlim(0, 1)

vals = ax.get_xticks()
ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])

plt.tight_layout()
plt.savefig(fname='bar_prob_bins', dpi=150)

# Check out what happens after market has already dropped by 5%
worst_probs = prob_bucket[1:]/sum(prob_bucket[1:])
print('# probability of decline more than 10%', sum(worst_probs[1:]))     # probability of decline more than 10%
print('probability of decline being more than 20%', sum(worst_probs[2:]))     # probability of decline being more than 20%
print('\n')

# Check the mean decline in each bucket
declines_df.groupby(by='decline_bin').mean()['peak_decline']

# Check out the metrics of each bucket and store in a dataframe for plotting
duration_df = declines_df.groupby(by='decline_bin').mean()[['peak_trough_dur','run_len']]
duration_df.reset_index(inplace=True)
duration_df['recover_dur'] = duration_df['run_len'] - duration_df['peak_trough_dur']
#duration_df

# Plot the metrics I am interestred in
fig, ax = plt.subplots(figsize=(10,6))
bin_names = ['-5% or Better','-5% to -10%','-10% to -20%','-20% to -30%','-30% or Worse']
sns.barplot(x=bin_names, y=duration_df['recover_dur'])
ax.set_xlabel("Market Decline Bin",fontsize=14)
ax.set_ylabel("Duration in Months",fontsize=14)

plt.tight_layout()
plt.savefig(fname='bar_plot_dur', dpi=150)

# Number and percentage of negative months
print('number and percentage of negative months')
print(len([i for i in modern_df['mth_return'] if i<0]))
print(len([i for i in modern_df['mth_return'] if i<0])/modern_df.shape[0])
print('\n')
# Mean length of decline
print('Mean length of decline:', np.mean(declines_df['peak_trough_dur']))

