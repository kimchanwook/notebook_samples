#%%====================================================== Import and Config
''''''
#%% Imports
import sys
import csv

import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#Plotting Config
mpl.style.use("default")
mpl.rcParams["axes.facecolor"] = "#EAEAF2"
mpl.rcParams['figure.dpi']     = 100
mpl.rcParams['savefig.dpi']    = 100

#Path defined
sys.path.append("./data/")

#%%===================================================================== 1. Data Handling
''''''
#%%====================================================== Data
''''''
#%% Data import

#Stock data
google = pd.read_csv('./data/stock_time_series_data/GOOGL_2006-01-01_to_2018-01-01.csv', index_col="Date",
                     parse_dates=["Date"])

#Weather humidity data
humidity = pd.read_csv("./data/weather_time_series_data/humidity.csv", index_col="datetime",
                       parse_dates=["datetime"])

#%%====================================================== Data Cleansing
''''''
#%% Data Cleansing

#Humidity data
#    -"ffill" parameter which propagates last valid observation to fill gaps.
#    -"bfill" to propogate next valid observation to fill gaps
#        -You can call "ffill" first and then "bfill" next to make sure non-NAN
humidity = humidity.fillna(method="ffill")

#%%====================================================== Data Visualize
''''''
#%% Humidity plot
humidity_plot = plt.figure(figsize=(10, 6))

#Plot
plt.plot(humidity["Kansas City"].asfreq("M").index, humidity["Kansas City"].asfreq("M"), lw=5)

#Plot stats
mean_humidity = np.mean(humidity["Kansas City"])
std_humidity  = np.std(humidity["Kansas City"])
plt.axhline(mean_humidity, linestyle="--", color="black", alpha=1, linewidth=1.5)
plt.fill_between(humidity["Kansas City"].asfreq("M").index, mean_humidity+std_humidity, mean_humidity-std_humidity, alpha=0.2)

#Text about stats
plt.text(0.8, 0.15, s="$\mu$-$\sigma$ {:>10.2f}\n$\mu$+$\sigma$ {:>8.2f}".format(mean_humidity - std_humidity, mean_humidity + std_humidity),
         fontsize=15, transform=plt.gca().transAxes, bbox=dict(facecolor="none", pad=5.0))

#Grids
plt.gca().xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(6))
plt.gca().yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
plt.grid(visible=True, which="major", color="gray", linewidth=.5, linestyle="--")
plt.grid(visible=True, which="minor", color="gray", linewidth=0.25, linestyle="--")

#Labels and lim
plt.xlim()
plt.ylim()
plt.xlabel("Measurement date", fontsize=15)
plt.ylabel("Relative Humidity [%]", fontsize=15)
plt.title("Kansas City Humidity in Monthly Frequency")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
humidity_plot.savefig("./plots/{:s}".format("humidity_plot"))

#%% Google Stock plot
google_stock_plot, subplot = plt.subplots(5, 1, sharex=True, figsize=(15, 15))

#Plot
subplot[0].plot(google["2008":"2010"]["Open"], label="Open")
subplot[1].plot(google["2008":"2010"]["High"], label="High")
subplot[2].plot(google["2008":"2010"]["Low"], label="Low")
subplot[3].plot(google["2008":"2010"]["Close"], label="Close")
subplot[4].plot(google["2008":"2010"]["Volume"], label="Volume")

#Horizontal lines + fills
for i in range(len(subplot)-1):
    column_name = google.columns[i]
    mean = np.mean(google["2008":"2010"][column_name])
    std  = np.std(google["2008":"2010"][column_name])

    #Plot
    subplot[i].axhline(mean, linestyle="--", color="black", alpha=1, linewidth=1.5)
    subplot[i].fill_between(google["2008":"2010"].index, mean+std, mean-std, alpha=0.2)

    #Text: Stats
    subplot[i].text(0.85, 0.15, s="$\mu$-$\sigma$ {:>10.2f}\n$\mu$+$\sigma$ {:>9.2f}".format(mean-std, mean+std),
                    fontsize=15, transform=subplot[i].transAxes, bbox=dict(facecolor="none", pad=5.0))

#For each subplot
for i in subplot:
    # Grids
    i.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    i.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    i.grid(b=True, which="major", color="gray", linewidth=1.0, linestyle="--")
    i.grid(b=True, which="minor", color="gray", linewidth=0.5, linestyle="--")
    # Legend + tick size
    i.legend(fontsize=15)
    i.tick_params(labelsize=13)

#Y-label, X-label, xlim, ylim
subplot[-1].set_xlabel("\n Time [Year-Month]", fontsize=15)
subplot[0].set_ylabel("Value [$] \n", fontsize=15)
subplot[1].set_ylabel("Value [$] \n", fontsize=15)
subplot[2].set_ylabel("Value [$] \n", fontsize=15)
subplot[3].set_ylabel("Value [$] \n", fontsize=15)
subplot[4].set_ylabel("Volume \n", fontsize=15)

#Title
google_stock_plot.text(0.5, 0.91, "Google stock stats from 2008 to 2010", fontsize=20, ha="center")

# google_stock_plot.subplots_adjust(hspace=0.25)
plt.show()
google_stock_plot.savefig("./plots/{:s}".format("google_finance_plot"))


#%%====================================================== Timestamps and Periods
''''''
#%% Timestamps and Periods
"""
-Timestamps: used to represent a point in time
-period: an interval of time it takes to complete one full cycle: or any interval of time
"""

#Creat Timestamp
#    -When only integer inputed: year, month, day, hour, min, sec
timestamp   = pd.Timestamp(2017, 2, 3, 23, 11, 56)
timestamp_2 = pd.Timestamp(2017, 2, 4, 23, 11, 56)


#Creat Period: starting from 2/3/2017 to 2/4/2017
period = pd.Period("2017-02-03-23", freq="D")

#Check if the given timestamp exists in the given priod
if (period.start_time < timestamp < period.end_time):
    print(f"The timestamp {timestamp} lies in between period starttime {period.start_time} and end time {period.end_time}")
if (period.start_time < timestamp_2 < period.end_time):
    print(f"The timestamp_2 {timestamp_2} lies in between period starttime {period.start_time} and end time {period.end_time}")
else:
    print(False)


#Create new period from timestamp
period_new = timestamp.to_period(freq="H")
print(period_new)

#Create new timestamp fram
timestamp_new = period.to_timestamp(freq="H", how="start")
print(timestamp_new)

















#%%====================================================== Using date_range
''''''
#%% Using date_range
"""
date_range: method that returns a fixed frequency datetimeindex
    -Useful when creating your own timeseries attribute for pre-existing data
    -Can arrange the whole data around the timeseries attribute you created
"""

dr1 = pd.date_range(start="1/1/18", end="1/9/18")
print(dr1)
dr2 = pd.date_range(start="1/1/18", end="1/9/19", freq="M")
print(dr2)
dr3 = pd.date_range(end='1/4/2014', periods=8, freq="D")
print(dr3)
dr4 = pd.date_range(end='1/4/2014', periods=8, freq="M")
print(dr4)

#%%====================================================== Shifting and Lags
''''''
#%% Shifting and Lags
"""
-We can shift index by desired number of periods with an optional time frequency
    -Useful when comparing the time series with a past of itself
"""

#Shift by 10 months to compare
humidity_Vancouver         = humidity["Vancouver"].asfreq("M")
humidity_Vancouver_shifted = humidity["Vancouver"].asfreq("M").shift(10)

#%% Humidity plot
humidity_shift_plot = plt.figure(figsize=(10, 6))

#Plot
plt.plot(humidity_Vancouver.index, humidity_Vancouver, lw=3, color="#4C72B0")
plt.plot(humidity_Vancouver_shifted.index, humidity_Vancouver_shifted, lw=2, ls="--", color="#C44E52")

#Plot stats
mean_humidity = np.mean(humidity_Vancouver)
std_humidity  = np.std(humidity_Vancouver)
plt.axhline(mean_humidity, linestyle="--", color="black", alpha=1, linewidth=1.5)
plt.fill_between(humidity_Vancouver.index, mean_humidity+std_humidity, mean_humidity-std_humidity, alpha=0.2)

#Text about stats
plt.text(0.8, 0.15, s="$\mu$-$\sigma$ {:>10.2f}\n$\mu$+$\sigma$ {:>8.2f}".format(mean_humidity - std_humidity, mean_humidity + std_humidity),
         fontsize=15, transform=plt.gca().transAxes, bbox=dict(facecolor="none", pad=5.0))

#Grids
plt.gca().xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(6))
plt.gca().yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
plt.grid(visible=True, which="major", color="gray", linewidth=.5, linestyle="--")
plt.grid(visible=True, which="minor", color="gray", linewidth=0.25, linestyle="--")

#Labels and lim
plt.xlim()
plt.ylim()
plt.xlabel("Measurement date", fontsize=15)
plt.ylabel("Relative Humidity [%]", fontsize=15)
plt.title("Vancouver Humidity in Monthly Frequency vs Shifted")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
humidity_shift_plot.savefig("./plots/{:s}".format("humidity_shift_plot"))

#%%====================================================== Resampling
''''''
#%% Resampling
"""
-Upsampling: Timeseries data resamples from low to high frequency: 
    -Monthly to daily
    -Filling ot interpolating missing data
    
-Downsampling: resample from high to low frequency:
    -Weekly to monthly
    -Aggregation of existing data: combining: mean, median, or medium
"""

#Pressure from weather data
pressure = pd.read_csv("./data/weather_time_series_data/pressure.csv", index_col="datetime",
                       parse_dates=["datetime"])

#Cleaning
pressure = pressure.iloc[1:]
pressure = pressure.fillna(method="ffill")
pressure = pressure.fillna(method="bfill")

#Downssampling
#    -Hourly to 3 day frequency: aggregated using "mean"
#        -3D: 3 days
#        -30S: 30 seconds
#        -3T: 3 minutes
pressure = pressure.resample("3D").mean()

#Upsampling
#    -3 day frequnecy to Daily frequency
#    -pad(): similar to fillna("ffill"): missing values filled
pressure = pressure.resample("D").pad()

#%%===================================================================== 2. Visualize more statistics
''''''
#%% Visualize more statistics


#New column: "Change": How much the stock changed on daily basis
#    -div(): Divide google["High"] by google["High"].shift()
#    -shift(periods=1): shift all column values by 1
google["Change"] = google["High"].div(google["High"].shift(periods=1))






#%%===================================================================== 3. Timeseries Decomposition and Random Walks
''''''


#%%===================================================================== 4. ARIMA model to Predict
''''''
