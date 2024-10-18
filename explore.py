import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def targ_fig(df):
    sns.histplot(data=df, x='calories_burned', color='gainsboro', edgecolor='black')
    sns.despine()
    plt.xlabel('Calories Burned')
    plt.ylabel('Count')
    plt.title('Distribution centers around 900')
    plt.show()

def duration_fig(df):
    sns.scatterplot(data=df, x='session_duration (hours)', y='calories_burned', color='gainsboro', edgecolor='black')
    sns.despine()
    plt.xlabel('Session (Hours)')
    plt.ylabel('Calories Burned')
    plt.title('Positive correlation between duration and calories')
    plt.show()   

def fat_fig(df):
    sns.scatterplot(data=df, x='fat_percentage', y='calories_burned', color='gainsboro', edgecolor='black')
    sns.despine()
    plt.xlabel('Fat Percentage')
    plt.ylabel('Calories Burned')
    plt.title('Negative correlation between fat and calories')
    plt.show()  

def bpm_fig(df):
    sns.scatterplot(data=df, x='avg_bpm', y='calories_burned', color='gainsboro', edgecolor='black')
    sns.despine()
    plt.xlabel('Average BPM')
    plt.ylabel('Calories Burned')
    plt.title('Positive correlation between average BPM and calories')
    plt.show() 