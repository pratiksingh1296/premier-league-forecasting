import pandas as pd
import numpy as np

# Load raw Premier League data
df = pd.read_csv("C:/Users/Pratik/DS/premier-league-ml/data/raw/PremierLeague.csv")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=False) # Converting Date column to datetime format
df_2 = df[(df['Date'] > '2018-08-01') & (df['Date'] < '2025-08-01')] #Filtering data from 18/19 to 24/25 season

df_2 = df_2.loc[:,['Season','Date','MatchWeek','HomeTeam','AwayTeam', 'FullTimeHomeTeamGoals', 'FullTimeAwayTeamGoals', 'FullTimeResult']] # Selecting relevant columns
df_2.rename(columns={'FullTimeHomeTeamGoals': 'FTHG', 'FullTimeAwayTeamGoals': 'FTAG', 'FullTimeResult': 'Result'}, inplace=True) # Since column names are too long

df_2.info() # Checking the data types 
df_2.sort_values(by=['Date','MatchWeek'], inplace=True) # Sorting the data by Date and MatchWeek
df_2.reset_index(drop=True, inplace=True) # Resetting the index after sorting
print(df_2.head())

# Creating Match outcome (FTR) column based on full-time goals
df_2['FTR'] = np.where(df_2['FTHG'] > df_2['FTAG'], 'H', np.where(df_2['FTHG'] < df_2['FTAG'], 'A', 'D'))  
# Validating the newly created FTR column against the original Result column
is_it_equal = df_2['FTR'].equals(df_2['Result']) #
print("Is the newly created FTR column equal to the original Result column?", is_it_equal) # Additional check to ensure correctness of FTR column 

df_2.drop(columns=['Result'], inplace=True) # Dropping the original Result column as it's redundant now
# Match outcome (FTR) is derived from full-time goals and validated against the provided result column before dropping the original label.

# Save the processed data
df_2.to_csv("C:/Users/Pratik/DS/premier-league-ml/data/processed/PL_processed.csv", index=False)
