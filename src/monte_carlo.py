from turtle import lt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df =  pd.read_csv("C:/Users/Pratik/DS/premier-league-ml/reports/tables/match_probabilities_xgb_calibrated.csv") 
teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel()) # Get unique team names 

# Function to simulate a single season
def season_simulation(df):
    points = {team: 0 for team in teams}
    
    for _, row in df.iterrows():
        outcome = np.random.choice(
            ['H', 'D', 'A'],
            p=[row['P_Home'], row['P_Draw'], row['P_Away']]
        )
        if outcome == 'H':
            points[row['HomeTeam']] += 3
        elif outcome == 'A':
            points[row['AwayTeam']] += 3
        else:
            points[row['HomeTeam']] += 1
            points[row['AwayTeam']] += 1
    return points

# Validate probabilities sum to 1
assert np.allclose(
    df[['P_Home','P_Draw','P_Away']].sum(axis=1), 
    1.0, 
    atol=1e-6
)

# Monte Carlo Simulations
n_sim = 10000
results = []

for _ in range(n_sim):
    season_points = season_simulation(df)
    results.append(season_points)

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)
ranked = results_df.rank(axis=1, ascending=False) # Rank teams for each simulation

# CI Calculation 5th , 50th & 95th Percentiles 
def compute_ci(results_df):
    ci_df = pd.DataFrame()
    ci_df['Mean'] = results_df.mean()
    ci_df['Std_Dev'] = results_df.std()
    ci_df['5th_Percentile'] = results_df.quantile(0.05)
    ci_df['50th_Percentile'] = results_df.quantile(0.50)
    ci_df['95th_Percentile'] = results_df.quantile(0.95)
    return ci_df

# Confidence Intervals Table
CI_table = compute_ci(results_df).sort_values(by='Mean', ascending=False)
print("\nMonte Carlo Simulation - Confidence Intervals:\n")
print(CI_table,'\n')

# Summary Statistics
mc_summary = pd.DataFrame({
    "Expected_Points": results_df.mean(),
    "Std_Points": results_df.std(),
    "Title_Prob": (ranked == 1).mean(),
    "Top4_Prob": (ranked <= 4).mean(),
    "Relegation_Prob": (ranked >= 18).mean()
}).sort_values("Expected_Points", ascending=False)
print(mc_summary,'\n')

# Comparintg with Expected Points from league_table.py
league_table = pd.read_csv("C:/Users/Pratik/DS/premier-league-ml/reports/tables/league_table_expected_vs_actual.csv", index_col='Position')
comparison = league_table.merge(
    mc_summary,
    left_on='Team',
    right_index=True,
    suffixes=('_LeagueTable', '_MonteCarlo')
)
print("\nComparison of League Table Expected Points vs Monte Carlo Expected Points:\n")
print(comparison)


# --- Plots ---

# Filtering Big 6 Teams CI Table
top_teams = ['Man United', 'Man City', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham'] 
filtered_CI = CI_table.loc[top_teams]
print("Confidence Intervals for Big 6 Teams:\n")
print(filtered_CI,'\n')

# Boxplot for Big 6 Teams
plot_CI = filtered_CI[['5th_Percentile', '50th_Percentile', '95th_Percentile']]
plt.figure(figsize=(10,6))
plt.barh(plot_CI.index, plot_CI['95th_Percentile'] - plot_CI['5th_Percentile'], left=plot_CI['5th_Percentile'], color='lightblue', alpha=0.6, label='90% CI')
plt.barh(plot_CI.index, 0.1, left=plot_CI['50th_Percentile'] - 0.05, color='blue', alpha=0.9, label='Median')
plt.xlim(35, 85)
plt.title('Season Points Big 6 Teams with Confidence Intervals')
plt.xlabel('Season Points')
plt.ylabel('Teams')
plt.axvline(40, color='red', linestyle='dashed', linewidth=1, label='Relegation Zone')
plt.axvline(60, color='green', linestyle='dashed', linewidth=1, label='European Qualification Zone')
plt.axvline(70, color='blue', linestyle='dashed', linewidth=1, label='Title Contention Zone')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("C:/Users/Pratik/DS/premier-league-ml/reports/figures/big_6_season_points_ci.png",dpi=300)
plt.show()

# Histogram Subplots for Big 6 Teams
fig , axes = plt.subplots(len(top_teams),1, figsize=(10,15), sharex=True)
for i, team in enumerate(top_teams):
    axes[i].hist(results_df[team], bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[i].set_xlim(30,85)
    axes[i].axvline(40, color='red', linestyle='dashed', linewidth=1, label='Relegation Zone')
    axes[i].axvline(60, color='green', linestyle='dashed', linewidth=1, label='European Qualification Zone')
    axes[i].axvline(70, color='blue', linestyle='dashed', linewidth=1, label='Title Contention Zone')
    axes[i].axvline(results_df[team].mean(), color='black', linestyle='solid', linewidth=2, label='Mean')   
    axes[i].set_title(f'Season Points Distribution for {team}')    
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
    axes[i].grid(alpha=0.3)
axes[-1].set_xlabel('Season Points')
plt.tight_layout(pad=3.0)  
plt.savefig("C:/Users/Pratik/DS/premier-league-ml/reports/figures/big_6_season_points_histograms.png",dpi=300)
plt.show()

# --- Saving Results --- 

comparison.to_csv("C:/Users/Pratik/DS/premier-league-ml/reports/tables/league_table_monte_carlo_comparison.csv", index=True) #Saving comparison results
mc_summary.to_csv("C:/Users/Pratik/DS/premier-league-ml/reports/tables/monte_carlo_season_summary.csv", index=True) # Saving Monte Carlo Summary
results_df.to_csv("C:/Users/Pratik/DS/premier-league-ml/reports/tables/monte_carlo_season_simulations.csv", index=False) # Saving all simulation results