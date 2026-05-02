import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

# Load league table with xPts and ActualPts

df = pd.read_csv("C:/Users/Pratik/DS/premier-league-ml/data/processed/league_table_expected_vs_actual.csv")

big_6 = ["Man United","Man City","Liverpool","Arsenal","Chelsea","Tottenham"]

# Separate DataFrames for Big 6 and other teams
big_6_df = df[df["Team"].isin(big_6)]
others_df = df[~df["Team"].isin(big_6)]

# Scatter Plot: Expected Points vs Actual Points
# Plot all teams
plt.figure(figsize=(8, 8))
plt.scatter(df["xPts"],df["ActualPts"],color="steelblue",alpha=0.7,label="other teams")

# Plot Big 6 teams with different color
plt.scatter(big_6_df["xPts"],big_6_df["ActualPts"],color="crimson",edgecolor="black", s=90,label="Big 6")

# Reference line y=x
max_pts = max(df["xPts"].max(), df["ActualPts"].max())
plt.plot([0, max_pts],[0, max_pts],linestyle="--",color="gray",label="Perfect prediction")

# Annotate teams 
texts = []
for _, row in others_df.iterrows():
    texts.append(
        plt.text(row["xPts"] + 0.2, row["ActualPts"] + 0.2, row["Team"], fontsize=8, color="dimgray", alpha=0.8)
    )
adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray"),expand_points=(1.2,1.2))

# Highlight Big 6 team names
for _, row in big_6_df.iterrows():
    plt.text(row["xPts"] + 0.4,row["ActualPts"] + 0.4,row["Team"],fontsize=10,fontweight="bold",color="black")

# Labels and Title
plt.xlabel("Expected Points (xPts)")
plt.ylabel("Actual Points")
plt.title("Expected Points vs Actual Points (Big 6 Highlighted)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("C:/Users/Pratik/DS/premier-league-ml/reports/figures/xpts_vs_actual_scatter.png",dpi=300)
plt.show()