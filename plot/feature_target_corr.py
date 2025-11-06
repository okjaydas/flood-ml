import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
transformed_data_path = project_root / 'data' / 'Flood-Data-Transformed.csv'

features = [
    "Rainfall_today", "DrainLevel_today", "RoadLevel_today", "SoilMoisture_today", "Rainfall_tomorrow"
]
targets = [
    "DrainLevel_tomorrow", "RoadLevel_tomorrow", "SoilMoisture_tomorrow", "FloodProbability"
]

df = pd.read_csv(transformed_data_path)

for target in targets:
    print(f"\nCorrelations with {target}:")
    corrs = df[features + [target]].corr()[target][features]
    print(corrs.sort_values(ascending=False))

corr_matrix = df[features + targets].corr()
plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix.loc[features, targets], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Feature vs Target Correlation Heatmap")
plt.tight_layout()
plt.show()

for target in targets:
    corrs = df[features + [target]].corr()[target][features]
    top_feats = corrs.abs().sort_values(ascending=False).head(2).index
    for feat in top_feats:
        plt.figure()
        sns.scatterplot(x=df[feat], y=df[target], alpha=0.3)
        plt.title(f"{feat} vs {target} (corr={corrs[feat]:.2f})")
        plt.xlabel(feat)
        plt.ylabel(target)
        plt.tight_layout()
        plt.show()
        
print(df["FloodProbability"].mean(), df["FloodProbability"].median())
plt.hist(df["FloodProbability"], bins=50)
plt.show()