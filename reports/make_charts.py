
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from src.data_generator import generate_data

out = Path(__file__).parent
df = generate_data(30000, 42)

plt.figure(figsize=(9,5))
plt.scatter(df["price"].sample(5000, random_state=42), df["units_sold"].sample(5000, random_state=42), alpha=.25)
plt.xlabel("Price")
plt.ylabel("Units sold")
plt.title("Price vs. units sold — descriptive relationship")
plt.tight_layout()
plt.savefig(out/"price_vs_demand.png", dpi=180)
plt.close()

plt.figure(figsize=(9,5))
df.groupby("category")["units_sold"].mean().sort_values().plot(kind="bar")
plt.ylabel("Average units sold")
plt.title("Average demand by category")
plt.tight_layout()
plt.savefig(out/"demand_by_category.png", dpi=180)
plt.close()

plt.figure(figsize=(9,5))
plt.hist(df["price"], bins=40)
plt.xlabel("Price")
plt.ylabel("Observations")
plt.title("Price distribution")
plt.tight_layout()
plt.savefig(out/"price_distribution.png", dpi=180)
plt.close()

plt.figure(figsize=(9,5))
df.groupby("promotion")["units_sold"].mean().plot(kind="bar")
plt.xticks([0,1], ["No promotion","Promotion"], rotation=0)
plt.ylabel("Average units sold")
plt.title("Average demand by promotion status")
plt.tight_layout()
plt.savefig(out/"promotion_demand.png", dpi=180)
plt.close()

print("Created 4 presentation-ready charts.")
