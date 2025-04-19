import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

log_path = "strategy_comparison_log4.txt"
output_file = "rl_vs_llm_heatmap.png"

action_pairs = Counter()

with open(log_path, "r") as file:
    for line in file:
        if "RL=" in line and "LLM=" in line:
            parts = line.strip().split(":")[-1].strip().split(",")
            rl_action = parts[0].split("=")[-1].strip()
            llm_action = parts[1].split("=")[-1].strip()
            action_pairs[(rl_action, llm_action)] += 1

actions = ["fold", "call", "check", "raise", "bet"]
df = pd.DataFrame(0, index=actions, columns=actions)

for (rl, llm), count in action_pairs.items():
    if rl in actions and llm in actions:
        df.loc[rl, llm] = count

plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
plt.title("RL Agent vs DeepSeek LLM - Action Match Heatmap")
plt.xlabel("LLM Action")
plt.ylabel("RL Agent Action")
plt.tight_layout()
plt.savefig(output_file)
print(f"Saved heatmap to {output_file}")
