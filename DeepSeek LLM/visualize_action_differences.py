import matplotlib.pyplot as plt
from collections import Counter

log_path = "strategy_comparison_log4.txt"

rl_actions = []
llm_actions = []

with open(log_path, "r") as f:
    for line in f:
        if "RL=" in line and "LLM=" in line:
            try:
                rl = line.split("RL=")[1].split(",")[0].strip()
                llm = line.split("LLM=")[1].split(",")[0].strip()
                rl_actions.append(rl)
                llm_actions.append(llm)
            except:
                continue

rl_counts = Counter(rl_actions)
llm_counts = Counter(llm_actions)

all_actions = ["fold", "call", "check", "raise", "bet"]
rl_values = [rl_counts.get(a, 0) for a in all_actions]
llm_values = [llm_counts.get(a, 0) for a in all_actions]

x = range(len(all_actions))
plt.figure(figsize=(10, 6))
plt.bar(x, rl_values, width=0.4, label="RL Agent", align="center")
plt.bar([i + 0.4 for i in x], llm_values, width=0.4, label="LLM", align="center")
plt.xticks([i + 0.2 for i in x], all_actions)
plt.ylabel("Frequency")
plt.title("Action Distribution: RL Agent vs DeepSeek LLM")
plt.legend()
plt.tight_layout()
plt.savefig("action_comparison.png")
plt.show()
