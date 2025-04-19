import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

LOG_DIR = "llm_logs"
LOG_FILE = "strategy_comparison_log4.txt"

stage_match_counts = defaultdict(lambda: {"match": 0, "total": 0})

with open(LOG_FILE, "r") as f:
    lines = f.readlines()[2:]

for line in lines:
    parts = line.strip().split(":")[0]
    prompt_filename = parts.strip()

    prompt_path = os.path.join(LOG_DIR, prompt_filename)
    if not os.path.exists(prompt_path):
        continue

    stage = None
    with open(prompt_path, "r") as pf:
        for pline in pf:
            if "Round:" in pline:
                stage = pline.strip().split(":")[-1].strip().upper()
                break

    if stage is None:
        continue

    match_str = line.strip().split("Match=")[-1]
    match = match_str.lower() == "true"

    stage_match_counts[stage]["total"] += 1
    if match:
        stage_match_counts[stage]["match"] += 1

stages = ["PREFLOP", "FLOP", "TURN", "RIVER"]
rates = [
    100 * stage_match_counts[stage]["match"] / stage_match_counts[stage]["total"]
    if stage_match_counts[stage]["total"] > 0 else 0
    for stage in stages
]

plt.figure(figsize=(8, 5))
plt.bar(stages, rates, color="teal")
plt.ylim(0, 100)
plt.title("LLM-RL Agreement Rate by Game Stage")
plt.xlabel("Game Stage")
plt.ylabel("Agreement Rate (%)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("llm_rl_agreement_by_stage_v2.png")
plt.show()
