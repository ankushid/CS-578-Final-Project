import os

# --- BEGIN: LLM Override Setup ---
import builtins

if not hasattr(builtins, 'llm_action_suggestion'):
    builtins.llm_action_suggestion = {}

llm_action_map = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4}
# --- END: LLM Override Setup ---

import re
import time
from openai import OpenAI

client = OpenAI(
    api_key="sk-544792c2c7de4a128e035fffed01daf9",
    base_url="https://api.deepseek.com/v1"
)

log_dir = "llm_logs"

def get_llm_response(prompt):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                   "Your task is to simulate an RL agent’s behavior, not a conservative human player. Match the aggressive style of the RL agent. Respond with the most profitable action: fold, call, check, raise, or bet. \n"
                   #"You are an AI poker agent trained using reinforcement learning. You should make decisions that match those made by an RL policy. Respond with the most likely RL-style action given the game state.\n"
                   "using ONLY this format:\n"
                   "<action>, <confidence>, <reason>\n"
                   "Valid actions: fold, call, check, raise, bet.\n"
                   "Confidence should be: low, medium, or high.\n"
                   "Do not return anything outside this format.\n"
                   "Only fold if continuing would lead to significant losses with no opportunity to bluff or draw equity. Prefer aggression or value-seeking actions if there's any chance to pressure the opponent. \n"

                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        reply = response.choices[0].message.content.strip()
        print("[LLM RAW OUTPUT]:", reply)
        action, confidence, reason = reply.split(",", 2)
        action = action.strip().lower()
        if action not in ["fold", "call", "check", "raise", "bet"]:
            raise ValueError("Unrecognized action")

        return action.strip(), confidence.strip(), reason.strip()
    except Exception as e:
        print("LLM ERROR:", e)
        return "fold", "low", "Fallback: LLM error"

def main():
    prompt_files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("llm_prompt_") and f.endswith(".txt")]
    )

    for prompt_file in prompt_files:
        prompt_id = prompt_file.split("_")[-1].replace(".txt", "")
        response_file = f"llm_response_deepseek_{int(time.time() * 1000)}_{prompt_id}.txt"
        response_path = os.path.join(log_dir, response_file)

        # if any(f.endswith(f"{prompt_id}.txt") for f in os.listdir(log_dir) if "llm_response_deepseek" in f):
           # continue

        with open(os.path.join(log_dir, prompt_file), "r") as f:
            prompt_text = f.read()
            few_shot_examples = """
            Example 1:
            Game State:
            - Round: PREFLOP
            - Community Cards: []
            - Your Hand: ['HQ', 'C4']
            - Pot Size: 6
            - Stack Sizes: You: 98 | Opponent: 96
            - Opponent Actions: [[1, 2, [0, 1, 2, 3, 4]]]
            - Your Action: 2
            action: check
            confidence: 0.9
            reason: You have a weak hand but are in position. Checking is the safest low-risk play.
            
            Example 2:
            Game State:
            - Round: FLOP
            - Community Cards: ['S4', 'D7', 'H7']
            - Your Hand: ['C9', 'H9']
            - Pot Size: 10
            - Stack Sizes: You: 96 | Opponent: 90
            - Opponent Actions: [[1, 1, [0, 1, 2, 3]]]
            - Your Action: 3
            action: raise
            confidence: 0.85
            reason: Top pair with a good kicker — you can apply pressure on a weak board.
            """
            combined_prompt = few_shot_examples.strip() + "\n\n" + prompt_text.strip()
        action, confidence, reason = get_llm_response(combined_prompt)
        #action, confidence, reason = get_llm_response(prompt_text)

        print(f"[!] Overwriting DeepSeek response for: {prompt_file}")

        with open(response_path, "w") as f:
            f.write(f"Original Prompt File: {prompt_file}\n")
            f.write(f"Suggested Action: {action}\n")
            f.write(f"Confidence: {confidence}\n")
            f.write(f"Reason: {reason}\n")

        print(f"Saved response to {response_path}")
        time.sleep(0.5)

if __name__ == "__main__":
    main()
