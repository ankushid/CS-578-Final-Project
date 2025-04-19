import os
import gym
import numpy as np
import builtins
from gym import spaces
from ray.rllib.agents.impala.vtrace_policy import VTraceTFPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
import copy
from io import StringIO 
import sys
tf = try_import_tf()
import rlcard
from rlcard.utils import set_seed
import random
from generate_llm_response import get_llm_response
import os
from datetime import datetime

def log_llm_prompt(game_state_dict, player_id, action_taken):
    prompt_dir = os.path.expanduser("~/AlphaNLHoldem/llm_logs")
    os.makedirs(prompt_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prompt_dir}/llm_prompt_{timestamp}.txt"

    prompt = f"""
Game State:
- Round: {game_state_dict['round']}
- Community Cards: {game_state_dict['community']}
- Your Hand: {game_state_dict['hand']}
- Pot Size: {game_state_dict['pot']}
- Stack Sizes: You: {game_state_dict['player_stack']} | Opponent: {game_state_dict['opp_stack']}
- Opponent Actions: {game_state_dict['opp_actions']}
- Your Action: {action_taken}

What should the player do next?
"""

    with open(filename, "w") as f:
        f.write(prompt)
    return filename

def get_llm_suggestion(prompt_path):
    response_path = prompt_path.replace("_prompt_", "_response_")
    if not os.path.exists(response_path):
        return None, None, None
    with open(response_path, "r") as f:
        lines = f.readlines()
        action, confidence, reason = None, None, None
        for line in lines:
            if line.lower().startswith("suggested action:"):
                action = line.split(":")[1].strip().lower()
            elif line.lower().startswith("confidence:"):
                confidence = line.split(":")[1].strip().lower()
            elif line.lower().startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
    return action, confidence, reason

#def auto_generate_llm_response(prompt_path):
    #response_path = prompt_path.replace("_prompt_", "_response_")
    #print(f"[DEBUG] Attempting to generate response for {response_path}")

    #if os.path.exists(response_path):
        #print(f"[DEBUG] Response already exists. Skipping: {response_path}")
        #return  # Don't overwrite

    #import random
    #action = random.choice(["fold", "call", "raise", "check", "bet"])
    #confidence = random.choice(["low", "medium", "high"])
    #reason = f"Mock reason: LLM chose {action} with {confidence} confidence."

    #print(f"[DEBUG] Writing mock response: {action}, {confidence}, {reason}")

    #with open(response_path, "w") as f:
        #f.write(f"Suggested Action: {action}\n")
        #f.write(f"Confidence: {confidence}\n")
        #f.write(f"Reason: {reason}\n")


color2ind = dict(zip("CDHS",[0,1,2,3]))
rank2ind = dict(zip("23456789TJQKA",[0,1,2,3,4,5,6,7,8,9,10,11,12]))

class NlHoldemEnvWrapper():
    def __init__(self,policy_config,weights=None):
        self.policy_config = policy_config
        seed = random.randint(0,1000000)
        self.env = rlcard.make(
            'no-limit-holdem',
            config={
                'seed': seed,
            }
        )
        set_seed(seed)
        self.action_num = 5
        
        
        space = {
                'card_info': spaces.Box(low=-1024, high=1024, shape=(4,13,6)),
                'action_info': spaces.Box(low=-256, high=256, shape=(4,self.action_num,4 * 6 + 1)),
                'extra_info': spaces.Box(low=-256, high=256, shape=(2,)),
                'legal_moves': spaces.Box(
                    low=-1,
                    high=1,
                    shape=list(
                        [self.action_num,]
                    )
                ),
            }
        
        self.observation_space = spaces.Dict(space)
        self.action_space = spaces.Discrete(self.action_num)
        
    @property
    def unwrapped(self):
        return None

    def _get_observation(self,obs):
        card_info = np.zeros([4,13,6],np.uint8)
        action_info = np.zeros([4,self.action_num,4 * 6 + 1],np.uint8) # 25 channel
        extra_info = np.zeros([2],np.uint8) # 25 channel
        legal_actions_info = np.zeros([self.action_num],np.uint8) # 25 channel
        
        hold_card = obs[0]["raw_obs"]["hand"]
        public_card = obs[0]["raw_obs"]["public_cards"]
        current_legal_actions = [i.value for i in obs[0]["raw_obs"]["legal_actions"]]
        
        for ind in current_legal_actions:
            legal_actions_info[ind] = 1
        
        flop_card = public_card[:3]
        turn_card = public_card[3:4]
        river_card = public_card[4:5]
        
        for one_card in hold_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][0] = 1
            
        for one_card in flop_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][1] = 1
            
        for one_card in turn_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][2] = 1
            
        for one_card in river_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][3] = 1
            
        for one_card in public_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][4] = 1
            
        for one_card in public_card + hold_card:
            card_info[color2ind[one_card[0]]][rank2ind[one_card[1]]][5] = 1
            
        
        for ind_round,one_history in enumerate(self.history):
            for ind_h,(player_id,action_id,legal_actions) in enumerate(one_history[:6]):
                action_info[player_id,action_id,ind_round * 6 + ind_h] = 1
                action_info[2,action_id,ind_round * 6 + ind_h] = 1
                
                for la_ind in legal_actions:
                    action_info[3,la_ind,ind_round * 6 + ind_h] = 1
                    
        action_info[:,:,-1] = self.my_agent()
        
        extra_info[0] = obs[0]["raw_obs"]["stakes"][0]
        extra_info[1] = obs[0]["raw_obs"]["stakes"][1]
        
        return {
            "card_info": card_info,
            "action_info": action_info,
            "legal_moves": legal_actions_info,
            "extra_info": extra_info,
        }
    
    def _log_action(self,action_ind):
        self.history[
            self.last_obs[0]["raw_obs"]["stage"].value
        ].append([
            self.last_obs[0]["raw_obs"]["current_player"],
            action_ind,
            [x.value for x in self.last_obs[0]["raw_obs"]["legal_actions"]]
        ])
    
    def my_agent(self):
        return self.env.get_player_id()
    
    def convert(self,reward):
        return float(reward)
        
    def step(self, action):
        # --- BEGIN: LLM Action Override ---
        try:
            prompt_file = self.last_prompt_file
            if hasattr(builtins, 'llm_action_suggestion') and prompt_file in builtins.llm_action_suggestion:
                llm_action = builtins.llm_action_suggestion[prompt_file]
                if llm_action is not None:
                    print(f"[LLM Override] Replacing RL action {action} with LLM suggestion: {llm_action}")
                    action = llm_action
        except Exception as e:
            print(f"[LLM Override Error] {e}")
        # --- END: LLM Action Override ---
        self._log_action(action)
        obs = self.env.step(action)
        self.last_obs = obs

        if self.env.get_player_id() == 0 and not self.env.game.is_over():
            raw_obs = self.last_obs[0]["raw_obs"]
            game_state = {
                "round": raw_obs["stage"].name,
                "community": raw_obs["public_cards"],
                "hand": raw_obs["hand"],
                "pot": raw_obs["pot"],
                "player_stack": raw_obs["stakes"][0],
                "opp_stack": raw_obs["stakes"][1],
                "opp_actions": [entry for round_hist in self.history for entry in round_hist if entry[0] == 1],
            }
            prompt_path = log_llm_prompt(game_state, player_id=0, action_taken=action)
            #auto_generate_llm_response(prompt_path)
            with open(prompt_path, 'r') as f:
                prompt_content = f.read()
            get_llm_response(prompt_content)
            llm_action, llm_conf, llm_reason = get_llm_suggestion(prompt_path)

            #prompt_path = log_llm_prompt(game_state, player_id=0, action_taken=action)
            # prompt_dir = os.path.expanduser("~/AlphaNLHoldem/llm_logs")
            # prompt_files = sorted([f for f in os.listdir(prompt_dir) if f.startswith("llm_prompt_")], reverse=True)
            # prompt_path = os.path.join(prompt_dir, prompt_files[0])  latest prompt file
            # auto_generate_llm_response(prompt_path)
            # llm_action, llm_conf, llm_reason = get_llm_suggestion(prompt_path)

            with open("log/llm_comparison_log.txt", "a") as f:
                f.write(f"RL Action: {action}, LLM Action: {llm_action}, Confidence: {llm_conf}, Agreement: {action == llm_action}\n")

            #if llm_action and llm_conf == "high" and llm_action != str(action):
                #print(f"[LLM Override] Replacing RL action {action} with LLM suggestion: {llm_action}")
                #try:
                    #action = int(llm_action)
                #except:
                    #print(f"[Override Error] Could not convert LLM action '{llm_action}' to integer.")

        # Use DeepSeek LLM suggestion if confidence is high and action is different
            if llm_action and llm_conf == "high" and llm_action != str(action):
                print(f"[LLM Override] Replacing RL action {action} with LLM suggestion: {llm_action}")
    
                llm_action_map = {
                    "fold": 0,
                    "call": 1,
                    "check": 1,  # sometimes DeepSeek says "check" when it's same as call
                    "raise": 2,
                    "bet": 3,
                }

                if llm_action in llm_action_map:
                    action = llm_action_map[llm_action]
                else:
                    print(f"[Override Error] Unrecognized LLM action: {llm_action}")

        obs = self._get_observation(obs)
        
        done = False
        reward = [0,0]
        aggressive_actions = [3, 4]
        if action in aggressive_actions:
            reward[self.env.get_player_id()] -= 0.1
            #if not self.player_won_last_round:
                #reward -= 0.1

        info = {}
        if self.env.game.is_over():
            done = True
            reward = list(self.env.get_payoffs())
            
        return obs,reward,done,info

    def reset(self):
        self.history = [[],[],[],[]]
        obs = self.env.reset()
        self.last_obs = obs
        return self._get_observation(obs)
    
    def legal_moves(self):
        pass
    

class NlHoldemEnvWithOpponent(NlHoldemEnvWrapper):
    def __init__(self,policy_config,weights=None,opponent="nn"):
        super(NlHoldemEnvWithOpponent, self).__init__(policy_config,weights)
        self.opponent = opponent
        self.rwd_ratio = policy_config["env_config"]["custom_options"].get("rwd_ratio",1)
        self.is_done = False
        if self.opponent == "nn":
            self.oppo_name = None
            self.oppo_preprocessor = ModelCatalog.get_preprocessor_for_space(self.observation_space, policy_config.get("model"))
            self.graph = tf.Graph()
            with self.graph.as_default():
                with tf.variable_scope('oppo_policy'):
                    self.oppo_policy = VTraceTFPolicy(
                        obs_space=self.oppo_preprocessor.observation_space,
                        action_space=self.action_space,
                        config=policy_config,
                    )
            if weights is not None:
                import pickle
                with open(weights,'rb') as fhdl:
                    weights = pickle.load(fhdl)
                self.oppo_policy.set_weights(weights)
        
    def _opponent_step(self,obs):
        if self.opponent == "random":
            rwd = [0,0]
            done = False
            info = {}
            while self.my_agent() != self.our_pid:
                legal_moves = obs["legal_moves"]
                action_ind = np.random.choice(np.where(legal_moves)[0])
                obs,rwd,done,info = super(NlHoldemEnvWithOpponent, self).step(action_ind)
                if done:
                    break
            return obs,rwd,done,info
        elif self.opponent == "nn":
            rwd = [0,0]
            done = False
            info = {}
            while self.my_agent() != self.our_pid:
                observation = self.oppo_preprocessor.transform(obs)
                action_ind = self.oppo_policy.compute_actions([observation])[0][0]
                obs,rwd,done,info = super(NlHoldemEnvWithOpponent, self).step(action_ind)
                if done:
                    break
            return obs,rwd,done,info
        else:
            raise        
        
    def reset(self):
        self.last_reward = 0
        self.is_done = False
        self.our_pid = random.randint(0,1)
        
        obs = super(NlHoldemEnvWithOpponent, self).reset()
        
        while True:
            obs,rwd,done,info = self._opponent_step(obs)
            if not done:
                return obs
            else:
                obs = super(NlHoldemEnvWithOpponent, self).reset()
            
    def step(self,action):
        obs,reward,done,info = super(NlHoldemEnvWithOpponent, self).step(action)
        reward = [i * self.rwd_ratio for i in reward]
        if done:
            self.is_done = True
            self.last_reward = reward[self.our_pid]
            return obs,reward[self.our_pid],done,info
        else:
            obs,reward,done,info = self._opponent_step(obs)
            reward = [i * self.rwd_ratio for i in reward]
            if done:
                self.is_done = True
                self.last_reward = reward[self.our_pid]
            return obs,reward[self.our_pid],done,info
        
