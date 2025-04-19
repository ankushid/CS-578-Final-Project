# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:41:11 2025

@author: jmatulis
"""
'''
Steps to prepare cases for ML



categories of situation
start no blind
start small blind
start big blind
post flop
post turn
post river


info (x)
number of players confirmed in
number of players potential in
current call
pot size
score
hand score
stage

actions (y E [fold, check/call, raise])     Combining call/check bc they both mean "stay in". in future maybe separate, maybe try to predict raise size


other info to calculate for a situation
delta_loss/win
In games where player stays in until end, should/should_not have folded




cycle through games. extract situations from games
'''

import pandas as pd
import re
import sys
import ast
import json
import treys
import math
from treys import Card, Evaluator
import numpy as np



game_data = pd.read_csv("cleaned_game_data.csv")
actions = pd.read_csv("cleaned_actions.csv")
player_stats = pd.read_csv("cleaned_player_stats.csv")


#%%

games_to_examine = range(1000)


_CHEN_BASE = {
    'A': 10.0,
    'K': 8.0,
    'Q': 7.0,
    'J': 6.0,
    'T': 5.0,  # Treys uses 'T' for Ten
    '9': 4.5,
    '8': 4.0,
    '7': 3.5,
    '6': 3.0,
    '5': 2.5,
    '4': 2.0,
    '3': 1.5,
    '2': 1.0,
}

_card_re_treys  = re.compile(r'^([2-9TJQKA])([hdcs])$', re.IGNORECASE)

def chen_score_treys(card1: str, card2: str) -> float:
    """
    Compute the Chen formula score for two hole cards in Treys format.
    Accepts ranks 2-9, T, J, Q, K, A and suits h,d,c,s.
    Returns the rounded Chen score.
    """
    def parse(c):
        m = _card_re_treys.match(c.strip())
        if not m:
            raise ValueError(f"Invalid Treys card: {c!r}. Expected e.g. 'Th', 'As', '7d'.")
        return m.group(1).upper(), m.group(2).lower()

    r1, s1 = parse(card1)
    r2, s2 = parse(card2)

    v1 = _CHEN_BASE[r1]
    v2 = _CHEN_BASE[r2]
    # Identify high and low
    if v1 >= v2:
        high_rank, high_val, low_rank, low_val = r1, v1, r2, v2
    else:
        high_rank, high_val, low_rank, low_val = r2, v2, r1, v1

    # 1) Pair
    if high_rank == low_rank:
        score = 2 * high_val
        if score < 5.0:
            score = 5.0
    else:
        # 2) Base = high‐card
        score = high_val
        # 3) Suited bonus
        if s1 == s2:
            score += 2.0

        # 4) Gap penalty
        # map ranks to numeric values for gap
        rank_order = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
                      '8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
        gap = abs(rank_order[high_rank] - rank_order[low_rank]) - 1
        if gap == 0:
            pass
        elif gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        else:
            score -= 5.0

        # 5) Straight‐draw bonus for connectors
        if gap == 0:
            score += 1.0

    # 6) Round up to nearest half‑point
    return math.ceil(score * 2.0) / 2.0

_card_re = re.compile(r'^(10|[2-9AJQK])([hdcs])$', re.IGNORECASE)


def convert_card_to_treys(card: str) -> str:
    """
    Convert a single card from your format (e.g. "10h", "As", "kd") into Treys format (e.g. "Th", "As", "Kd").
    
    - Ranks: 2–9, A, J, Q, K, or 10  → Treys wants 2–9, A, J, Q, K, T
    - Suits: h, d, c, s (case‑insensitive)
    
    Raises ValueError if the input doesn’t match the expected pattern.
    """
    m = _card_re.match(card.strip())
    if not m:
        raise ValueError(f"Invalid card string: {card!r}. Expected e.g. '10h', 'As', '7d'.")
    rank, suit = m.group(1).upper(), m.group(2).lower()
    if rank == "10":
        rank = "T"
    return f"{rank}{suit}"

def convert_cards_to_treys(cards: list[str]) -> list[str]:
    """
    Map a list of card strings in your format to Treys format.
    
    Example:
      >>> convert_cards_to_treys(["10h","As","7d"])
      ["Th","As","7d"]
    """
    return [convert_card_to_treys(c) for c in cards]




class Situation:
    def __init__(self,
                 stage: str,
                 hand: list,
                 hand_score: int,
                 board_cards: list,
                 score: int,
                 confirmed_in: int,
                 potential_in: int,
                 current_call: float,
                 pot_size: float,
                 action_fold: int,
                 action_check: int,
                 action_call: int,
                 action_raise: float):
        self.stage          = stage
        self.hand           = hand
        self.hand_score     = hand_score
        self.board_cards    = board_cards
        self.score          = score
        self.confirmed_in   = confirmed_in
        self.potential_in   = potential_in
        self.current_call   = current_call
        self.pot_size       = pot_size
        self.action_fold    = action_fold
        self.action_check   = action_check
        self.action_call    = action_call
        self.action_raise   = action_raise
        

        # if self.board_cards:
        #     evaluator  = Evaluator()
        #     print(self.board_cards)
        #     for c in range(len(self.board_cards)):
        #         print(self.board_cards[c], flush=True)
        #         sys.stdout.flush ()
        #     for raw in self.hand:
        #         print("RAW HAND ELEMENT:", repr(raw))
        #     for raw in self.board_cards:
        #         print("RAW BOARD ELEMENT:", repr(raw))
        #     hole_ints  = [Card.new(c) for c in self.hand]          # must be length 2
        #     board_ints = [Card.new(c) for c in self.board_cards]   # length 1–5
        #     self.score = evaluator.evaluate(board_ints, hole_ints)
        # else:
        #     self.score = 0
        
        
        

    def __repr__(self):
        return (f"Situation(stage={self.stage}, board={self.board_cards}, score={self.score}, "
                f"hand={self.hand}, hand score={self.hand_score}, "
                f"confirmed_in={self.confirmed_in}, potential_in={self.potential_in}, "
                f"current_call={self.current_call}, pot_size={self.pot_size}, "
                f"fold={self.action_fold}, check={self.action_check}, "
                f"call={self.action_call}, raise={self.action_raise})")



def divide_into_loops(actions_df: pd.DataFrame) -> list:
    """
    Divide a DataFrame of actions into loops.
    A loop ends and a new one starts whenever:
      1) a player who has already acted in the current loop acts again, or
      2) the 'state' (stage) changes from the previous action.
    """
    loops = []
    current_indices = []
    seen_players = set()
    prev_stage = None

    for idx, row in actions_df.iterrows():
        player = row["name"]
        stage  = row["state"]

        # boundary if stage changes or player repeats
        boundary = (prev_stage is not None and stage != prev_stage) or (player in seen_players)

        if boundary and current_indices:
            loops.append(actions_df.loc[current_indices].copy())
            current_indices = []
            seen_players = set()

        current_indices.append(idx)
        seen_players.add(player)
        prev_stage = stage

    if current_indices:
        loops.append(actions_df.loc[current_indices].copy())

    return loops

class Game:
    """
    Class representing a complete game.
    
    Attributes:
        full_data_player_seat (int): The seat number of the player whose data is complete.
        data (pd.DataFrame): Pandas DataFrame containing the game log.
        situations (list): List of Situation objects generated from the game log.
    """
    def __init__(self, game_row: pd.Series, actions_slice: pd.DataFrame, player_stats_slice: pd.DataFrame):
        """
        Initialize a Game object using a single row from the games DataFrame 
        and the corresponding slice of actions from the actions DataFrame.
        
        Parameters:
            game_row (pd.Series): A single row from the games DataFrame.
            actions_slice (pd.DataFrame): A DataFrame slice containing the actions
                for this game (assumed duplicate-free and relevant to this game).
        """
        # Store basic game-level data
        self.game_id = game_row["gameId"]
        self.duration = game_row["duration"]
        self.button = game_row["button"]
        self.end_state = game_row["endState"]
        self.pot = game_row["pot"]
        self.rake = game_row["rake"]
        self.fee = game_row["fee"]

        # Parse the seats and players columns (assumed to be JSON strings)
        try:
            self.seats = ast.literal_eval(game_row["seats"])
        except Exception:
            self.seats = None

        # Process the 'players' field.
        # Original format (example):
        # "[{""seat"": 1, ""chips"": 105.78, ""name"": ""StephCurry"", ""cards"": []}, 
        #   {""seat"": 2, ""chips"": 101.0, ""name"": ""PANDAisEVIL"", ""cards"": []}, 
        #   ...,
        #   {""seat"": 9, ""chips"": 273.59, ""name"": ""VegetablesArentYummy"", ""cards"": [""9d"", ""7d""]}]"
        #
        # The trick here is to fix the nonstandard quoting.
        players_str = game_row["players"]
        # Strip off surrounding quotes (if present)
        if players_str.startswith('"') and players_str.endswith('"'):
            players_str = players_str[1:-1]
        # Replace doubled double-quotes with a single double quote
        players_str = players_str.replace('""', '"')
        try:
            players_list = json.loads(players_str)
        except Exception as e:
            players_list = None

        # Find the seat of the main player "IlxxxlI"
        self.main_seat = None
        if players_list:
            for player in players_list:
                if player.get("name") == "IlxxxlI":
                    self.main_seat = player.get("seat")
                    break

        # Get the cards for the last player in the list.
        self.cards = [None, None]
        if players_list and len(players_list) > 0:
            last_player = players_list[-1]
            cards = last_player.get("cards", [])
            if len(cards) >= 1:
                self.cards[0] = cards[0]
                self.cards[1] = cards[1] if len(cards) > 1 else None
                
        self.main_bet     = None
        self.main_collect = None
        self.main_win     = None
        self.main_lose    = None
        # Also, a list to hold cards data for all other players.
        self.other_cards  = []  # List of lists; inner lists like ['Ad', '10d']

        if player_stats_slice is not None and not player_stats_slice.empty:
            for idx, row in player_stats_slice.iterrows():
                if row["name"] == "IlxxxlI":
                    # This is our main player.
                    self.main_bet     = row["bet"]
                    self.main_collect = row["collect"]
                    self.main_win     = row["win"]
                    self.main_lose    = row["lose"]
                else:
                    # For other players, check if the cards data is populated.
                    cards_str = row["cards"]
                    if cards_str or cards_str.strip() != "":
                        try:
                            # The cards data is stored as a string like "['Ad', '10d']".
                            card_list = ast.literal_eval(cards_str)
                            if isinstance(card_list, list):
                                self.other_cards.append(card_list)
                        except Exception:
                            pass  # Ignore if parsing fails.


        # Process the 'board' field (assumed to be a JSON string).
        try:
            self.board = json.loads(game_row["board"])
        except Exception:
            self.board = None
        
        self.cards = convert_cards_to_treys(self.cards) 
        self.board = convert_cards_to_treys(self.board) 
        
        self.hand_score = chen_score_treys(self.cards[0],self.cards[1])
        
        # Save the slice of actions for later processing (e.g., to create Situation objects).
        self.actions = actions_slice.copy()
        self.situations = []  # To be populated later
        
        
        self.generate_situations()


    def generate_situations(self):
        """
        Process the PREFLOP actions to create a Situation object.
        
        Calculation details:
          - players_before: Count of players who acted before IlxxxlI and did not fold.
          - players_after: Number of players with actions after IlxxxlI (considered to be yet to play).
          - players_still_in: Sum of players_before (non-folded) plus players_after.
          - The highest bet is determined by taking the last positive bet (value > 0) among the actions
            before IlxxxlI and then summing all bets made by that player in this stage.
          - current_call and call_delta are both set to that highest bet.
          - pot_size is the cumulative sum of all bets (across all stages of the game so far).
          - The ML target is encoded as one of four values:
                fold: 1 if IlxxxlI folded, otherwise 0,
                check: 1 if checked, otherwise 0,
                call: 1 if called, otherwise 0,
                raise: a float value if raised, 0 otherwise.
        """
        loops = divide_into_loops(self.actions)
    
        # # For debugging, you might print out the loops:
        # for i, loop in enumerate(loops):
        #     print(f"Loop {i}:")
        #     print(loop)
        
        
        for loop_df in loops:
            stage = loop_df["state"].iloc[0]   # e.g. "PREFLOP", "FLOP", etc.
            # 4) find the index label where IlxxxlI acts
            ilx_rows = loop_df[loop_df["name"] == "IlxxxlI"]
            if ilx_rows.empty:
                continue  # skip loops where IlxxxlI doesn't act

            action_idx = ilx_rows.index[0]  # original index label

            # 5) build all actions up to (but not including) IlxxxlI’s action
            up_to_call = self.actions.loc[self.actions.index < action_idx]

            # 6) cumulative bets per player up to this point
            cum_bets = up_to_call.groupby("name")["value"].sum()
            max_bet  = float(cum_bets.max()) if not cum_bets.empty else 0.0

            # 7) confirmed_in: how many (excluding IlxxxlI) are at that max_bet
            confirmed_in = int((cum_bets.drop("IlxxxlI", errors="ignore") == max_bet).sum())

            # 8) potential_in:
            #   a) yet_to_play: players after IlxxxlI in this loop who haven't folded
            idxs = list(loop_df.index)
            after_idxs = [i for i in idxs if i > action_idx]
            after_df   = loop_df.loc[after_idxs]
            # yet_to_play = after_df[after_df["action"].str.lower() != "fold"]["name"].nunique()
            yet_to_play = (
                after_df[
                    (after_df["action"].str.lower() != "fold") &
                    (after_df["name"].map(lambda p: cum_bets.get(p, 0.0)) < max_bet)
                ]["name"]
                .nunique()
            )

            #   b) under_bet: players who acted this loop before IlxxxlI,
            #      didn't fold, but cum_bets < max_bet
            before_idxs   = [i for i in idxs if i < action_idx]
            before_df     = loop_df.loc[before_idxs]
            played_not_fold = before_df[before_df["action"].str.lower() != "fold"]["name"].unique()
            under_bet       = sum(1 for p in played_not_fold if cum_bets.get(p, 0.0) < max_bet)

            potential_in = yet_to_play + under_bet

            # 9) current_call is simply max_bet
            current_call = max_bet

            # 10) pot_size is sum of all bets up to (but not including) this action
            pot_size = up_to_call["value"].sum()

            # 11) encode IlxxxlI’s action in this loop
            row = ilx_rows.iloc[0]
            act = row["action"].lower()
            action_fold  = 1 if act == "fold" else 0
            action_check = 1 if act == "check" else 0
            action_call  = 1 if act == "calls" else 0
            action_raise = row["value"] if act in {"raise", "raises"} else 0.0

            # select board based on stage
            if stage == "PREFLOP":
                board_cards = []
            elif stage == "FLOP":
                board_cards = self.board[:3]
            elif stage == "TURN":
                board_cards = self.board[:4]
            elif stage == "RIVER":
                board_cards = self.board[:]
            else:
                board_cards = self.board or []

            
            if board_cards:
                evaluator = Evaluator()
                # hole_norm  = [normalize_treys_card(c) for c in self.cards]
                # board_norm = [normalize_treys_card(c) for c in self.board_cards]
                try:
                    hole_ints  = [Card.new(c) for c in self.cards]
                    board_ints = [Card.new(c) for c in board_cards]
                    
                    score = evaluator.evaluate(board_ints, hole_ints)
                except Exception as e:
                    # Log exactly what went into Card.new/evaluate
                    print(">>> Exception scoring hand! <<<", flush=True)
                    print("  hole_norm  =", self.cards,  flush=True)
                    print("  board_norm =", board_cards, flush=True)
                    raise
            else:
                score = 0
            
                
            sit = Situation(
                stage=stage,
                hand=self.cards,
                hand_score=self.hand_score,
                board_cards=board_cards,
                score=score,
                confirmed_in=confirmed_in,
                potential_in=potential_in,
                current_call=current_call,
                pot_size=pot_size,
                action_fold=action_fold,
                action_check=action_check,
                action_call=action_call,
                action_raise=action_raise
            )
            self.situations.append(sit)

    def __repr__(self):
        return (f"Game(ID={self.game_id}, main_seat={self.main_seat}, main_bet={self.main_bet}, "
                f"main_collect={self.main_collect}, main_win={self.main_win}, main_lose={self.main_lose}, "
                f"other_cards={self.other_cards}, Situations={self.situations})")







games_list = []
# Loop through each row in the games DataFrame
for index, game_row in game_data.iterrows():
    if index == 45000:
        break
    if index%10 == 0:
        print(index)
    # if index == 4:
    #     continue
    # if index == 6:
    #     continue
    # if index == 50:
    #     break
    game_id = game_row['gameId']
    # Filter the actions_df to get only actions that belong to this game.
    actions_slice = actions[actions['gameId'] == game_id]
    player_stats_slice = player_stats[player_stats['gameId'] == game_id]
    game_obj = Game(game_row, actions_slice, player_stats_slice)
    games_list.append(game_obj)
    
#%%
    
'''
Tomorrow, turn data into one or two 52 length card vectors. and then vectors containing the other attributes. Then f

'''
'''

class_map = {
    0: "fold",
    1: "check",
    2: "call",
    3: "raise",
}

# 2) Collect features & labels
rows = []
labels = []
STAGE_FILTER = "PREFLOP"

for game in games_list:
    for sit in game.situations:
        # input features
        if sit.stage != STAGE_FILTER:
        # if sit.stage == STAGE_FILTER:
            continue
        feat = {
            "hand_score":    sit.hand_score,
            "board_score":   sit.score,
            "confirmed_in":  sit.confirmed_in,
            "potential_in":  sit.potential_in,
            "current_call":  sit.current_call,
            "pot_size":      sit.pot_size,
        }
        rows.append(feat)
        
        # target label
        if sit.action_raise > 0:
            lbl = 3
        elif sit.action_call:
            lbl = 2
        elif sit.action_check:
            lbl = 1
        else:
            lbl = 0
        labels.append(lbl)

# 3) Build DataFrame / Series
X_df = pd.DataFrame(rows)           # shape (n_samples, 6)
y_ds    = pd.Series(labels, name="action_label")  # shape (n_samples,)



# # 1) Build a single DataFrame with your features + label
# df = X_df.copy()
# df['action_label'] = y_ds

# # 2) Find the smallest class size
# counts    = df['action_label'].value_counts()
# min_count = counts.min()
# # min_count = 1000

# print("Original class counts:\n", counts, "\nBalancing to:", min_count)

# # 3) Undersample each class to that size
# balanced = pd.concat([
#     # df[df['action_label'] == cls].sample(n=min_count, random_state=42, replace=True)
#     df[df['action_label'] == cls].sample(n=min_count, random_state=42, replace=False)
#     for cls in counts.index
# ], ignore_index=True)

# # 4) Shuffle the balanced DataFrame
# balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# # 5) Split back into X and y
# y_bal = balanced['action_label'].values
# X_bal = balanced.drop('action_label', axis=1).values

# # 6) Confirm
# print("Balanced class counts:\n", pd.Series(y_bal).value_counts())


# X_df = balanced.drop('action_label', axis=1)
# y_ds = balanced['action_label']

# X = X_bal
# y = y_bal




# 4) (Optional) Convert to NumPy for scikit‑learn
X = X_df.values     # array of shape (n_samples, 6)
y = y_ds.values        # array of shape (n_samples,)

# print("Feature matrix X:", X_df.head(), sep="\n")
# print("Labels y distribution:\n", y_ds.value_counts(), sep="")
# print("Class mapping:", class_map)


from sklearn.model_selection import train_test_split
from sklearn.tree       import DecisionTreeClassifier
from sklearn.ensemble   import AdaBoostClassifier
from sklearn.metrics    import classification_report

labels = [0,1,2,3]
names  = [ class_map[i] for i in labels ]  # ['fold','check','call','raise']



# 1) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_ds.values,
    stratify=y_ds,
    test_size=0.2,
    random_state=42
)



# 2) Build a decision‐stump base estimator
# base_stump = DecisionTreeClassifier(max_depth=1)
base_stump = DecisionTreeClassifier(max_depth=4)

# 3) Create AdaBoost ensemble
clf = AdaBoostClassifier(
    estimator=base_stump,
    n_estimators=100    ,
    learning_rate=1.0,
    algorithm="SAMME"
)

# 4) Train
clf.fit(X_train, y_train)

# 5) Predict & evaluate
y_pred = clf.predict(X_test)


print(classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=names
))
# print(classification_report(
#     y_test,
#     y_pred,
#     target_names=[class_map[i] for i in sorted(class_map)]
# ))
# raw_errors = []
# for t, est in enumerate(clf.estimators_):
#     y_pred = est.predict(X_train)
#     err    = np.mean(y_pred != y_train)   # fraction misclassified
#     raw_errors.append(err)
#     print(f"Estimator #{t:2d} unweighted error = {err:.4f}")

# # If you want the array:
# raw_errors = np.array(raw_errors)

'''
#%% This code does the metaanalysis

import optuna

from sklearn.model_selection import train_test_split
from sklearn.tree       import DecisionTreeClassifier
from sklearn.ensemble   import AdaBoostClassifier
from sklearn.metrics    import f1_score, classification_report
from sklearn.base       import clone

# ──────────────────────────────────────────────────────────────────────────────
# 1) Extract your Preflop dataset
# ──────────────────────────────────────────────────────────────────────────────
STAGE_FILTER = "PREFLOP"

rows, labels = [], []
for game in games_list:
    for sit in game.situations:
        # if sit.stage != STAGE_FILTER:
        if sit.stage == STAGE_FILTER:
            continue
        rows.append({
            "hand_score":   sit.hand_score,
            "board_score":  sit.score,
            "confirmed_in": sit.confirmed_in,
            "potential_in": sit.potential_in,
            "current_call": sit.current_call,
            "pot_size":     sit.pot_size,
        })
        if sit.action_raise > 0:
            labels.append(3)
        elif sit.action_call:
            labels.append(2)
        elif sit.action_check:
            labels.append(1)
        else:
            labels.append(0)

df = pd.DataFrame(rows)
df["action_label"] = labels

# ──────────────────────────────────────────────────────────────────────────────
# 2) Compute √(support) weights for evaluation
# ──────────────────────────────────────────────────────────────────────────────
counts = df["action_label"].value_counts().sort_index()  # idx 0,1,2,3
# sqrt_counts = counts.apply(math.sqrt)
# eval_weights = (sqrt_counts / sqrt_counts.sum()).to_dict()

eval_weights = (counts / counts.sum()).to_dict()

# e.g. {0: w0, 1: w1, 2: w2, 3: w3}

def weighted_f1(y_true, y_pred):
    """
    Weighted F1 where class c’s weight = eval_weights[c].
    """
    f1s = f1_score(y_true, y_pred, labels=[0,1,2,3], average=None)
    return sum(eval_weights[c] * f1s[c] for c in range(4))

# ──────────────────────────────────────────────────────────────────────────────
# 3) Train ∶ Val ∶ Test split (70 : 15 : 15)
# ──────────────────────────────────────────────────────────────────────────────
X = df.drop("action_label", axis=1).values
y = df["action_label"].values

# hold out test first (15%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# now split trainval into train (70%) and val (15%)
# since trainval is 85% of data, validation_size = 15/85 ≃ 0.17647
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.17647,
    stratify=y_trainval,
    random_state=42
)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Optuna objective for tuning
# ──────────────────────────────────────────────────────────────────────────────
def objective(trial):
    # discrete params
    n_estimators = trial.suggest_int("n_estimators", 75, 175)
    max_depth    = trial.suggest_int("max_depth",    3,   5)
    # continuous raw weights
    # w0 = trial.suggest_float("w0", 0.0, 1.0)
    # w1 = trial.suggest_float("w1", 0.0, 1.0)
    # w2 = trial.suggest_float("w2", 0.0, 1.0)
    # w3 = trial.suggest_float("w3", 0.0, 1.0)
    # raw = np.array([w0, w1, w2, w3])
    # total = raw.sum()
    # if total == 0:
    #     weights = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
    # else:
    #     norm = raw / total
    #     weights = {i: norm[i] for i in range(4)}
    
    
    # stick‑breaking for 4 weights
    # w0 = trial.suggest_float("w0", 0.02, 0.2)
    # w3 = trial.suggest_float("w3", 0.10, min(0.25,1-w0))
    # w2 = trial.suggest_float("w2", 0.15, min(0.3,1.0 - w0 - w3))
    # w1 = 1.0 - w0 - w3 - w2
    
    w0 = trial.suggest_float("w0", 0.05, 0.2)
    w1 = trial.suggest_float("w1", 0.10, min(0.25,1-w0))
    w2 = trial.suggest_float("w2", 0.15, min(0.35,1.0 - w0 - w1))
    w3 = 1.0 - w0 - w1 - w2
    
    # w0 = 0.170
    # w3 = 0.272
    # w2 = 0.309
    # w1 = 0.249
    
    
    # w0 = 0.176
    # w3 = 0.238
    # w2 = 0.293
    # w1 = 0.293
    
    # w0 = 0.172
    # w3 = 0.343
    # w2 = 0.249
    # w1 = 0.237
    
    weights = {0:w0, 1:w1, 2:w2, 3:w3}

    # SINGLE SHOT VERSION
    
    # # build model with sampled class_weight
    # base = DecisionTreeClassifier(
    #     max_depth=max_depth,
    #     class_weight=weights
    # )
    # clf = AdaBoostClassifier(
    #     estimator=base,
    #     n_estimators=n_estimators,
    #     learning_rate=1.0,
    #     algorithm="SAMME"
    # )

    # # train on train split
    # try:
    #     clf.fit(X_train, y_train)
    # except ValueError as e:
    #     msg = str(e)
    #     if "worse than random" in msg:
    #         # give this trial a terrible score and stop
    #         return 0.0
    #     else:
    #         # re‑raise anything unexpected
    #         raise
    # # predict on val split
    # y_pred = clf.predict(X_val)
    # # evaluate with our √‑weighted F1
    # return weighted_f1(y_val, y_pred)
    
    #MULTI SHOT VERSION
    
    # 3) Build your model template
    base = DecisionTreeClassifier(max_depth=max_depth, class_weight=weights)
    clf_template = AdaBoostClassifier(estimator=base,
                                      n_estimators=n_estimators,
                                      algorithm="SAMME")
    
    # 4) Repeat train/val 5× with different seeds, average the weighted‐F1
    seeds = [0, 1, 2, 3, 4]
    scores = []
    for seed in seeds:
        # a) re‐split train/val with a fresh random_state
        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_trainval, y_trainval,
            test_size=0.17647,
            stratify=y_trainval,
            random_state=seed
        )
        # b) clone & fit a fresh model
        clf = clone(clf_template)
        try:
            clf.fit(X_tr, y_tr)
        except ValueError as e:
            msg = str(e)
            if "worse than random" in msg:
                # give this trial a terrible score and stop
                return 0.0
            else:
                # re‑raise anything unexpected
                raise
        y_pred = clf.predict(X_vl)
        scores.append(weighted_f1(y_vl, y_pred))
    
    # 5) return the mean score
    return np.mean(scores)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Run Optuna
# ──────────────────────────────────────────────────────────────────────────────
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(">> Best hyperparameters:", study.best_trial.params)
print(">> Best weighted‑f1:",  study.best_value)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Retrain best model on train+val, test on hold‑out
# ──────────────────────────────────────────────────────────────────────────────
best = study.best_trial.params

# Since we only sampled w0,w1,w2, compute w3 as the remainder
# w0 = best["w0"]
# w3 = best["w3"]
# w2 = best["w2"]
# w1 = 1.0 - (w0 + w3 + w2)

w0 = best["w0"]
w1 = best["w1"]
w2 = best["w2"]
w3 = 1.0 - (w0 + w1 + w2)

# w0 = 0.170
# w3 = 0.272
# w2 = 0.309
# w1 = 0.249


# w0 = 0.176
# w3 = 0.238
# w2 = 0.293
# w1 = 0.293

# w0 = 0.172
# w3 = 0.343
# w2 = 0.249
# w1 = 0.237

# Build the class_weight dict directly—no further normalization needed
cw = {0: w0, 1: w1, 2: w2, 3: w3}

print(f"Final class weights: w0={w0:.3f}, w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}")

final_base = DecisionTreeClassifier(
    max_depth=best["max_depth"],
    class_weight=cw
)
final_clf = AdaBoostClassifier(
    estimator=final_base,
    n_estimators=best["n_estimators"],
    learning_rate=1.0,
    algorithm="SAMME"
)

# train on train+val
final_clf.fit(X_trainval, y_trainval)

# evaluate on test
y_test_pred = final_clf.predict(X_test)

print("\n-- Final Test Performance --")
print("Weighted‑F1 (√‑support):", weighted_f1(y_test, y_test_pred))
print("\nClassification report (unweighted):")
print(classification_report(
    y_test,
    y_test_pred,
    labels=[0,1,2,3],
    target_names=["fold","check","call","raise"]
))