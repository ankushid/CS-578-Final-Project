# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:24:27 2025

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
number of players still in
number of players yet to play
current call
delta between current call and start call for round
pot size

actions (y E [fold, check/call, raise])     Combining call/check bc they both mean "stay in". in future maybe separate, maybe try to predict raise size


other info to calculate for a situation
delta_loss/win
In games where player stays in until end, should/should_not have folded




cycle through games. extract situations from games
'''

import pandas as pd

game_data = pd.read_csv("game_data.csv")
actions = pd.read_csv("actions.csv")

#%%

games_to_examine = range(1000)




def get_first_last_action_indices(actions_df: pd.DataFrame):
    """
    Given an actions DataFrame, returns two lists:
      - first_indices: the index of the first action for each game.
      - last_indices: the index of the last action for each game.
      
    The grouping is done by the 'gameId' column.
    """
    # Group the DataFrame by 'gameId'; using sort=False preserves row order.
    grouped = actions_df.groupby("gameId", sort=False)
    
    first_indices = []
    last_indices = []
    
    # For each game, record the first and last row index from the group.
    for game_id, group in grouped:
        first_index = group.index[0]
        last_index = group.index[-1]
        first_indices.append(first_index)
        last_indices.append(last_index)
    
    return first_indices, last_indices


firsts, lasts = get_first_last_action_indices(actions)

def remove_duplicate_action_blocks(actions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate contiguous blocks of actions for each game id.
    
    Given that many duplicate "stretches" exist in the actions DataFrame,
    this function scans through the DataFrame (assumed to be ordered) and,
    for each game id, only keeps the first contiguous block of rows.
    All additional blocks (i.e. duplicate occurrences of the game id) are dropped.
    
    Parameters:
        actions_df (pd.DataFrame): Original actions data with duplicates.
          Assumes an ordering (e.g., by time) and that rows sharing the same 
          gameId in a contiguous block represent one complete stretch.
    
    Returns:
        pd.DataFrame: A duplicate-free DataFrame for actions.
    """
    # Reset index to ensure a clean, sequential ordering.
    actions_df = actions_df.reset_index(drop=True)
    
    # Boolean mask to keep rows, initially assume every row is kept.
    keep_mask = [True] * len(actions_df)
    seen_game_ids = set()
    
    i = 0
    while i < len(actions_df):
        # Determine the game id for the current block.
        current_game_id = actions_df.loc[i, 'gameId']
        
        # A new block is identified if it's the first row or if the gameId
        # differs from the previous row.
        if i == 0 or actions_df.loc[i, 'gameId'] != actions_df.loc[i-1, 'gameId']:
            if current_game_id in seen_game_ids:
                # Duplicate block – mark entire contiguous block of this gameId for removal.
                j = i
                while j < len(actions_df) and actions_df.loc[j, 'gameId'] == current_game_id:
                    keep_mask[j] = False
                    j += 1
                i = j  # Move to the next block
                continue
            else:
                # This is the first time we see this game id.
                seen_game_ids.add(current_game_id)
                # Mark all rows in this contiguous block as kept.
                j = i
                while j < len(actions_df) and actions_df.loc[j, 'gameId'] == current_game_id:
                    keep_mask[j] = True
                    j += 1
                i = j
        else:
            # If not a block boundary, just move to the next row.
            i += 1

    # Return only the rows marked to keep, and reset the index.
    return actions_df[keep_mask].reset_index(drop=True)


def remove_duplicate_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate game records from the games DataFrame.
    
    Since each game duplicate occupies one index, this function drops duplicate
    rows based on the 'gameId' column.
    
    Parameters:
        games_df (pd.DataFrame): Original games DataFrame with potential duplicates.
    
    Returns:
        pd.DataFrame: The duplicate-free games DataFrame.
    """
    return games_df.drop_duplicates(subset=['gameId']).reset_index(drop=True)


# ------------------
# Example usage
# ------------------



# Clean the actions DataFrame.
cleaned_actions = remove_duplicate_action_blocks(actions)
cleaned_actions.to_csv("cleaned_actions.csv")
print("Cleaned Actions DataFrame:")
print(cleaned_actions)



# Clean the games DataFrame.
cleaned_game_data = remove_duplicate_games(game_data)
cleaned_game_data.to_csv("cleaned_game_data.csv")
print("\nCleaned Games DataFrame:")
print(cleaned_game_data)

#%%

player_stats = pd.read_csv("player_stats.csv")

def remove_duplicate_player_stats(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the player_stats DataFrame.
    
    For each unique combination of gameId and player name, only the first occurrence is kept.
    
    Parameters:
        player_stats_df (pd.DataFrame): The original DataFrame containing the player statistics.
    
    Returns:
        pd.DataFrame: The duplicate-free DataFrame.
    """
    # Drop duplicate rows based on gameId and name combination,
    # keeping the first occurrence.
    cleaned_df = player_stats_df.drop_duplicates(subset=["gameId", "name"]).reset_index(drop=True)
    return cleaned_df

cleaned_player_stats = remove_duplicate_player_stats(player_stats)
cleaned_player_stats.to_csv("cleaned_player_stats.csv")
print("\nPlayer Stats DataFrame:")
print(cleaned_player_stats)