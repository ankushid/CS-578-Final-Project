import pandas as pd
import os
import re
from datetime import datetime
import json
import glob

def GetGamesData():
    result = []
    
    for dirname, _, filenames in os.walk('./data'):
        for filename in filenames:
            print(filename)
            file = open(os.path.join(dirname, filename), 'r')
            content = file.read()

            result += re.findall(r'Game started at: (.+)\nGame ID: (\d+).+\(Hold\'em\)\nSeat (\d+) is the button\n((?:Seat .+\n)+)([^$]+?)------ Summary ------\n(Pot: .+?)\n(?:(Board: .+?)\n)?([^$]+?)Game ended at: (.+)', content)

            file.close()
    
    print ("Total of", len(result), "games")
    
    return (result)

games = GetGamesData()

datetimeFormat = '%Y/%m/%d %H:%M:%S'
seatRe = re.compile(r"Seat (\d+): (.+) \((\d+(?:\.\d+)?)\)")
betRe = re.compile(r"Pot: (\d+(?:\.\d+)?).+Rake (\d+(?:\.\d+)?)(?:.+JP fee (\d+(?:\.\d+)?))?")
boardRe = re.compile(r"\[([^]]*)\]")
playerNameRe = re.compile(r"Player ([^ ]+)")
actionRe = re.compile(r"Player .+ ([^ ]+)s? \((\d+(?:\.\d+)?)\)")
endingNumberRe = re.compile(r".+\((\d+(?:\.\d+)?)\)")
cardsRe = re.compile(r".*\[([^\]]+)\]")
winningRe = re.compile(r"Player .+?(?: \[(.+)\])?\)?\. ?Bets: (\d+(?:\.\d+)?)\. Collects: (\d+(?:\.\d+)?)\.(?: Wins: (\d+(?:\.\d+)?)\.)?(?: Loses: (\d+(?:\.\d+)?)\.)?")

def GetPlayerName(line):
    match = playerNameRe.search(line)
    
    if match is None:
        return ""
    
    return match.group(1)

def GetDuration(startTime, endTime):
    tStart = datetime.strptime(startTime, datetimeFormat)
    tEnd = datetime.strptime(endTime, datetimeFormat)
    tDelta = tEnd - tStart
    return int(tDelta.total_seconds())

def GetAsFloat(value):
    return float(value if value else 0)

def ClearDataframes():
    global gameDf
    gameDf = pd.DataFrame({'gameId': [], 'duration': [], 'button': [], 'seats': [], 'players': [], 'endState': [], 'pot': [], 'rake': [], 'fee': [], 'board': []})
    
    global playerGameStatsDf
    playerGameStatsDf = pd.DataFrame({'name': [], 'cards': [], 'bet': [], 'collect': [], 'win': [], 'lose': [], 'gameId': []})
    
    global gameActionsDf
    gameActionsDf = pd.DataFrame({'state': [], 'name': [], 'seat': [], 'action': [], 'value': [], 'gameId': []})
    
tmpPath = 'tmp'
gameDataframeFileName = 'game_data.csv'
gameActionsDataframeFileName = 'actions.csv'
playerGameStatsDataframeFileName = 'player_stats.csv'

def CleanEverything():
    for file in glob.glob(f'*.csv'):
        os.remove(file)

def PrepareTmpFolder(delete = False):
    ClearDataframes()
    
    if not os.path.exists(tmpPath):
        os.makedirs(tmpPath)
    elif delete:
        files = os.listdir(tmpPath)
        for file in files:
            file_path = os.path.join(tmpPath, file)
            os.remove(file_path)

def SaveTmpDatasets(index):
    currentIndex = str(index).rjust(5, '0')
    gameDf.to_csv(f'{tmpPath}/{currentIndex}-{gameDataframeFileName}', index=False)
    gameActionsDf.to_csv(f'{tmpPath}/{currentIndex}-{gameActionsDataframeFileName}', index=False)
    playerGameStatsDf.to_csv(f'{tmpPath}/{currentIndex}-{playerGameStatsDataframeFileName}', index=False)
    
def GetCurrentPlayer(playerDict, name):
    if name in playerDict:
        return playerDict[name]

    # Beware: Hack to handle space in username. Hopefully this is enough.
    playerNames = list(playerDict.keys())

    for playerName in playerNames:
        if playerName.startswith(name):
            return playerDict[playerName]


def ExtractGameData(currentGame):
    startTime = currentGame[0]
    gameId = int(currentGame[1])
    button = int(currentGame[2])
    seats = currentGame[3]
    gameActions = currentGame[4]
    bet = currentGame[5]
    board = currentGame[6]
    winnings = currentGame[7]
    endTime = currentGame[8]


    # Extract players
    players = seatRe.findall(seats)
    
    playerDict = {}
    seatsInPlay = []
    
    for player in players:
        playerSeat = int(player[0])
        playerName = player[1]
        playerChips = float(player[2])

        seatsInPlay += [playerSeat]
        playerDict[playerName] = {'seat': playerSeat, 'chips': playerChips, 'name': playerName, 'cards': []}


    # Extract actions
    actionsList = gameActions.split("\n")
    currentState = "PREFLOP"
    
    for action in actionsList:
        if action == "":
            continue
        if action.startswith("*** "):
            stateGroup = re.search(r'\*{3} ([^ ]+) \*{3}', action)
            currentState = stateGroup.group(1)
            continue
        if action.startswith("Uncalled bet"):
            continue
        
        currentPlayerName = GetPlayerName(action)
        currentPlayer = GetCurrentPlayer(playerDict, currentPlayerName)
        
        if currentPlayer is None:
            # game 30837 contains lines where the player name is missing
            continue

        if "has small blind" in action:
            smallBlindValue = float(endingNumberRe.search(action).group(1))
            gameActionsDf.loc[len(gameActionsDf)] = [currentState, currentPlayer['name'], currentPlayer["seat"], 'SB', smallBlindValue, gameId]
            continue
        if "has big blind" in action:
            bigBlindValue = float(endingNumberRe.search(action).group(1))
            gameActionsDf.loc[len(gameActionsDf)] = [currentState, currentPlayer['name'], currentPlayer["seat"], 'BB', bigBlindValue, gameId]
            continue
        if action.endswith(" received a card."):
            continue
        if " received card: [" in action:
            playerDict[playerName]["cards"] += cardsRe.search(action).group(1).split(" ")
            continue

        if action.endswith("is timed out."):
            if (currentPlayer):
                gameActionsDf.loc[len(gameActionsDf)] = [currentState, currentPlayer['name'], currentPlayer["seat"], 'timed out', 0, gameId]
            continue
        if action.endswith("sitting out"):
            gameActionsDf.loc[len(gameActionsDf)] = [currentState, currentPlayer['name'], currentPlayer["seat"], 'sitting out', 0, gameId]
            continue
        if action.endswith("wait BB"):
            gameActionsDf.loc[len(gameActionsDf)] = [currentState, currentPlayer['name'], currentPlayer["seat"], 'wait BB', 0, gameId]
            continue
        if action.endswith("mucks cards"):
            gameActionsDf.loc[len(gameActionsDf)] = [currentState, currentPlayer['name'], currentPlayer["seat"], 'muck', 0, gameId]
            continue
        if action.endswith("folds"):
            gameActionsDf.loc[len(gameActionsDf)] = [currentState, currentPlayer['name'], currentPlayer["seat"], 'fold', 0, gameId]
            continue
        if action.endswith("checks"):
            gameActionsDf.loc[len(gameActionsDf)] = [currentState, currentPlayer['name'], currentPlayer["seat"], 'check', 0, gameId]
            continue

        actionMatch = actionRe.match(action)
        gameActionsDf.loc[len(gameActionsDf)] = [currentState, currentPlayer['name'], currentPlayer["seat"], actionMatch.group(1), float(actionMatch.group(2)), gameId]


    # Extract bet values
    betGroups = betRe.search(bet).groups()
    gamePot = GetAsFloat(betGroups[0])
    gameRake = GetAsFloat(betGroups[1])
    gameFee = GetAsFloat(betGroups[2])


    # Extract board values
    boardCards = []
    if board:
        boardMatch = boardRe.search(board)
        boardCards = boardMatch.group(1).split(" ")


    # Extract winnings
    for winning in re.sub("\*", "", winnings).split("\n"):
        if winning == "":
            continue

        winningMatches = winningRe.search(winning)
        winningCards = winningMatches.group(1)

        currentPlayerName = GetPlayerName(winning)
        currentPlayer = GetCurrentPlayer(playerDict, currentPlayerName)
        
        playerGameStatsDf.loc[len(playerGameStatsDf)] = [currentPlayer['name'], winningCards.split(" ") if winningCards else None, GetAsFloat(winningMatches.group(2)), GetAsFloat(winningMatches.group(3)), GetAsFloat(winningMatches.group(4)), GetAsFloat(winningMatches.group(5)), gameId]


    # Extract duration
    duration = GetDuration(startTime, endTime)
    

    # Save result
    playersJson = json.dumps(list(playerDict.values()))
    seatsJson = json.dumps(seatsInPlay)
    boardJson = json.dumps(boardCards)
    gameDf.loc[len(gameDf)] = [gameId, duration, button, seatsJson, playersJson, currentState, gamePot, gameRake, gameFee, boardJson]
    
PrepareTmpFolder(True)
totalCount = len(games)

for index, currentGame in enumerate(games):
    ExtractGameData(currentGame)

    if index % 1000 == 0 and index > 0:
        SaveTmpDatasets(int(index / 1000))
        print (f'{index}/{totalCount} done')

print ('All done')

tmpFileNames = [gameDataframeFileName, gameActionsDataframeFileName, playerGameStatsDataframeFileName]

for tmpFileName in tmpFileNames:
    joined_files = os.path.join(tmpPath, f'*-{tmpFileName}')
    joined_list = sorted(glob.glob(joined_files))

    df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
    df.to_csv(tmpFileName, index=False)
    
    # Delete temporary data
if os.path.exists('/kaggle/working/tmp'):
    files = os.listdir('/kaggle/working/tmp')
    for file in files:
        file_path = os.path.join('/kaggle/working/tmp', file)
        os.remove(file_path)
    os.rmdir('/kaggle/working/tmp')