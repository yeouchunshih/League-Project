import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
#input stuff
league_data=pd.read_csv('chaincc-players-2026.csv')
temp_data=pd.read_csv('2026_LoL_esports_match_data_from_OraclesElixir.csv')
league_data=league_data.assign(teams=league_data.apply(lambda row: " ".join(sorted([row.get('team_name'), row.get('opponent_team_name')])), axis=1))

#merge the dataframes on common columns
common_columns = [
    column for column in league_data.columns.intersection(temp_data.columns)
    if column not in ['game_id', 'side', 'position', 'playername', 'champion']]


temp_data.drop(columns=common_columns, inplace=True)
temp_data = temp_data.rename(columns={'gameid': 'game_id'})

league_data = league_data.merge(temp_data,on=['game_id', 'side', 'position', 'playername', 'champion'],how='inner')
print(league_data.shape[0])

league_data = league_data.assign(
    Champion_with_position=(league_data.get('position') + ': ' + league_data.get('champion')))

player_games = (
    league_data[['playername', 'game_id']]
    .drop_duplicates()
    .sort_values(['playername', 'game_id']))
player_games['game_number'] = player_games.groupby('playername').cumcount() + 1
league_data = league_data.merge(
    player_games,
    on=['playername', 'game_id'],
    how='left')
#print(league_data.columns)
league_data['game_id_team'] = league_data.get('game_id') + '_' + league_data.get('team_name')
#input champion and outputs its winrate, pickrate, and most common position
champion = input('Enter a champion: ')
champion=champion.title()
champion_data=league_data[league_data.get('champion')==champion]
champion_wins=np.count_nonzero(champion_data.get('result'))

champion_position=champion_data.groupby('position').size().idxmax()
#this assigns a position with each champion in the game
def expand_position(position):
    if position == 'top':
        return 'Top Lane'
    elif position == 'jng':
        return 'Jungle'
    elif position == 'mid':
        return 'Mid Lane'
    elif position == 'bot':
        return 'Bot Lane'
    elif position == 'sup':
        return 'Support'
    else:
        return 'Unknown'
#trying to get all games where the champion was present then take the champions that are with it

champion_game_ids = champion_data.get('game_id_team')
champion_present=league_data[league_data.get('game_id_team').isin(champion_game_ids)]
wins_of_champions_in_game=champion_present.get(['champion','result','position'])
wins_of_champions_in_game = wins_of_champions_in_game[wins_of_champions_in_game.get('champion') != champion ]
#all of the laners winrates
top_lane_games = wins_of_champions_in_game[
    wins_of_champions_in_game.get('position') == 'top'].copy()
top_lane_games['wins'] = top_lane_games.get('result')==1
winrate_top_lane = (
    top_lane_games.groupby('champion', as_index=False)
    .agg(games=('champion', 'size'), wins=('wins', 'sum')))
winrate_top_lane['winrate'] = winrate_top_lane['wins'] / winrate_top_lane['games'] * 100
mid_lane_games = wins_of_champions_in_game[
    wins_of_champions_in_game.get('position') == 'mid'].copy()
mid_lane_games['wins'] = mid_lane_games.get('result')==1
winrate_mid_lane = (
    mid_lane_games.groupby('champion', as_index=False)
    .agg(games=('champion', 'size'), wins=('wins', 'sum')))
winrate_mid_lane['winrate'] = winrate_mid_lane['wins'] / winrate_mid_lane['games'] * 100
jng_lane_games = wins_of_champions_in_game[
    wins_of_champions_in_game.get('position') == 'jng'].copy()
jng_lane_games['wins'] = jng_lane_games.get('result')==1
winrate_jng_lane = (
    jng_lane_games.groupby('champion', as_index=False)
    .agg(games=('champion', 'size'), wins=('wins', 'sum')))
winrate_jng_lane['winrate'] = winrate_jng_lane['wins'] / winrate_jng_lane['games'] * 100
bot_lane_games = wins_of_champions_in_game[
    wins_of_champions_in_game.get('position') == 'bot'].copy()
bot_lane_games['wins'] = bot_lane_games.get('result')==1
winrate_bot_lane = (
    bot_lane_games.groupby('champion', as_index=False)
    .agg(games=('champion', 'size'), wins=('wins', 'sum')))
winrate_bot_lane['winrate'] = winrate_bot_lane['wins'] / winrate_bot_lane['games'] * 100
sup_lane_games = wins_of_champions_in_game[
    wins_of_champions_in_game.get('position') == 'sup'].copy()
sup_lane_games['wins'] = sup_lane_games.get('result')==1
winrate_sup_lane = (
    sup_lane_games.groupby('champion', as_index=False)
    .agg(games=('champion', 'size'), wins=('wins', 'sum')))
winrate_sup_lane['winrate'] = winrate_sup_lane['wins'] / winrate_sup_lane['games'] * 100
#possible_pickrate_games['not available']

game_count=league_data.copy().sort_values(by=['date', 'playername'])
game_count['day']=game_count.get('date').str.split(' ').str[0]
game_count=game_count.assign(day_teams=game_count.apply(lambda row: " ".join(sorted([row.get('day'), row.get('teams')])), axis=1))
game_count['game_count']= game_count.groupby(['day_teams','playername']).cumcount()+1


#unavailable champions for bans
#get the game_counts that are of the same day_team and greater, then extract the champion value of each of the rows and append it to a list and then put that into a new column; from there 
def mark_unavailable_champions(game_df):
    df=game_df.copy()
    grouped_lists=(
        df.groupby(['day_teams', 'game_count'])['champion'].apply(list).reset_index(name='champions_in_game')
    )
    df = df.merge(grouped_lists, on=['day_teams', 'game_count'], how='left')
    return df
#print(mark_unavailable_champions(game_count))
unavailable_champions= mark_unavailable_champions(game_count)
champions_by_game = unavailable_champions.groupby(['day_teams', 'game_count'])['champions_in_game'].first()
unavailable_champions['banned_out'] = unavailable_champions.apply(lambda row: [row['ban1'], row['ban2'], row['ban3'], row['ban4'], row['ban5']], axis=1)

def get_unavailable_champions(game_df):
    unavailable_champions_list = []
    for previous_game_count in range(1, game_df.get('game_count')):
        champions = champions_by_game.get((game_df.get('day_teams'), previous_game_count), [])
        unavailable_champions_list.extend(champions)
        
    return unavailable_champions_list
unavailable_champions['unavailable_champions'] = unavailable_champions.apply(get_unavailable_champions, axis=1)
unavailable_champions['unavailable_in_game'] = (unavailable_champions['unavailable_champions'] + unavailable_champions['banned_out'])
champion_available = unavailable_champions[unavailable_champions['unavailable_in_game'].apply(lambda bans: champion not in bans)]
compressed_games = champion_available.drop_duplicates('game_id')
champion_picks = champion_available[
    champion_available['champion'] == champion
]['game_id'].nunique()
print(champion_available.head(30))
print(f"Winrate: {champion_wins/champion_data.shape[0]*100}%")
print(f"Pickrate: {champion_picks/compressed_games.shape[0]*100}%")

#get pickrate of the champions played with another champion


#print(f"Winrate of support champions in games with {champion}: ")
#print(winrate_sup_lane.sort_values('winrate', ascending=False))

#print(f"Winrate of bot lane champions in games with {champion}: ")
#print(winrate_bot_lane.sort_values('winrate', ascending=False))

#print(f"Winrate of jungle champions in games with {champion}: ")
#print(winrate_jng_lane.sort_values('winrate', ascending=False))

#print(f"Winrate of mid lane champions in games with {champion}: ")
#print(winrate_mid_lane.sort_values('winrate', ascending=False))

#print(f"Winrate of top lane champions in games with {champion}: ")
#print(winrate_top_lane.sort_values('winrate', ascending=False))
#print(wins_of_champions_in_game)
#print(champion_game_ids)
#print(f"Total games for {champion}: {champion_game_count}")
#print(champion_data)
#print(f"Most common position: {expand_position(champion_position)}")
