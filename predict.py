import argparse
from json import dump
from pathlib import Path
from pickle import load

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leagueleaders


def all_nba_predict(rookies = False):

    if rookies:
        scope = "Rookies"
        model_file = 'All-NBA-Rookies_model.pkl'
        no_teams = 2
    else:
        scope = "S"
        model_file = 'All-NBA_model.pkl'
        no_teams = 3

    # get data from nba.com
    leaders = leagueleaders.LeagueLeaders(per_mode48="Totals", 
                                          scope=scope, 
                                          season="2023-24",    
                                          season_type_all_star="Regular Season")
    leaders_df = leaders.get_data_frames()[0]

    # prepare data with all available stats
    features = ['GP', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
                'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'EFF']
    data = leaders_df[features]

    # load trained model
    with open(model_file, 'rb') as file:
        model = load(file)

    # make predictions
    predictions = model.predict(data)

    # process and return results
    prediction_idxs = np.argsort(predictions)[::-1]
    predictions_names = leaders_df.loc[prediction_idxs[:no_teams * 5], 'PLAYER']

    return predictions_names.to_numpy().reshape(no_teams, 5).tolist()


def main():
    # parse path to results file
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()
    results_file = Path(args.results_file)

    # predict all-nba teams
    predictions_all_nba = all_nba_predict()
    predictions_all_nba_rookies = all_nba_predict(rookies=True)

    # merge predictions into dictionary
    predictions_dict = {}
    teams = ['first', 'second', 'third', 'first rookie', 'second rookie']
    for team, prediction in zip(teams, predictions_all_nba + predictions_all_nba_rookies):
        predictions_dict[team + " all-nba team"] = prediction

    # save results to json
    with open(results_file, 'w') as file:
        dump(predictions_dict, file)


if __name__ == '__main__':
    main()