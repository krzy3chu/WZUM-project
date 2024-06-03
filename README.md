# All-NBA Teams predictions
This repository implements the process of obtaining and evaluating a machine learning model to predict the players selected for the [All-NBA](https://www.nba.com/news/history-all-nba-teams) and [NBA All-Rookie](https://www.nba.com/news/history-all-rookie-teams) teams in a given season.

Project dependencies can be found in file *requirements.txt*. *All-NBA.ipynb* and *All-NBA-Rookies* Jupyter Notebooks are used to develop seperate models. General concepts and most of the code in both files are the same, but they are splitted between two notebooks for a more convenient workflow. Predict.py script allows you to predict teams in a given season based on trained models. It must be run with one argument, which specifies path to results file with predictions in json format.


## Machine learning workflow description

### Data preparation and analysis
Project uses players performance statistics from the [nba.com](https://www.nba.com/stats/leaders?SeasonType=Regular+Season&PerMode=Totals) website, which are accessed through a client [API](https://github.com/swar/nba_api). Statistics are fetched for each regular season separetly in a totals mode. Only 100 players who scored the most points in a given season are taken into account. Data from all specified seasons (1994-2024) is stacked up into one pandas DataFrame as in an example below. 

| SEASON   | PLAYER                  | TEAM   |   GP |   MIN |   PTS |   FGM |   FGA |   FG_PCT |   FG3M |   FG3A |   FG3_PCT |   REB |   EFF |
|:---------|:------------------------|:-------|-----:|------:|------:|------:|------:|---------:|-------:|-------:|----------:|------:|------:|
| 2023-24  | Luka Doncic             | DAL    |   70 |  2624 |  2370 |   804 |  1652 |    0.487 |    284 |    744 |     0.382 |   647 |  2580 |
| 2023-24  | Shai Gilgeous-Alexander | OKC    |   75 |  2553 |  2254 |   796 |  1487 |    0.535 |     95 |    269 |     0.353 |   415 |  2416 |
| 2023-24  | Giannis Antetokounmpo   | MIL    |   73 |  2567 |  2222 |   837 |  1369 |    0.611 |     34 |    124 |     0.274 |   841 |  2655 |
| 2023-24  | Jalen Brunson           | NYK    |   77 |  2726 |  2212 |   790 |  1648 |    0.479 |    211 |    526 |     0.401 |   278 |  1972 |
| 2023-24  | Nikola Jokic            | DEN    |   79 |  2737 |  2085 |   822 |  1411 |    0.583 |     83 |    231 |     0.359 |   976 |  3039 |
...
| 1994-95  | Terrell Brandon | CLE    |   67 |  1961 |   889 |   341 |   762 |    0.448 |     48 |    121 |     0.397 |   186 |   967 |
| 1994-95  | Bobby Phills    | CLE    |   80 |  2500 |   878 |   338 |   816 |    0.414 |     19 |     55 |     0.345 |   265 |   820 |
| 1994-95  | Spud Webb       | SAC    |   76 |  2458 |   878 |   302 |   689 |    0.438 |     48 |    145 |     0.331 |   174 |  1015 |

Informations about All-NBA teams picked in previous years are stored in the *targets.csv* file, where the first 15 names are players from three All-NBA teams, and the last 10 names are from two NBA All-Rookie teams. Data from this file is used to mark players in DataFrame with corresponding values (3 for first team, 2 for second, 1 for third and similarly for rookies). Vector of this data will be used then in machine learning process as target. Gathered data can be analyzed in an auxiliary dataframe, which may allow to detect any gaps in the data.

In next step the data is spilt between train and test dataframes based on seasons specified in a list (every fith year for default). Then the entire available set of features is passed to the pipeline. The figure below shows a graphical distribution of the normalized values ​​of two sample features (efficiency and points), with All-NBA teams players marked with different colors.

![alt](./assets/efficiency_points_distribution.svg)