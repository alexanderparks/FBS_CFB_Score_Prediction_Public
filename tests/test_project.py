import unittest

import numpy as np
import pandas as pd

from src.data import _home_margin_from_line, normalize_team_stats
from src.features import (
    build_modeling_table,
    build_team_games,
    check_for_leakage,
    get_model_features,
)
from src.models import candidate_models


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "season": 2024,
                    "week": 1,
                    "season_type": "regular",
                    "start_date": pd.Timestamp("2024-09-01", tz="UTC"),
                    "completed": True,
                    "neutral_site": False,
                    "conference_game": False,
                    "home_team": "Alpha",
                    "away_team": "Beta",
                    "home_points": 30,
                    "away_points": 20,
                    "home_classification": "fbs",
                    "away_classification": "fbs",
                },
                {
                    "game_id": 2,
                    "season": 2024,
                    "week": 2,
                    "season_type": "regular",
                    "start_date": pd.Timestamp("2024-09-08", tz="UTC"),
                    "completed": True,
                    "neutral_site": False,
                    "conference_game": True,
                    "home_team": "Beta",
                    "away_team": "Alpha",
                    "home_points": 21,
                    "away_points": 24,
                    "home_classification": "fbs",
                    "away_classification": "fbs",
                },
                {
                    "game_id": 3,
                    "season": 2024,
                    "week": 3,
                    "season_type": "regular",
                    "start_date": pd.Timestamp("2024-09-15", tz="UTC"),
                    "completed": False,
                    "neutral_site": True,
                    "conference_game": False,
                    "home_team": "Alpha",
                    "away_team": "Beta",
                    "home_points": np.nan,
                    "away_points": np.nan,
                    "home_classification": "fbs",
                    "away_classification": "fbs",
                },
            ]
        )
        rows = []
        for game in self.games.iloc[:2].itertuples(index=False):
            for team, points, home_away in [
                (game.home_team, game.home_points, "home"),
                (game.away_team, game.away_points, "away"),
            ]:
                rows.append(
                    {
                        "game_id": game.game_id,
                        "team": team,
                        "home_away": home_away,
                        "points": points,
                        "total_yards": 400 + points,
                        "total_offensive_plays": 70,
                        "turnovers": 1,
                    }
                )
        self.team_stats = pd.DataFrame(rows)
        self.advanced = pd.DataFrame(
            [
                {"game_id": row["game_id"], "team": row["team"], "offense_ppa": 0.2, "defense_ppa": 0.1}
                for row in rows
            ]
        )

    def test_formatted_line_is_converted_to_home_margin(self):
        home_favorite = {"formattedSpread": "Alpha -7.5"}
        away_favorite = {"formattedSpread": "Beta -3"}
        self.assertEqual(_home_margin_from_line(home_favorite, "Alpha", "Beta"), 7.5)
        self.assertEqual(_home_margin_from_line(away_favorite, "Alpha", "Beta"), -3.0)

    def test_pregame_snapshots_do_not_use_target_game(self):
        team_games = build_team_games(self.games, self.team_stats, self.advanced)
        table = build_modeling_table(self.games, team_games)
        self.assertTrue(check_for_leakage(table))

        week_two = table.loc[table["game_id"].eq(2)].iloc[0]
        self.assertEqual(week_two["away_points_for_last_5"], 30)
        self.assertEqual(week_two["home_points_for_last_5"], 20)

    def test_model_features_exclude_targets_and_market(self):
        team_games = build_team_games(self.games, self.team_stats, self.advanced)
        table = build_modeling_table(self.games, team_games)
        features = get_model_features(table)
        forbidden = {
            "home_points",
            "away_points",
            "actual_margin",
            "actual_total",
            "market_margin",
            "market_total",
            "attendance",
        }
        self.assertTrue(forbidden.isdisjoint(features))
        self.assertIn("elo_diff", features)

    def test_all_six_model_families_are_included(self):
        expected = {
            "Ridge",
            "Elastic Net",
            "Random Forest",
            "Gradient Boosting",
            "RBF Support Vector Regression",
            "Neural Network",
        }
        self.assertEqual(set(candidate_models(fast_mode=True)), expected)


if __name__ == "__main__":
    unittest.main()
