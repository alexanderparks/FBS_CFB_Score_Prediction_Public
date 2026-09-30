import numpy as np
import pandas as pd


ROLLING_SOURCE_COLUMNS = [
    "points_for",
    "points_against",
    "total_yards",
    "rushing_yards",
    "net_passing_yards",
    "total_offensive_plays",
    "yards_per_play",
    "turnovers",
    "third_down_pct",
    "fourth_down_pct",
    "possession_seconds",
    "offense_ppa",
    "defense_ppa",
    "offense_success_rate",
    "defense_success_rate",
    "offense_explosiveness",
    "defense_explosiveness",
    "offense_points_per_opportunity",
    "defense_points_per_opportunity",
]


def add_derived_box_score_features(team_stats):
    team_stats = team_stats.copy()
    if {"total_yards", "total_offensive_plays"}.issubset(team_stats.columns):
        team_stats["yards_per_play"] = team_stats["total_yards"] / team_stats[
            "total_offensive_plays"
        ].replace(0, np.nan)
    for prefix in ["third_down", "fourth_down"]:
        made = f"{prefix}_conversions"
        attempts = f"{prefix}_attempts"
        if {made, attempts}.issubset(team_stats.columns):
            team_stats[f"{prefix}_pct"] = team_stats[made] / team_stats[attempts].replace(
                0, np.nan
            )
    return team_stats


def build_team_games(games, team_stats, advanced_stats):
    """Create one completed-game record for each team."""
    completed = games.loc[
        games["completed"].eq(True)
        & games["home_points"].notna()
        & games["away_points"].notna()
    ].copy()

    home = completed[
        ["game_id", "season", "week", "season_type", "start_date", "home_team", "away_team", "home_points", "away_points"]
    ].rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_points": "points_for",
            "away_points": "points_against",
        }
    )
    home["is_home"] = 1

    away = completed[
        ["game_id", "season", "week", "season_type", "start_date", "home_team", "away_team", "home_points", "away_points"]
    ].rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_points": "points_for",
            "home_points": "points_against",
        }
    )
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True)
    team_stats = add_derived_box_score_features(team_stats)
    duplicate_columns = [
        column for column in ["points", "conference", "home_away", "team_id"] if column in team_stats.columns
    ]
    team_games = team_games.merge(
        team_stats.drop(columns=duplicate_columns, errors="ignore"),
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )
    team_games = team_games.merge(
        advanced_stats.drop(columns=["season", "week", "opponent"], errors="ignore"),
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )
    return team_games.sort_values(["start_date", "game_id", "is_home"]).reset_index(drop=True)


def create_team_snapshots(team_games, rolling_window=5, ewm_span=5):
    """Summarize each team's information immediately after each completed game."""
    snapshots = team_games.sort_values(["team", "start_date", "game_id"]).copy()
    stat_columns = [column for column in ROLLING_SOURCE_COLUMNS if column in snapshots.columns]

    snapshots["games_played"] = snapshots.groupby("team").cumcount() + 1
    snapshots["season_games_played"] = snapshots.groupby(["team", "season"]).cumcount() + 1

    for column in stat_columns:
        grouped = snapshots.groupby("team", group_keys=False)[column]
        snapshots[f"{column}_last_{rolling_window}"] = grouped.transform(
            lambda values: values.rolling(rolling_window, min_periods=1).mean()
        )
        snapshots[f"{column}_ewm"] = grouped.transform(
            lambda values: values.ewm(span=ewm_span, adjust=False, min_periods=1).mean()
        )
        snapshots[f"{column}_season"] = snapshots.groupby(
            ["team", "season"], group_keys=False
        )[column].transform(lambda values: values.expanding(min_periods=1).mean())

    feature_columns = [
        column
        for column in snapshots.columns
        if column.endswith(f"_last_{rolling_window}")
        or column.endswith("_ewm")
        or column.endswith("_season")
    ]
    keep = ["team", "start_date", "game_id", "games_played", "season_games_played"] + feature_columns
    return snapshots[keep].sort_values(["team", "start_date", "game_id"])


def _attach_previous_snapshot(games, snapshots, team_column, prefix):
    """Attach the latest strictly earlier snapshot for the home or away team."""
    pieces = []
    snapshot_features = [column for column in snapshots.columns if column not in ["team", "start_date", "game_id"]]
    for team, game_group in games.groupby(team_column, sort=False):
        left = game_group.sort_values("start_date").copy()
        right = snapshots.loc[snapshots["team"].eq(team)].drop(columns="team").sort_values("start_date").copy()
        if right.empty:
            for column in snapshot_features:
                left[f"{prefix}_{column}"] = np.nan
            left[f"{prefix}_last_game_date"] = pd.NaT
        else:
            right = right.rename(
                columns={
                    "start_date": f"{prefix}_last_game_date",
                    "game_id": f"{prefix}_previous_game_id",
                    **{column: f"{prefix}_{column}" for column in snapshot_features},
                }
            )
            left = pd.merge_asof(
                left.sort_values("start_date"),
                right.sort_values(f"{prefix}_last_game_date"),
                left_on="start_date",
                right_on=f"{prefix}_last_game_date",
                direction="backward",
                allow_exact_matches=False,
            )
        pieces.append(left)
    return pd.concat(pieces, ignore_index=True).sort_values(["start_date", "game_id"])


def calculate_pregame_elo(games, initial_rating=1500.0, k_factor=25.0, home_advantage=55.0):
    """Calculate Elo sequentially, updating ratings only after completed games."""
    ordered = games.sort_values(["start_date", "game_id"]).copy()
    ratings = {}
    last_season = None
    rows = []

    for game in ordered.itertuples(index=False):
        if last_season is not None and game.season != last_season:
            ratings = {team: 1500 + 0.75 * (rating - 1500) for team, rating in ratings.items()}
        last_season = game.season

        home_rating = ratings.get(game.home_team, initial_rating)
        away_rating = ratings.get(game.away_team, initial_rating)
        advantage = 0.0 if bool(game.neutral_site) else home_advantage
        expected_home = 1 / (1 + 10 ** (-(home_rating + advantage - away_rating) / 400))
        rows.append(
            {
                "game_id": game.game_id,
                "home_elo": home_rating,
                "away_elo": away_rating,
                "elo_home_win_probability": expected_home,
            }
        )

        if bool(game.completed) and pd.notna(game.home_points) and pd.notna(game.away_points):
            actual_home = 1.0 if game.home_points > game.away_points else 0.0
            if game.home_points == game.away_points:
                actual_home = 0.5
            margin = abs(float(game.home_points) - float(game.away_points))
            multiplier = np.log1p(margin) * (2.2 / ((home_rating - away_rating) * 0.001 + 2.2))
            change = k_factor * multiplier * (actual_home - expected_home)
            ratings[game.home_team] = home_rating + change
            ratings[game.away_team] = away_rating - change

    return pd.DataFrame(rows)


def build_modeling_table(games, team_games, talent=None, recruiting=None, lines=None):
    """Create one row per game using only information available before kickoff."""
    fbs_games = games.loc[
        games["home_classification"].eq("fbs") & games["away_classification"].eq("fbs")
    ].copy()
    fbs_games = fbs_games.sort_values(["start_date", "game_id"]).reset_index(drop=True)
    snapshots = create_team_snapshots(team_games)

    table = _attach_previous_snapshot(fbs_games, snapshots, "home_team", "home")
    table = _attach_previous_snapshot(table, snapshots, "away_team", "away")
    table = table.merge(calculate_pregame_elo(fbs_games), on="game_id", how="left")
    table["elo_diff"] = table["home_elo"] - table["away_elo"]
    table["home_rest_days"] = (
        table["start_date"] - pd.to_datetime(table["home_last_game_date"], utc=True)
    ).dt.total_seconds() / 86400
    table["away_rest_days"] = (
        table["start_date"] - pd.to_datetime(table["away_last_game_date"], utc=True)
    ).dt.total_seconds() / 86400
    table["rest_days_diff"] = table["home_rest_days"] - table["away_rest_days"]
    table["neutral_site"] = table["neutral_site"].astype(int)
    table["conference_game"] = table["conference_game"].astype(int)
    table["postseason"] = table["season_type"].eq("postseason").astype(int)
    table["covid_season"] = table["season"].eq(2020).astype(int)

    home_features = {column[5:]: column for column in table.columns if column.startswith("home_")}
    away_features = {column[5:]: column for column in table.columns if column.startswith("away_")}
    for base_name in sorted(set(home_features).intersection(away_features)):
        is_historical_feature = (
            base_name in ["games_played", "season_games_played"]
            or base_name.endswith("_last_5")
            or base_name.endswith("_ewm")
            or base_name.endswith("_season")
        )
        if not is_historical_feature:
            continue
        home_column = home_features[base_name]
        away_column = away_features[base_name]
        if pd.api.types.is_numeric_dtype(table[home_column]) and pd.api.types.is_numeric_dtype(table[away_column]):
            table[f"diff_{base_name}"] = table[home_column] - table[away_column]

    if talent is not None and not talent.empty:
        home_talent = talent[["season", "team", "talent_score"]].rename(
            columns={"team": "home_team", "talent_score": "home_talent_score"}
        )
        away_talent = talent[["season", "team", "talent_score"]].rename(
            columns={"team": "away_team", "talent_score": "away_talent_score"}
        )
        table = table.merge(home_talent, on=["season", "home_team"], how="left")
        table = table.merge(away_talent, on=["season", "away_team"], how="left")
        table["talent_diff"] = table["home_talent_score"] - table["away_talent_score"]

    if recruiting is not None and not recruiting.empty:
        columns = ["season", "team", "recruiting_rank", "recruiting_points"]
        home_recruiting = recruiting[columns].rename(
            columns={
                "team": "home_team",
                "recruiting_rank": "home_recruiting_rank",
                "recruiting_points": "home_recruiting_points",
            }
        )
        away_recruiting = recruiting[columns].rename(
            columns={
                "team": "away_team",
                "recruiting_rank": "away_recruiting_rank",
                "recruiting_points": "away_recruiting_points",
            }
        )
        table = table.merge(home_recruiting, on=["season", "home_team"], how="left")
        table = table.merge(away_recruiting, on=["season", "away_team"], how="left")
        table["recruiting_points_diff"] = (
            table["home_recruiting_points"] - table["away_recruiting_points"]
        )

    if lines is not None and not lines.empty:
        table = table.merge(lines, on="game_id", how="left", validate="one_to_one")

    table["actual_margin"] = table["home_points"] - table["away_points"]
    table["actual_total"] = table["home_points"] + table["away_points"]
    return table.sort_values(["start_date", "game_id"]).reset_index(drop=True)


def check_for_leakage(modeling_table):
    """Raise an error if a rolling snapshot is not strictly earlier than kickoff."""
    for side in ["home", "away"]:
        snapshot_date = pd.to_datetime(modeling_table[f"{side}_last_game_date"], utc=True)
        invalid = snapshot_date.notna() & snapshot_date.ge(modeling_table["start_date"])
        if invalid.any():
            raise AssertionError(f"Found {invalid.sum()} leaked {side}-team snapshots.")
    return True


def get_model_features(table):
    """Return numeric pregame features while excluding targets and market information."""
    exact_features = {
        "week",
        "neutral_site",
        "conference_game",
        "postseason",
        "covid_season",
        "home_rest_days",
        "away_rest_days",
        "rest_days_diff",
        "home_elo",
        "away_elo",
        "elo_diff",
        "elo_home_win_probability",
        "home_talent_score",
        "away_talent_score",
        "talent_diff",
        "home_recruiting_rank",
        "away_recruiting_rank",
        "home_recruiting_points",
        "away_recruiting_points",
        "recruiting_points_diff",
    }
    features = []
    for column in table.select_dtypes(include=np.number).columns:
        is_snapshot = (
            column.startswith(("home_", "away_", "diff_"))
            and (
                column.endswith("games_played")
                or column.endswith("_last_5")
                or column.endswith("_ewm")
                or column.endswith("_season")
            )
        )
        if column in exact_features or is_snapshot:
            features.append(column)
    return sorted(features)
