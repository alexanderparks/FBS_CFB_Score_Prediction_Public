import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import INTERIM_DIR, RAW_DIR, create_project_directories


BASE_URL = "https://api.collegefootballdata.com"


def load_api_key(search_paths=None):
    """Load the CFBD API key without adding a secret to the notebook."""
    if os.getenv("CFBD_API_KEY"):
        return os.environ["CFBD_API_KEY"].strip()

    if search_paths is None:
        search_paths = [Path.cwd() / ".env", Path.cwd().parent / ".env"]

    for path in search_paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip().startswith("CFBD_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"").strip("'")

    raise RuntimeError(
        "CFBD_API_KEY was not found. Create .env in the project root and add your key."
    )


class CFBDClient:
    """Small API client with local JSON caching."""

    def __init__(self, api_key=None, base_url=BASE_URL):
        self.api_key = api_key or load_api_key()
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def get(self, endpoint, params=None, cache_path=None, refresh=False):
        params = params or {}
        cache_path = Path(cache_path) if cache_path else None

        if cache_path and cache_path.exists() and not refresh:
            return json.loads(cache_path.read_text())

        response = self.session.get(
            f"{self.base_url}/{endpoint.lstrip('/')}", params=params, timeout=90
        )
        response.raise_for_status()
        data = response.json()

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, indent=2))
        return data


def to_snake(name):
    """Convert CFBD camelCase field names to snake_case."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_").lower()


def _number(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _ratio(value):
    match = re.match(r"^\s*([0-9.]+)\s*[-/]\s*([0-9.]+)\s*$", str(value))
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def _possession_seconds(value):
    match = re.match(r"^\s*(\d+):(\d+)\s*$", str(value))
    if not match:
        return np.nan
    return 60 * int(match.group(1)) + int(match.group(2))


def normalize_games(records):
    games = pd.json_normalize(records, sep="_")
    if games.empty:
        return games
    games.columns = [to_snake(column) for column in games.columns]
    games = games.rename(columns={"id": "game_id"})
    games["start_date"] = pd.to_datetime(games["start_date"], utc=True)
    return games


def normalize_team_stats(records):
    """Convert nested game box scores to one row per team and game."""
    rows = []
    paired_categories = {
        "completion_attempts": ("completions", "pass_attempts"),
        "third_down_eff": ("third_down_conversions", "third_down_attempts"),
        "fourth_down_eff": ("fourth_down_conversions", "fourth_down_attempts"),
        "total_penalties_yards": ("penalties", "penalty_yards"),
    }

    for game in records:
        for team in game.get("teams", []):
            row = {
                "game_id": game.get("id"),
                "team_id": team.get("teamId"),
                "team": team.get("team"),
                "conference": team.get("conference"),
                "home_away": team.get("homeAway"),
                "points": team.get("points"),
            }
            for item in team.get("stats", []):
                category = to_snake(item.get("category"))
                value = item.get("stat")
                if category in paired_categories:
                    first, second = _ratio(value)
                    row[paired_categories[category][0]] = first
                    row[paired_categories[category][1]] = second
                elif category == "possession_time":
                    row["possession_seconds"] = _possession_seconds(value)
                else:
                    row[category] = _number(value)
            rows.append(row)
    return pd.DataFrame(rows)


def _flatten_nested(prefix, value, row):
    if isinstance(value, dict):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}_{to_snake(key)}" if prefix else to_snake(key)
            _flatten_nested(next_prefix, nested_value, row)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        row[prefix] = value


def normalize_advanced_stats(records):
    rows = []
    for item in records:
        row = {
            "game_id": item.get("gameId"),
            "season": item.get("season"),
            "week": item.get("week"),
            "team": item.get("team"),
            "opponent": item.get("opponent"),
        }
        _flatten_nested("offense", item.get("offense", {}), row)
        _flatten_nested("defense", item.get("defense", {}), row)
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_talent(records):
    talent = pd.json_normalize(records)
    if talent.empty:
        return pd.DataFrame(columns=["season", "team", "talent"])
    talent.columns = [to_snake(column) for column in talent.columns]
    return talent.rename(columns={"year": "season", "talent": "talent_score"})


def normalize_recruiting(records):
    recruiting = pd.json_normalize(records)
    if recruiting.empty:
        return pd.DataFrame(columns=["season", "team", "recruiting_rank", "recruiting_points"])
    recruiting.columns = [to_snake(column) for column in recruiting.columns]
    return recruiting.rename(
        columns={"year": "season", "rank": "recruiting_rank", "points": "recruiting_points"}
    )


def _home_margin_from_line(line, home_team, away_team):
    """Convert a formatted sportsbook line to a home-team margin prediction."""
    formatted = str(line.get("formattedSpread") or "").strip()
    match = re.match(r"^(.*?)\s+([+-]?\d+(?:\.\d+)?)$", formatted)
    if match:
        listed_team = match.group(1).strip().lower()
        listed_spread = float(match.group(2))
        if listed_team == str(home_team).strip().lower():
            return -listed_spread
        if listed_team == str(away_team).strip().lower():
            return listed_spread
    return np.nan


def normalize_lines(records):
    """Create consensus market estimates by taking the median across providers."""
    rows = []
    for game in records:
        provider_rows = []
        for line in game.get("lines", []):
            provider_rows.append(
                {
                    "provider": line.get("provider"),
                    "market_margin": _home_margin_from_line(
                        line, game.get("homeTeam"), game.get("awayTeam")
                    ),
                    "market_total": _number(line.get("overUnder")),
                }
            )
        provider_frame = pd.DataFrame(provider_rows)
        rows.append(
            {
                "game_id": game.get("id"),
                "market_margin": provider_frame["market_margin"].median()
                if not provider_frame.empty
                else np.nan,
                "market_total": provider_frame["market_total"].median()
                if not provider_frame.empty
                else np.nan,
                "line_providers": len(provider_frame),
            }
        )
    return pd.DataFrame(rows)


def download_season(client, season, refresh=False, max_week=20):
    """Download and cache each source needed for one season."""
    season_dir = RAW_DIR / str(season)
    current_refresh = refresh

    games = client.get(
        "/games",
        {"year": season, "seasonType": "both", "classification": "fbs"},
        season_dir / "games.json",
        current_refresh,
    )

    team_stats = []
    for week in range(1, max_week + 1):
        records = client.get(
            "/games/teams",
            {
                "year": season,
                "week": week,
                "seasonType": "both",
                "classification": "fbs",
            },
            season_dir / f"team_stats_week_{week:02d}.json",
            current_refresh,
        )
        team_stats.extend(records)

    advanced = client.get(
        "/stats/game/advanced",
        {"year": season, "seasonType": "both", "excludeGarbageTime": False},
        season_dir / "advanced_stats.json",
        current_refresh,
    )
    lines = client.get(
        "/lines",
        {"year": season, "seasonType": "both"},
        season_dir / "lines.json",
        current_refresh,
    )
    talent = client.get(
        "/talent", {"year": season}, season_dir / "talent.json", current_refresh
    )
    recruiting = client.get(
        "/recruiting/teams",
        {"year": season},
        season_dir / "recruiting.json",
        current_refresh,
    )
    return games, team_stats, advanced, lines, talent, recruiting


def download_all_data(start_season, end_season, refresh_current=True):
    """Download, normalize, and save the complete historical dataset."""
    create_project_directories()
    client = CFBDClient()
    collections = {name: [] for name in ["games", "team_stats", "advanced", "lines", "talent", "recruiting"]}

    for season in range(start_season, end_season + 1):
        print(f"Downloading {season} data...")
        records = download_season(
            client, season, refresh=refresh_current and season == end_season
        )
        for name, values in zip(collections, records):
            collections[name].extend(values)

    frames = {
        "games": normalize_games(collections["games"]),
        "team_stats": normalize_team_stats(collections["team_stats"]),
        "advanced_stats": normalize_advanced_stats(collections["advanced"]),
        "lines": normalize_lines(collections["lines"]),
        "talent": normalize_talent(collections["talent"]),
        "recruiting": normalize_recruiting(collections["recruiting"]),
    }
    for name, frame in frames.items():
        frame.to_csv(INTERIM_DIR / f"{name}.csv", index=False)
    return frames


def load_interim_data():
    """Load the normalized tables created by the first notebook."""
    tables = {}
    for name in ["games", "team_stats", "advanced_stats", "lines", "talent", "recruiting"]:
        path = INTERIM_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Run 01_data_acquisition.ipynb first. Missing: {path}")
        tables[name] = pd.read_csv(path)
    tables["games"]["start_date"] = pd.to_datetime(tables["games"]["start_date"], utc=True)
    return tables
