"""ILP-based squad optimizer for FPL predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

try:
    import pulp  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - handled with message
    raise ImportError(
        "PuLP is required for ILP optimization. Install with `pip install pulp`."
    ) from exc


@dataclass(frozen=True)
class SquadConstraints:
    """Constraints for building an FPL squad."""

    budget: float = 100.0
    max_per_team: int = 3
    position_limits: dict[str, int] | None = None

    def resolved_position_limits(self) -> dict[str, int]:
        if self.position_limits is not None:
            return self.position_limits
        return {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def _resolve_common_gw(paths: list[Path]) -> int | None:
    gw_sets: list[set[int]] = []
    for path in paths:
        gws = (
            pd.read_csv(path, usecols=lambda col: col == "GW")
            .dropna()
            .astype({"GW": int})
            .loc[:, "GW"]
            .unique()
            .tolist()
        )
        gw_sets.append(set(gws))

    if not gw_sets:
        return None

    common = set.intersection(*gw_sets)
    if common:
        return max(common)

    return None


def _resolve_single_opponent(opponents: pd.Series) -> str | None:
    unique = opponents.dropna().unique().tolist()
    if len(unique) == 1:
        return cast(str, unique[0])
    return None


def _load_predictions(path: Path, position: str, gw: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    points_col = "combined_points" if "combined_points" in df.columns else "predicted_points"
    if gw is None:
        gw = int(df["GW"].max())

    df = df[df["GW"] == gw]
    df = df.dropna(subset=cast(list[str], [points_col, "now_cost"]))  # type: ignore[arg-type]
    if "availability_multiplier" not in df.columns:
        df = df.assign(availability_multiplier=1.0)
    if "opponent_name" not in df.columns:
        df = df.assign(opponent_name=pd.NA)

    group_keys = ["player_id", "name", "team", "now_cost"]
    agg_spec = {
        "predicted_points": (points_col, "sum"),
        "availability_multiplier": ("availability_multiplier", "max"),
        "opponent_name": ("opponent_name", _resolve_single_opponent),
    }
    if "fpl_id" in df.columns:
        agg_spec["fpl_id"] = ("fpl_id", "first")

    grouped = df.groupby(group_keys, as_index=False).agg(**agg_spec).assign(position=position)

    return grouped


def _apply_avoid_filter(players: pd.DataFrame, avoid_players: list[str]) -> pd.DataFrame:
    if not avoid_players:
        return players

    avoid_ids: set[int] = set()
    avoid_names: set[str] = set()
    for entry in avoid_players:
        cleaned = entry.strip()
        if not cleaned:
            continue
        if cleaned.isdigit():
            avoid_ids.add(int(cleaned))
        else:
            avoid_names.add(cleaned.casefold())

    filtered = players
    if avoid_ids:
        filtered = filtered[~filtered["player_id"].isin(avoid_ids)]
    if avoid_names and "name" in filtered.columns:
        filtered = filtered[
            ~filtered["name"].fillna("").astype(str).str.casefold().isin(avoid_names)
        ]
    return filtered


def build_optimal_squad(
    gk_path: Path,
    def_path: Path,
    mid_path: Path,
    fwd_path: Path,
    gw: int | None = None,
    constraints: SquadConstraints | None = None,
    bench_weight: float = 0.03,
    bench_max_cost: float | None = None,
    bench_gk_max_cost: float | None = None,
    min_total_spend: float | None = None,
    conflict_penalty_weight: float = 0.2,
    avoid_players: list[str] | None = None,
) -> tuple[pd.DataFrame, int | None]:
    """Build the optimal 15-player squad for a given gameweek."""

    constraints = constraints or SquadConstraints()
    position_limits = constraints.resolved_position_limits()

    paths = [gk_path, def_path, mid_path, fwd_path]
    resolved_gw = gw
    if resolved_gw is None:
        resolved_gw = _resolve_common_gw(paths)

    gk_df = _load_predictions(gk_path, "GK", resolved_gw)
    def_df = _load_predictions(def_path, "DEF", resolved_gw)
    mid_df = _load_predictions(mid_path, "MID", resolved_gw)
    fwd_df = _load_predictions(fwd_path, "FWD", resolved_gw)

    players = pd.concat([gk_df, def_df, mid_df, fwd_df], ignore_index=True)
    if avoid_players:
        players = _apply_avoid_filter(players, avoid_players)
    if players.empty:
        raise ValueError("No players available after filtering.")

    problem = pulp.LpProblem("fpl_squad_selection", pulp.LpMaximize)
    decision_vars = {
        idx: pulp.LpVariable(f"select_{idx}", lowBound=0, upBound=1, cat="Binary")
        for idx in players.index
    }
    starter_vars = {
        idx: pulp.LpVariable(f"starter_{idx}", lowBound=0, upBound=1, cat="Binary")
        for idx in players.index
    }

    objective = pulp.lpSum(
        (
            players.loc[idx, "predicted_points"] * starter_vars[idx]
            + bench_weight
            * players.loc[idx, "predicted_points"]
            * (decision_vars[idx] - starter_vars[idx])
        )
        for idx in players.index
    )

    problem += (
        pulp.lpSum(players.loc[idx, "now_cost"] * decision_vars[idx] for idx in players.index)
        <= constraints.budget
    )
    if min_total_spend is not None:
        problem += (
            pulp.lpSum(players.loc[idx, "now_cost"] * decision_vars[idx] for idx in players.index)
            >= min_total_spend
        )

    for position, limit in position_limits.items():
        position_indices = players.index[players["position"] == position]
        problem += pulp.lpSum(decision_vars[idx] for idx in position_indices) == limit

    for idx in players.index:
        problem += starter_vars[idx] <= decision_vars[idx]

    if bench_max_cost is not None:
        for idx in players.index:
            problem += (
                players.loc[idx, "now_cost"] * (decision_vars[idx] - starter_vars[idx])
                <= bench_max_cost
            )

    if bench_gk_max_cost is not None:
        bench_gk = players.index[players["position"] == "GK"]
        for idx in bench_gk:
            problem += (
                players.loc[idx, "now_cost"] * (decision_vars[idx] - starter_vars[idx])
                <= bench_gk_max_cost
            )

    problem += pulp.lpSum(starter_vars[idx] for idx in players.index) == 11

    starter_gk = players.index[players["position"] == "GK"]
    problem += pulp.lpSum(starter_vars[idx] for idx in starter_gk) == 1

    starter_def = players.index[players["position"] == "DEF"]
    starter_mid = players.index[players["position"] == "MID"]
    starter_fwd = players.index[players["position"] == "FWD"]

    problem += pulp.lpSum(starter_vars[idx] for idx in starter_def) >= 3
    problem += pulp.lpSum(starter_vars[idx] for idx in starter_def) <= 5
    problem += pulp.lpSum(starter_vars[idx] for idx in starter_mid) >= 2
    problem += pulp.lpSum(starter_vars[idx] for idx in starter_mid) <= 5
    problem += pulp.lpSum(starter_vars[idx] for idx in starter_fwd) >= 1
    problem += pulp.lpSum(starter_vars[idx] for idx in starter_fwd) <= 3

    for team in players["team"].unique():
        team_indices = players.index[players["team"] == team]
        problem += pulp.lpSum(decision_vars[idx] for idx in team_indices) <= constraints.max_per_team

    if conflict_penalty_weight > 0:
        attackers = players.index[players["position"].isin(["MID", "FWD"])]
        defenders = players.index[players["position"].isin(["DEF", "GK"])]
        defender_map: dict[tuple[str, str], list[int]] = {}
        for idx in defenders:
            team = players.loc[idx, "team"]
            opponent = players.loc[idx, "opponent_name"]
            if not isinstance(opponent, str) or not opponent:
                continue
            defender_map.setdefault((team, opponent), []).append(idx)

        conflict_vars: list[pulp.LpVariable] = []
        conflict_weights: list[float] = []
        for idx in attackers:
            team = players.loc[idx, "team"]
            opponent = players.loc[idx, "opponent_name"]
            if not isinstance(opponent, str) or not opponent:
                continue
            for defender_idx in defender_map.get((opponent, team), []):
                conflict_var = pulp.LpVariable(
                    f"conflict_{idx}_{defender_idx}", lowBound=0, upBound=1, cat="Binary"
                )
                conflict_vars.append(conflict_var)
                conflict_weights.append(
                    conflict_penalty_weight
                    * 0.5
                    * (
                        float(players.loc[idx, "predicted_points"])
                        + float(players.loc[defender_idx, "predicted_points"])
                    )
                )
                problem += conflict_var <= starter_vars[idx]
                problem += conflict_var <= starter_vars[defender_idx]
                problem += conflict_var >= starter_vars[idx] + starter_vars[defender_idx] - 1

        if conflict_vars:
            objective -= pulp.lpSum(
                weight * var for weight, var in zip(conflict_weights, conflict_vars, strict=False)
            )

    problem += objective

    solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)

    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"Optimization failed with status: {status}")

    selected_rows = [idx for idx, var in decision_vars.items() if var.value() == 1]
    selected = players.loc[selected_rows].copy()
    selected["is_starter"] = [starter_vars[idx].value() == 1 for idx in selected_rows]
    selected = selected.sort_values(["position", "predicted_points"], ascending=[True, False])
    return selected, resolved_gw


def summarize_squad(selected: pd.DataFrame) -> dict[str, float]:
    """Return total cost and points for a selected squad."""
    starter_points = selected.loc[selected["is_starter"], "predicted_points"].sum()
    bench_points = selected.loc[~selected["is_starter"], "predicted_points"].sum()
    return {
        "total_cost": float(selected["now_cost"].sum()),
        "total_points": float(selected["predicted_points"].sum()),
        "starter_points": float(starter_points),
        "bench_points": float(bench_points),
    }
