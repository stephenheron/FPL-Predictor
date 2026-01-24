import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_gk_lstm import LSTMRegressor, build_features, get_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GK predictions using LSTM model.")
    parser.add_argument(
        "--input-file",
        default="merged_fpl_understat_2025-26.csv",
        help="Season CSV to score.",
    )
    parser.add_argument(
        "--model-file",
        default="lstm_gk_model.pt",
        help="Saved LSTM model file.",
    )
    parser.add_argument(
        "--scaler-file",
        default="lstm_gk_scaler.pkl",
        help="Saved scaler file.",
    )
    parser.add_argument(
        "--position",
        default="GK",
        help="Position to score.",
    )
    parser.add_argument(
        "--roll-window",
        type=int,
        default=8,
        help="Rolling window for hint features.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length (defaults to model config).",
    )
    parser.add_argument(
        "--predict-gw",
        type=int,
        default=None,
        help="Optional GW filter to score only a single GW.",
    )
    parser.add_argument(
        "--fixtures-file",
        default="Fantasy-Premier-League/data/2025-26/fixtures.csv",
        help="Fixtures CSV for upcoming GWs.",
    )
    parser.add_argument(
        "--teams-file",
        default="Fantasy-Premier-League/data/2025-26/teams.csv",
        help="Teams CSV for opponent strengths.",
    )
    parser.add_argument(
        "--output-file",
        default="gk_predictions_lstm.csv",
        help="Where to write predictions.",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_future_rows(
    df: pd.DataFrame,
    fixtures_path: str,
    teams_path: str,
    predict_gw: int,
) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_path)
    fixtures = fixtures[fixtures["event"] == predict_gw]
    if fixtures.empty:
        raise ValueError(f"No fixtures found for GW {predict_gw} in {fixtures_path}")

    teams = pd.read_csv(teams_path)
    team_id_to_name = teams.set_index("id")["name"].to_dict()
    team_id_to_short = teams.set_index("id")["short_name"].to_dict()
    team_id_to_code = teams.set_index("id")["code"].to_dict()
    team_name_to_id = teams.set_index("name")["id"].to_dict()

    contexts = []
    for _, fixture in fixtures.iterrows():
        home_id = int(fixture["team_h"])
        away_id = int(fixture["team_a"])
        contexts.append(
            {
                "team_id": home_id,
                "opponent_id": away_id,
                "was_home": True,
                "fixture_id": fixture["id"],
            }
        )
        contexts.append(
            {
                "team_id": away_id,
                "opponent_id": home_id,
                "was_home": False,
                "fixture_id": fixture["id"],
            }
        )

    latest_rows = (
        df.sort_values(["season", "GW"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .copy()
    )
    latest_rows["team_id"] = latest_rows["team"].map(team_name_to_id.get)
    missing_mask = latest_rows["team_id"].isna()
    if bool(missing_mask.any()):
        missing = latest_rows.loc[missing_mask, "team"].dropna().unique().tolist()
        raise ValueError(f"Missing team ids for: {missing}")

    future_rows = []
    for context in contexts:
        team_rows = latest_rows[latest_rows["team_id"] == context["team_id"]].copy()
        if team_rows.empty:
            continue

        opponent_id = context["opponent_id"]
        opponent_name = team_id_to_name.get(opponent_id)
        opponent_short = team_id_to_short.get(opponent_id)
        opponent_code = team_id_to_code.get(opponent_id)
        team_rows["GW"] = predict_gw
        team_rows["round"] = predict_gw
        team_rows["fixture"] = context["fixture_id"]
        team_rows["was_home"] = context["was_home"]
        team_rows["opponent_team"] = opponent_id
        team_rows["opponent_name"] = opponent_name
        team_rows["opponent_short_name"] = opponent_short
        team_rows["opponent_code"] = opponent_code

        for col in ["opp_dyn_attack", "opp_dyn_defence", "opp_dyn_overall"]:
            team_rows[col] = np.nan

        team_rows["team_h_score"] = np.nan
        team_rows["team_a_score"] = np.nan
        team_rows["total_points"] = np.nan
        team_rows["is_future"] = True
        future_rows.append(team_rows)

    if not future_rows:
        raise ValueError(f"No future rows created for GW {predict_gw}")

    return pd.concat(future_rows, ignore_index=True)


def build_prediction_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    seq_len: int,
    predict_gw: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    sequences = []
    meta_rows = []
    df = df.sort_values(["season", "player_id", "GW"]).copy()
    for _, group in df.groupby(["season", "player_id"], sort=False):
        group = group.sort_values("GW")
        values = group[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        for idx in range(seq_len, len(group)):
            row = group.iloc[idx]
            if predict_gw is not None and int(row["GW"]) != predict_gw:
                continue
            sequences.append(values[idx - seq_len : idx])
            meta = row.to_dict()
            meta["row_index"] = row.name
            meta_rows.append(meta)
    if not sequences:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), pd.DataFrame()
    return np.stack(sequences), pd.DataFrame(meta_rows)


def compute_minutes_last_3(df: pd.DataFrame, predict_gw: int | None = None) -> pd.Series:
    df = df.sort_values(["season", "player_id", "GW"]).copy()
    if predict_gw is not None:
        history = df.loc[
            df["GW"] < predict_gw, ["season", "player_id", "GW", "minutes"]
        ].copy()
        history = pd.DataFrame(history)
        history["GW"] = pd.to_numeric(history["GW"], errors="coerce")
        minutes_lookup = history.groupby(["season", "player_id", "GW"], sort=False)[
            "minutes"
        ].sum()
        minutes_lookup = minutes_lookup.to_dict()

        keys = list(zip(df["season"], df["player_id"]))
        minutes_last_3 = [
            sum(
                minutes_lookup.get((season, player_id, gw), 0)
                for gw in range(predict_gw - 3, predict_gw)
            )
            for season, player_id in keys
        ]
        return pd.Series(minutes_last_3, index=df.index, dtype=float)

    grouped = df.groupby(["season", "player_id"], sort=False)
    shifted = grouped["minutes"].shift(1)
    shifted = pd.Series(shifted, index=df.index, dtype=float)
    minutes_last_3 = (
        shifted.groupby([df["season"], df["player_id"]])
        .rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    minutes_last_3 = minutes_last_3.reindex(df.index)
    return pd.Series(minutes_last_3.fillna(0), index=df.index, dtype=float)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_file)
    df["is_future"] = False
    if args.predict_gw is not None and (df["GW"] == args.predict_gw).sum() == 0:
        future_rows = build_future_rows(
            df,
            args.fixtures_file,
            args.teams_file,
            args.predict_gw,
        )
        df = pd.concat([df, future_rows], ignore_index=True)

    checkpoint = torch.load(args.model_file, map_location="cpu")
    roll_window = checkpoint.get("roll_window", args.roll_window)
    df = build_features(df, args.position, roll_window)

    feature_cols = checkpoint.get("feature_cols") or get_feature_columns(df, roll_window)
    seq_len = args.seq_len or checkpoint.get("seq_len")
    if seq_len is None:
        raise ValueError("Sequence length is missing. Provide --seq-len.")

    sequences, meta_rows = build_prediction_sequences(
        df,
        feature_cols,
        seq_len,
        args.predict_gw,
    )

    if len(sequences) == 0:
        raise ValueError("No sequences available for prediction.")

    with Path(args.scaler_file).open("rb") as handle:
        scaler = pickle.load(handle)

    flat = sequences.reshape(-1, sequences.shape[-1])
    scaled = scaler.transform(flat).reshape(sequences.shape).astype(np.float32)

    device = get_device()
    model = LSTMRegressor(
        input_size=scaled.shape[-1],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(scaled).to(device)
        preds = model(inputs).cpu().numpy()

    minutes_last_3 = compute_minutes_last_3(df, args.predict_gw)
    minutes_last_3 = minutes_last_3.reindex(meta_rows["row_index"]).fillna(0).to_numpy()

    multipliers = np.select(
        [
            minutes_last_3 >= 180,
            minutes_last_3 >= 90,
            minutes_last_3 > 0,
        ],
        [1.0, 0.7, 0.4],
        default=0.2,
    )

    preds = preds * multipliers

    output_cols = [
        "name",
        "player_id",
        "team",
        "season",
        "GW",
        "was_home",
        "opponent_name",
        "total_points",
    ]
    existing_cols = [col for col in output_cols if col in meta_rows.columns]
    result = meta_rows[existing_cols].copy()
    result["availability_multiplier"] = multipliers
    result["predicted_points"] = preds
    result.to_csv(args.output_file, index=False)
    print(f"Saved predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
