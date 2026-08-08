"""
Regenerates the legacy backtest summary/prediction artifacts from
data/processed/dashboard_data.json, so they can't silently drift out of sync
with the dashboard's apples-to-apples numbers again.

Background: data/processed/backtest_summary.json, poisson_backtest_summary.json,
and combined_backtest_summary.json were last written at the initial commit and
never regenerated after matches.parquet grew — they reported 1,520/1,420/1,420
matches while the dataset has since expanded. build_dashboard_data.py fixed the
underlying comparability bug (Elo scored on i=0..N vs Poisson/Combined on
i=100..N) and computed correct, identical-fixture-set numbers; this script just
propagates those numbers to the older files other tooling still reads, instead
of hand-editing them.

Does not touch src/models, src/evaluation, or any hyperparameter — it only
reformats numbers that already exist in dashboard_data.json into the older
files' schemas.

Usage
-----
  python scripts/sync_backtest_summaries.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DATA_PATH = ROOT / "data" / "processed" / "dashboard_data.json"
PROCESSED_DIR = ROOT / "data" / "processed"

# Fixed model config — unchanged from src/evaluation/backtest*.py and
# scripts/build_dashboard_data.py, reproduced here only as descriptive
# metadata on the summary files (not used in any calculation).
START_RATING = 1500.0
K = 40.0
HOME_ADV = 65.0
DRAW_PROB = 0.23
MAX_GOALS = 6


def main() -> None:
    data = json.loads(DASHBOARD_DATA_PATH.read_text())
    summary = data["summary"]
    min_train = data["min_train_matches"]
    elo_weight = data["elo_weight"]

    elo_summary = {
        "model": "elo",
        "matches": summary["elo"]["matches"],
        "accuracy_1x2": summary["elo"]["accuracy"],
        "log_loss": summary["elo"]["log_loss"],
        "brier_score": summary["elo"]["brier_score"],
        "start_rating": START_RATING,
        "k": K,
        "home_adv": HOME_ADV,
        "draw_prob": DRAW_PROB,
        "note": f"Restricted to i>={min_train} to match Poisson/Combined's fixture set — see scripts/build_dashboard_data.py",
    }
    (PROCESSED_DIR / "backtest_summary.json").write_text(json.dumps(elo_summary, indent=2))

    poisson_summary = {
        "model": "poisson",
        "matches": summary["poisson"]["matches"],
        "accuracy_1x2": summary["poisson"]["accuracy"],
        "log_loss": summary["poisson"]["log_loss"],
        "brier_score": summary["poisson"]["brier_score"],
        "min_train_matches": min_train,
        "max_goals": MAX_GOALS,
    }
    (PROCESSED_DIR / "poisson_backtest_summary.json").write_text(json.dumps(poisson_summary, indent=2))

    combined_summary = {
        "model": "combined",
        "matches": summary["combined"]["matches"],
        "accuracy_1x2": summary["combined"]["accuracy"],
        "log_loss": summary["combined"]["log_loss"],
        "brier_score": summary["combined"]["brier_score"],
        "elo_weight": elo_weight,
        "poisson_weight": 1 - elo_weight,
        "min_train_matches": min_train,
    }
    (PROCESSED_DIR / "combined_backtest_summary.json").write_text(json.dumps(combined_summary, indent=2))

    # elo_predictions.parquet is the one prediction artifact actually read
    # live (src/api/routes.py: /model/config's matches_backtested, and the
    # /backtest/summary fallback when the json above is missing) — rebuild it
    # from the same apples-to-apples rows so it can't disagree with the json.
    import pandas as pd

    elo_rows = [
        {
            "Date": m["date"],
            "HomeTeam": m["home_team"],
            "AwayTeam": m["away_team"],
            "FTR": m["ftr"],
            "p_home": m["elo_p_home"],
            "p_draw": m["elo_p_draw"],
            "p_away": m["elo_p_away"],
        }
        for m in data["matches"]
    ]
    pd.DataFrame(elo_rows).to_parquet(PROCESSED_DIR / "elo_predictions.parquet", index=False)

    print("Synced from dashboard_data.json:")
    print(f"  backtest_summary.json           matches={elo_summary['matches']}  acc={elo_summary['accuracy_1x2']:.4f}")
    print(f"  poisson_backtest_summary.json   matches={poisson_summary['matches']}  acc={poisson_summary['accuracy_1x2']:.4f}")
    print(f"  combined_backtest_summary.json  matches={combined_summary['matches']}  acc={combined_summary['accuracy_1x2']:.4f}")
    print(f"  elo_predictions.parquet         rows={len(elo_rows)}")
    print()
    print("NOT regenerated (stale, but not read live by the app — see PR notes):")
    print("  poisson_predictions.parquet, combined_predictions.parquet")
    print("  (dashboard_data.json doesn't carry their xg_home/xg_away columns;")
    print("   regenerating them requires rerunning the walk-forward Poisson fit.)")


if __name__ == "__main__":
    main()
