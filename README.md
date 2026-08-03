# AI-Calibrated Distributionally Robust Staffing and Routing

Public data and code for the manuscript **AI-calibrated distributionally
robust staffing and routing for environmental inspection systems**.

## Contents

- `data/processed/`: anonymized Chengdu dispatch ledger and data notes.
- `src/train_calibrated_risk_model.py`: chronological RoBERTa training,
  temperature calibration, and conformal intervals.
- `src/run_actionability_audit.py`: held-out policy evaluation.
- `src/run_two_stage_svrptw_experiment.py`: two-stage routing and recourse.
- `src/history_diagonal_metric.py`: proposed history-diagonal cut score.
- `src/summarize_iise_staffing.py`: validates and summarizes the audited
  12-date by 5-team staffing grid.
- `src/make_iise_staffing_figure.py`: generates the staffing-cost figure from
  the canonical aggregate outputs.
- `outputs/`: files underlying the manuscript tables and tests.

The raw administrative ledger is not included. Identifiers are hashed, exact
coordinates are replaced by surrogate coordinates, and addresses, contact
fields, staff names, organizations, handling notes, and original free text are
removed.

## Run

```bash
python -m pip install -r requirements.txt
python src/train_calibrated_risk_model.py --config config/defaults.json --skip-save
python src/run_actionability_audit.py --config config/defaults.json \
  --risk-file outputs/risk_predictions_bert_chrono_public.csv \
  --output-prefix outputs/actionability_residual
python src/summarize_iise_staffing.py
python src/make_iise_staffing_figure.py
```

The decomposition benchmark outputs are in
`outputs/full_benchmark_all_rows.csv`. The core history-diagonal update is
provided as a solver-independent Python module; IBM ILOG CPLEX and third-party
deepest-cut source are not redistributed.

The IISE staffing analysis is recorded in
`outputs/iise_staffing_grid_canonical.csv` (60 date--team rows),
`outputs/iise_staffing_policy_by_date.csv`,
`outputs/iise_staffing_policy_summary.csv`, and
`outputs/iise_staffing_paired_differences.csv`. These files contain aggregate
decision and performance measures only; they contain no report text, exact
coordinates, addresses, staff information, or organization identifiers.

Contact: Wei Xie, Southwest Jiaotong University,
`xiew7739@outlook.com`.
