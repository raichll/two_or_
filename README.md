# AI-Calibrated Robust Staffing and Routing

Public data and code for the manuscript **AI-calibrated robust decision
support for environmental inspection staffing and routing**.

## Contents

- `data/processed/`: anonymized Chengdu dispatch ledger and data notes.
- `src/train_calibrated_risk_model.py`: chronological RoBERTa training,
  temperature calibration, and conformal intervals.
- `src/run_actionability_audit.py`: held-out policy evaluation.
- `src/run_two_stage_svrptw_experiment.py`: two-stage routing and recourse.
- `src/history_diagonal_metric.py`: proposed history-diagonal cut score.
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
```

The decomposition benchmark outputs are in
`outputs/full_benchmark_all_rows.csv`. The core history-diagonal update is
provided as a solver-independent Python module; IBM ILOG CPLEX and third-party
deepest-cut source are not redistributed.

Contact: Wei Xie, Southwest Jiaotong University,
`xiew7739@outlook.com`.
