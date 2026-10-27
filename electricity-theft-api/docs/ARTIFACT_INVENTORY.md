# Artifact inventory

The inventory below reflects the final backend directory. Paths are relative
to `electricity-theft-api`.

## Retained artifacts

| Original filename | Final path | Classification | Purpose |
|---|---|---|---|
| `cnn_ens_state_dicts.pt` | `artifacts/models/cnn_ens_state_dicts.pt` | Required for runtime inference | Five `WideDeepV2` PyTorch state dictionaries used to generate and average the CNN feature. |
| `final_stack_model.joblib` | `artifacts/models/final_stack_model.joblib` | Required for runtime inference | XGBoost stack classifier with 1,036 input features and `predict_proba`. |
| `stack_feature_names.joblib` | `artifacts/metadata/stack_feature_names.joblib` | Required for runtime metadata and explainability | Ordered names for 1,034 readings, the missing fraction, and the CNN prediction. |
| `explainer.joblib` | `artifacts/explainability/explainer.joblib` | Required for `/explain` | SHAP `TreeExplainer` for the final XGBoost stack model. |

## Removed artifacts

These files were inspected and confirmed to be unused by the application,
tests, Dockerfile, and documented runtime workflow:

| Original filename | Previous path | Classification | Reason removed |
|---|---|---|---|
| `lime_assets.joblib` | `artifacts/explainability/lime_assets.joblib` | Optional explainability | Contains LIME reference arrays but no serialized LIME explainer; no backend code uses it. |
| `config.json` | `artifacts/metadata/config.json` | Training/notebook metadata | Runtime constants and feature metadata are defined/validated by the backend; no code reads this file. |
| `final_stack_model.json` | `artifacts/models/final_stack_model.json` | Redundant model serialization | Runtime requires the joblib sklearn wrapper; no active workflow in this project reads the JSON export. |
| `p_test.npy` | `artifacts/evaluation/p_test.npy` | Evaluation-only | Saved test probabilities; not loaded by runtime or tests. |
| `Xt_stack.npy` | `artifacts/evaluation/Xt_stack.npy` | Evaluation-only | Saved evaluation matrix; not loaded by runtime or tests. |
| `y_test.npy` | `artifacts/evaluation/y_test.npy` | Evaluation-only | Saved test labels; not loaded by runtime or tests. |
| `test_consumer_ids.joblib` | `artifacts/evaluation/test_consumer_ids.joblib` | Evaluation-only | IDs aligned with removed evaluation arrays; not loaded by runtime or tests. |

The evaluation files were used during the technical audit to compare one
trusted saved result before removal. The audit result and the unresolved
provenance limitation remain documented in `README.md`.

## Missing or external files

`Xt.npy` was not present in the API directory or the original export set and
was not created. Training notebooks, datasets, and export scripts remain
outside this API directory and were not deleted by the cleanup.

## Runtime requirements

The API requires:

- `artifacts/models/final_stack_model.joblib`
- `artifacts/models/cnn_ens_state_dicts.pt`
- `artifacts/metadata/stack_feature_names.joblib`

The SHAP artifact is optional for prediction but required for `/explain`.
