# Construction Task Delay Prediction Model

## Project approach
## Model Selection: XGBoost Regressor

### Why XGBoost?
1. **Excellent for tabular data** with mixed feature types
2. **Handles non-linearity** in delay patterns
3. **Built-in regularization** prevents overfitting
4. **Feature importance** for interpretability
5. **Robust performance** on small-medium datasets

## Features Used

### 1. Temporal Features
- `planned_start_month`, `planned_start_quarter`, `planned_start_day_of_week`
- `planned_start_week_of_year`, `is_weekend_start`
- **Rationale**: Seasonal effects and workday patterns affect delays

### 2. Duration Features
- `planned_duration_days`, `duration_category_encoded`
- **Rationale**: Longer tasks have different delay patterns

### 3. Categorical Features
- `crew_name_encoded`, `region_encoded`, `task_label_encoded`
- **Rationale**: Crew efficiency, regional factors, task complexity

### 4. Task Complexity
- `task_sequence`, `project_total_tasks`
- **Rationale**: Project scale and task dependencies

## Data Split

- **Training**: 80% of data
- **Validation**: 20% of data
- **Method**: Random stratified split (random_state=42)

Since we're predicting delays for new tasks (not time-series forecasting), random split is appropriate.

## Model Hyperparameters
```python
n_estimators=200       # Boosting rounds
max_depth=6            # Tree depth
learning_rate=0.05     # Conservative learning
subsample=0.8          # Row sampling
colsample_bytree=0.8   # Column sampling
```

## Evaluation Metrics

- **MAE**: Mean Absolute Error (easy to interpret in days)
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **R²**: Variance explained


## How to Run it
- **instalation**: after clone make virtual with lates python version
- **requirements.txt**: to install all packages run"pip install -r requirements.txt"
- **api**: then run simple command "uvicorn src.api:app --reload" provide required parameters 