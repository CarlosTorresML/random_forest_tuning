# Random Forest Classifier – Hyperparameter Tuning with GridSearchCV

This project demonstrates the use of a Random Forest classifier on the `water_potability.csv` dataset, including comparison with a single Decision Tree and the process of hyperparameter tuning using `GridSearchCV`.

## Dataset
- **Source**: `water_potability.csv`
- **Target variable**: `Potability`
- **Features**: Physical and chemical properties of water samples
- **Missing values**: Removed using `.dropna()` (note: imputation is preferred in real-world scenarios)

## Objective
- Train a baseline `RandomForestClassifier`
- Compare it with a single `DecisionTreeClassifier`
- Perform hyperparameter tuning on the Random Forest using `GridSearchCV`
- Evaluate performance improvements on the test set

## Tools Used
- `pandas`
- `scikit-learn`: 
  - `RandomForestClassifier`, `DecisionTreeClassifier`
  - `GridSearchCV`, `train_test_split`
  - `accuracy_score`

## Hyperparameter Tuning
Tuning was done using:
- `n_estimators = [1000]`
- `max_depth = [25, 30, None]`
(For speed, this grid was reduced. A wider search is recommended for real projects.)

The best combination found by `GridSearchCV` was:
{'max_depth': 30, 'n_estimators': 1000}

## Results

| Model                | Accuracy (Test Set) |
|---------------------|---------------------|
| Decision Tree        | ~67%                |
| Random Forest (default) | ~70%                |
| Random Forest (optimized) | ~70.2%            |

## Notes
- Although the improvement may seem small, hyperparameter tuning is essential in real applications.
- In my opinion, `verbose=2` in `GridSearchCV` provides a good balance between detail and clarity.
