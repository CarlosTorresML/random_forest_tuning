# *** Random Forest Classifier ***
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
dataset = pd.read_csv('water_potability.csv')

# Print initial shape
print(f'Shape: {dataset.shape}')

# Remove rows with missing values and print the new shape
# Note: In practice, imputing is preferred to avoid data loss
dataset.dropna(inplace=True)
print(f'Shape after dropna: {dataset.shape}')

# Separate features and target
X = dataset.drop('Potability', axis=1)
y = dataset['Potability']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Create and train a default Random Forest classifier
rf = RandomForestClassifier(random_state=1)  # 100 trees by default
rf.fit(X_train, y_train)

# Make predictions
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Evaluate accuracy
print('\n*** RANDOM FOREST ACCURACY ***')
print(f'Training accuracy: {accuracy_score(y_train, y_pred_train)}')
print(f'Test accuracy:     {accuracy_score(y_test, y_pred_test)}')
# 100% in training vs ~70% in test typically indicates overfitting

# Train a single Decision Tree for comparison
tree = DecisionTreeClassifier(random_state=1)
tree.fit(X_train, y_train)

# Predict and evaluate
y_pred_train_tree = tree.predict(X_train)
y_pred_test_tree = tree.predict(X_test)
print('\n*** DECISION TREE ACCURACY ***')
print(f'Training accuracy: {accuracy_score(y_train, y_pred_train_tree)}')
print(f'Test accuracy:     {accuracy_score(y_test, y_pred_test_tree)}')

# Random Forest already performs better than a single Decision Tree.
# Now we fine-tune hyperparameters to improve further.

# *** Tuning Random Forest - GridSearchCV with Cross-Validation ***

# Define the grid of hyperparameters to test
# Reduced number of combinations for faster execution
param_grid = {
    'n_estimators': [1000],
    'max_depth': [25, 30, None]
}

# Base model
model = RandomForestClassifier(random_state=1)

# GridSearchCV will test all parameter combinations using cross-validation
cv = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    verbose=2  # Use 2 for progress updates; increase for more detail
)

# Train using training set
cv.fit(X_train, y_train)

# Output the best hyperparameters
print('\n*** BEST HYPERPARAMETERS ***')
print(f'{cv.best_params_}')

# *** Compare optimized vs non-optimized model performance ***

# Original (non-optimized) model
non_optimized = RandomForestClassifier(random_state=1)

# Optimized model using GridSearchCV results
optimized = RandomForestClassifier(
    random_state=1,
    max_depth=cv.best_params_['max_depth'],
    n_estimators=cv.best_params_['n_estimators']
)

# Train both models
non_optimized.fit(X_train, y_train)
optimized.fit(X_train, y_train)

# Make predictions on test set
y_pred_test_non_opt = non_optimized.predict(X_test)
y_pred_test_opt = optimized.predict(X_test)

# Compare results
print('\n*** TEST ACCURACY COMPARISON ***')
print(f'Non-optimized model accuracy: {accuracy_score(y_test, y_pred_test_non_opt)}')
print(f'Optimized model accuracy:     {accuracy_score(y_test, y_pred_test_opt)}')

# Even slight improvements demonstrate the value of hyperparameter tuning.
