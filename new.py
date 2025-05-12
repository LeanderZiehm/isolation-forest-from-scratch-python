import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from utils import frequency_encode_df

metaverse = pd.read_csv('./datasets/MetaverseFinancial/metaverse_transactions_dataset.csv')

risk_score = metaverse['risk_score']
metaverse.drop(columns=['risk_score'], inplace=True)

anomaly = metaverse['anomaly']
metaverse.drop(columns=['anomaly'], inplace=True)

y = anomaly.map({'low_risk': 0, 'moderate_risk': 1, 'high_risk': 1})

X = metaverse

X['timestamp'] = pd.to_datetime(X['timestamp'])
X['timestamp'] = X['timestamp'].astype('int64') // 10**9

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numeric columns: {num_cols}")
print(f"Categorical columns: {cat_cols}")

X = frequency_encode_df(X, cat_cols)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)  

print(f"Train set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

contamination_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
best_contamination = None
best_f1 = -1

print("\n=== Cross-validation to find optimal contamination ===")
for contamination in contamination_values:
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    
    iso_forest.fit(X_train)
    
    val_preds = iso_forest.predict(X_val)
    val_outliers = (val_preds == -1).astype(int)
    
    f1 = f1_score(y_val.astype(int), val_outliers)
    
    print(f"Contamination: {contamination}, Validation F1 Score: {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_contamination = contamination

print(f"\nBest contamination value: {best_contamination} (F1: {best_f1:.4f})")

print("\n=== Training final model with optimal contamination ===")
final_model = IsolationForest(
    n_estimators=100,
    contamination=best_contamination,
    random_state=42
)

final_model.fit(X_trainval)

print("\n=== Evaluating on test set ===")
test_scores = final_model.decision_function(X_test)
test_preds = final_model.predict(X_test)
test_outliers = (test_preds == -1).astype(int)

print("Classification Report:")
print(classification_report(y_test.astype(int), test_outliers, target_names=['normal', 'outlier']))

print("\n=== Anomaly Analysis ===")

df_scores = pd.DataFrame({
    'score': test_scores,
    'is_outlier': test_outliers,
    'true_label': y_test.astype(int)
})

print("Decision score summary:")
print(df_scores['score'].describe())
print(f"Fraction flagged as anomalies: {np.mean(test_outliers):.4f}")

true_positives = X_test[(test_outliers == 1) & (y_test.astype(int) == 1)]
print(f"\nCorrectly identified anomalies (True Positives): {len(true_positives)}")

false_positives = X_test[(test_outliers == 1) & (y_test.astype(int) == 0)]
print(f"Incorrectly flagged normal transactions (False Positives): {len(false_positives)}")

if len(true_positives) > 0:
    print("\nSample of correctly identified anomalies:")
    print(true_positives.head(5))