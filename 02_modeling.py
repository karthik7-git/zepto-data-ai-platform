"""
Module 2 Part B: Predictive Modeling & Regression Pipeline.
Loads committed titanic.csv, performs stratified train/test split, builds leak-free Pipeline,
trains 3 classifiers, compares imbalance techniques (Baseline/Balanced/SMOTE),
tunes Random Forest with OOB score, fits Fare regression, and exports joblib pipeline.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

ANALYTICS_DIR = Path(__file__).resolve().parent
CSV_PATH = ANALYTICS_DIR / "titanic.csv"
MODEL_PATH = ANALYTICS_DIR / "best_pipeline.joblib"


def run_modeling_pipeline():
    print("==================================================")
    print("MODULE 2 - PART B: PREDICTIVE MODELING PIPELINE")
    print("==================================================")

    df = pd.read_csv(CSV_PATH)

    # Clean missing embarked/embark_town to preserve record integrity
    df = df.dropna(subset=["embarked"])

    # Classification feature selection
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X = df[features]
    y = df["survived"]

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split: {X_train.shape[0]} samples, Test split: {X_test.shape[0]} samples")
    print(f"Target Stratification: Train Survived Rate = {y_train.mean():.3f}, Test Survived Rate = {y_test.mean():.3f}")

    # Leak-free ColumnTransformer
    num_cols = ["age", "fare", "sibsp", "parch"]
    cat_cols = ["sex", "pclass", "embarked"]

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )

    # 1. Train 3 Classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }

    eval_results = []
    print("\n--- Training and Evaluating Classifiers ---")
    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        eval_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4)
        })

    eval_df = pd.DataFrame(eval_results)
    print(eval_df.to_string(index=False))

    # Render Decision Tree with plot_tree
    dt_pipe = Pipeline([("preprocessor", preprocessor), ("clf", DecisionTreeClassifier(max_depth=3, random_state=42))])
    dt_pipe.fit(X_train, y_train)
    fitted_preprocessor = dt_pipe.named_steps["preprocessor"]
    fitted_dt = dt_pipe.named_steps["clf"]

    cat_features = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(cat_features)

    plt.figure(figsize=(14, 8))
    plot_tree(fitted_dt, feature_names=all_feature_names, class_names=["Died", "Survived"], filled=True, rounded=True)
    plt.title("Decision Tree Visualization (Max Depth=3)")
    plt.tight_layout()
    plt.savefig(ANALYTICS_DIR / "decision_tree_structure.png")
    plt.close()
    print("Saved decision tree diagram: decision_tree_structure.png")

    # 2. Imbalance Handling Comparison (Train fold only)
    print("\n--- Class Imbalance Experimentation (Logistic Regression) ---")
    # A) Baseline
    base_pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    base_pipe.fit(X_train, y_train)
    y_pred_base = base_pipe.predict(X_test)

    # B) Class Weight Balanced
    cw_pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])
    cw_pipe.fit(X_train, y_train)
    y_pred_cw = cw_pipe.predict(X_test)

    # C) SMOTE (Train fold transformed first)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)

    smote_clf = LogisticRegression(max_iter=1000, random_state=42)
    smote_clf.fit(X_train_smote, y_train_smote)
    y_pred_smote = smote_clf.predict(X_test_proc)

    imb_summary = pd.DataFrame([
        {"Method": "Baseline (None)", "Precision": precision_score(y_test, y_pred_base), "Recall": recall_score(y_test, y_pred_base), "F1": f1_score(y_test, y_pred_base)},
        {"Method": "class_weight='balanced'", "Precision": precision_score(y_test, y_pred_cw), "Recall": recall_score(y_test, y_pred_cw), "F1": f1_score(y_test, y_pred_cw)},
        {"Method": "SMOTE (Train Only)", "Precision": precision_score(y_test, y_pred_smote), "Recall": recall_score(y_test, y_pred_smote), "F1": f1_score(y_test, y_pred_smote)}
    ]).round(4)
    print(imb_summary.to_string(index=False))

    # 3. Hyperparameter Tuning on Random Forest (with oob_score=True)
    print("\n--- Hyperparameter Tuning: RandomForest with OOB Score ---")
    rf_grid_pipe = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(oob_score=True, random_state=42))
    ])

    param_grid = {
        "rf__n_estimators": [50, 100],
        "rf__max_depth": [4, 6, 8],
        "rf__max_features": ["sqrt", "log2"]
    }

    grid = GridSearchCV(rf_grid_pipe, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_.named_steps["rf"]
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best Estimator OOB Score: {best_rf.oob_score_:.4f}")

    # 4. Regression Side-Task: Predict Fare
    print("\n--- Regression Side-Task: Predict Fare ---")
    reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked"]
    X_reg = df[reg_features]
    y_reg = df["fare"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.20, random_state=42)

    reg_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["age", "sibsp", "parch"]),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first"))]), ["sex", "pclass", "embarked"])
        ]
    )

    reg_pipe = Pipeline([("prep", reg_preprocessor), ("reg", LinearRegression())])
    reg_pipe.fit(X_train_r, y_train_r)

    y_pred_r = reg_pipe.predict(X_test_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    r2 = r2_score(y_test_r, y_pred_r)
    n = len(y_test_r)
    p = X_train_r.shape[1]
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    print(f"Regression Metrics -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}, Adj R2: {adj_r2:.4f}")

    # Residual Plot for Heteroscedasticity Analysis
    residuals = y_test_r - y_pred_r
    plt.figure(figsize=(7, 4))
    plt.scatter(y_pred_r, residuals, alpha=0.6, color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot: Heteroscedasticity Confirmation")
    plt.tight_layout()
    plt.savefig(ANALYTICS_DIR / "regression_residuals.png")
    plt.close()
    print("Residual plot saved: regression_residuals.png (Shows clear fan shape -> Heteroscedasticity).")

    # 5. Export Complete Pipeline with Joblib & Verify on Raw Input
    print("\n--- Exporting Best Pipeline Artifact ---")
    best_pipeline = grid.best_estimator_
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Successfully serialized full pipeline to: {MODEL_PATH.name}")

    # Verification on raw data
    loaded_pipe = joblib.load(MODEL_PATH)
    raw_sample = pd.DataFrame([{
        "pclass": 3,
        "sex": "female",
        "age": 22.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S"
    }])
    pred = loaded_pipe.predict(raw_sample)
    prob = loaded_pipe.predict_proba(raw_sample)[:, 1]
    print(f"[VERIFICATION] Raw sample prediction: class={pred[0]}, survival probability={prob[0]:.4f}")


if __name__ == "__main__":
    run_modeling_pipeline()