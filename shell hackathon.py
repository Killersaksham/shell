# =============================================================================
# IMPORTS & CONFIGURATION
# =============================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import os
import logging
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures

# Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_PATH = "./data"
SUBMISSION_FILE = "submission.csv"
N_SPLITS = 10
SEED = 42
N_TRIALS = 15  # For Optuna


# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

def load_data():
    logging.info("📂 Loading data...")
    try:
        train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
        test_df = pd.read_csv(f"{DATA_PATH}/test.csv")

        rename_dict = {f'Component{i}_fraction': f'Component{i}' for i in range(1, 6)}
        train_df.rename(columns=rename_dict, inplace=True)
        test_df.rename(columns=rename_dict, inplace=True)
        return train_df, test_df
    except FileNotFoundError as e:
        logging.error(f"❌ Missing data file: {e}")
        return None, None


def feature_engineering(df):
    logging.info("⚙️ Performing feature engineering...")
    df_feat = df.copy()
    base_weighted_features = []

    for i in range(1, 11):
        prop = f'Property{i}'
        comp_cols = [f'Component{j}' for j in range(1, 6)]
        prop_cols = [f'Component{j}_{prop}' for j in range(1, 6)]

        weighted_avg = np.sum(df_feat[comp_cols].values * df_feat[prop_cols].values, axis=1)
        col_name = f'weighted_avg_{prop}'
        df_feat[col_name] = weighted_avg
        base_weighted_features.append(col_name)

        df_feat[f'std_dev_{prop}'] = df_feat[prop_cols].std(axis=1)
        df_feat[f'range_{prop}'] = df_feat[prop_cols].max(axis=1) - df_feat[prop_cols].min(axis=1)
        df_feat[f'skew_{prop}'] = df_feat[prop_cols].skew(axis=1)
        df_feat[f'kurt_{prop}'] = df_feat[prop_cols].kurtosis(axis=1)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_data = poly.fit_transform(df_feat[base_weighted_features])
    poly_names = poly.get_feature_names_out(base_weighted_features)
    poly_df = pd.DataFrame(poly_data, columns=poly_names, index=df_feat.index)
    interaction_cols = [col for col in poly_df.columns if " " in col]
    df_feat = pd.concat([df_feat, poly_df[interaction_cols]], axis=1)

    logging.info(f"🧠 Features after engineering: {df_feat.shape[1]}")
    return df_feat


def find_best_hyperparameters(X, y_target):
    logging.info(f"🔍 Running Optuna for {N_TRIALS} trials...")

    def objective(trial):
        params = {
            'objective': 'regression_l1',
            'metric': 'mape',
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbosity': -1,
            'n_jobs': -1,
            'seed': SEED
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, val_idx in kf.split(X, y_target):
            model = lgb.LGBMRegressor(**params)
            model.fit(X.iloc[train_idx], y_target.iloc[train_idx],
                      eval_set=[(X.iloc[val_idx], y_target.iloc[val_idx])],
                      eval_metric='mape',
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict(X.iloc[val_idx])
            scores.append(mean_absolute_percentage_error(y_target.iloc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, timeout=1800)
    logging.info(f"✅ Best MAPE: {study.best_value:.6f}")
    return study.best_params


def train_and_predict_final(X, y, X_test, best_params):
    logging.info("🏁 Final training and prediction phase started...")
    target_cols = y.columns
    final_preds = pd.DataFrame()
    oof_scores = []

    for target in target_cols:
        logging.info(f"🔧 Training target: {target}")
        oof_pred = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

        for train_idx, val_idx in kf.split(X):
            model = lgb.LGBMRegressor(**best_params)
            model.fit(X.iloc[train_idx], y[target].iloc[train_idx],
                      eval_set=[(X.iloc[val_idx], y[target].iloc[val_idx])],
                      eval_metric='mape',
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_pred[val_idx] = model.predict(X.iloc[val_idx])
            test_pred += model.predict(X_test) / N_SPLITS

        mape = mean_absolute_percentage_error(y[target], oof_pred)
        oof_scores.append(mape)
        logging.info(f"📊 OOF MAPE for {target}: {mape:.4f}")
        final_preds[target] = test_pred

    avg_mape = np.mean(oof_scores)
    logging.info(f"\n📉 Average OOF MAPE: {avg_mape:.4f} → Accuracy ≈ {100 - avg_mape * 100:.2f}%")
    return final_preds


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    start = time.time()
    train_df, test_df = load_data()
    if train_df is None or test_df is None:
        return

    target_cols = [f'BlendProperty{i}' for i in range(1, 11)]
    test_ids = test_df['ID'] if 'ID' in test_df.columns else pd.Series(range(len(test_df)))

    y = train_df[target_cols]
    train_features = train_df.drop(columns=[col for col in ['ID'] + target_cols if col in train_df.columns])
    test_features = test_df.drop(columns=['ID']) if 'ID' in test_df.columns else test_df

    combined = pd.concat([train_features, test_features], axis=0)
    combined_features = feature_engineering(combined)

    X = combined_features.iloc[:len(train_df)]
    X_test = combined_features.iloc[len(train_df):]

    best_params = find_best_hyperparameters(X, y[target_cols[0]])
    best_params.update({
        'objective': 'regression_l1',
        'metric': 'mape',
        'n_estimators': 2000,
        'verbosity': -1,
        'n_jobs': -1,
        'seed': SEED
    })

    predictions = train_and_predict_final(X, y, X_test, best_params)

    submission = pd.DataFrame({'ID': test_ids})
    submission = pd.concat([submission, predictions], axis=1)
    submission.to_csv(SUBMISSION_FILE, index=False)

    logging.info(f"\n✅ Submission saved as '{SUBMISSION_FILE}'")
    logging.info(f"⏱️ Total runtime: {time.time() - start:.2f} seconds")


# Final Entry Point
if __name__ == "__main__":
    main()
