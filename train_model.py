#!/usr/bin/env python3
"""Train models to predict next-day price for a chosen crypto.

Usage:
  python train_model.py --coin_id <coin_id>

If --coin_id is omitted the script trains on the coin with most records.
Saves models to `models_<coin_id>.joblib` in the current folder.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd


def load_and_merge(hist_path, yearly_path):
    df = pd.read_csv(hist_path)
    df1 = pd.read_csv(yearly_path)
    data = pd.merge(df, df1, on='coin_id', how='inner')
    # basic cleaning
    data.columns = [c.strip() for c in data.columns]
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
    # drop exact duplicates
    data.drop_duplicates(inplace=True)
    # require basic columns
    required = [c for c in ['coin_id','date','price'] if c in data.columns]
    data.dropna(subset=required, inplace=True)
    # coerce numeric columns
    for col in ['price','market_cap','volume','price_ma7','price_ma30','volatility_7d','cumulative_return']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col].fillna(data[col].median(), inplace=True)
    # compute daily_return if missing
    if 'daily_return' not in data.columns and 'price' in data.columns:
        data = data.sort_values(['coin_id','date'])
        data['daily_return'] = data.groupby('coin_id')['price'].pct_change().fillna(0)
    # compute 7d volatility if missing
    if 'volatility_7d' not in data.columns:
        data['volatility_7d'] = (
            data.groupby('coin_id')['daily_return']
            .rolling(window=7)
            .std()
            .reset_index(level=0, drop=True)
        ).fillna(0)
    # save cleaned merged dataset for reproducibility
    try:
        data.to_csv('crypto_cleaned.csv', index=False)
    except Exception:
        pass
    return data


def prepare_ml_data(data, coin_id, feature_candidates=None):
    df_ml = data[data['coin_id'] == coin_id].sort_values('date').copy()
    if df_ml.empty:
        raise ValueError(f'No data for coin_id={coin_id}')
    if feature_candidates is None:
        feature_candidates = ['price','price_ma7','price_ma30','volume','market_cap','volatility_7d']
    feat_cols = [c for c in feature_candidates if c in df_ml.columns]
    if 'price' not in df_ml.columns:
        raise ValueError('price column required')
    if not feat_cols:
        feat_cols = ['price']
    df_ml['target'] = df_ml['price'].shift(-1)
    df_ml = df_ml.dropna(subset=feat_cols + ['target']).reset_index(drop=True)
    if df_ml.empty:
        raise ValueError('No rows available after preparing ML dataset')
    X = df_ml[feat_cols]
    y = df_ml['target']
    return X, y, feat_cols


def train_and_save(X, y, feat_cols, coin_id, out_dir='.', n_estimators=100):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    pred_lr = lr.predict(X_test_s)
    rmse_lr = mean_squared_error(y_test, pred_lr, squared=False)
    r2_lr = r2_score(y_test, pred_lr)

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    rmse_rf = mean_squared_error(y_test, pred_rf, squared=False)
    r2_rf = r2_score(y_test, pred_rf)

    # Permutation importance on test set
    try:
        perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=1, n_jobs=-1)
        imp_df = pd.DataFrame({
            'feature': feat_cols,
            'importance_mean': perm.importances_mean,
            'importance_std': perm.importances_std,
        }).sort_values('importance_mean', ascending=False)
        imp_csv = str(out_dir / f'feature_importances_{coin_id}.csv')
        imp_df.to_csv(imp_csv, index=False)
        # plot
        plt.figure(figsize=(8, max(4, len(imp_df)*0.4)))
        plt.barh(imp_df['feature'][::-1], imp_df['importance_mean'][::-1], xerr=imp_df['importance_std'][::-1])
        plt.xlabel('Permutation importance (mean)')
        plt.title(f'Feature importances for {coin_id}')
        plt.tight_layout()
        imp_png = str(out_dir / f'feature_importances_{coin_id}.png')
        plt.savefig(imp_png)
        plt.close()
    except Exception as e:
        imp_csv = None
        imp_png = None
        print('Permutation importance failed:', e)

    out_path = out_dir / f'models_{coin_id}.joblib'
    joblib.dump({'lr': lr, 'rf': rf, 'scaler': scaler, 'features': feat_cols, 'coin_id': coin_id}, out_path)

    # Save actual vs predicted plot (compare on test set)
    try:
        y_test_vals = y_test.reset_index(drop=True).values
        pred_lr_vals = pred_lr if isinstance(pred_lr, (list, np.ndarray)) else np.array(pred_lr)
        pred_rf_vals = pred_rf if isinstance(pred_rf, (list, np.ndarray)) else np.array(pred_rf)
        nplot = min(200, len(y_test_vals))
        plt.figure(figsize=(10,5))
        idx = list(range(nplot))
        plt.plot(idx, y_test_vals[:nplot], label='Actual', linewidth=1)
        plt.plot(idx, pred_lr_vals[:nplot], label='Pred LR', linewidth=1)
        plt.plot(idx, pred_rf_vals[:nplot], label='Pred RF', linewidth=1)
        plt.title(f'Actual vs Predicted Next-Day Price for {coin_id}')
        plt.legend()
        plt.tight_layout()
        avp_png = str(out_dir / f'actual_vs_pred_{coin_id}.png')
        plt.savefig(avp_png)
        plt.close()
    except Exception:
        avp_png = None

    results = {
        'rmse_lr': rmse_lr,
        'r2_lr': r2_lr,
        'rmse_rf': rmse_rf,
        'r2_rf': r2_rf,
        'model_file': str(out_path),
        'feature_importance_csv': imp_csv,
        'feature_importance_png': imp_png,
        'actual_vs_pred_png': avp_png,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hist', default='crypto_historical_365days.csv')
    parser.add_argument('--yearly', default='crypto_yearly_performance.csv')
    parser.add_argument('--coin_id', default=None)
    parser.add_argument('--out_dir', default='.')
    args = parser.parse_args()

    data = load_and_merge(args.hist, args.yearly)
    if args.coin_id is None:
        args.coin_id = data['coin_id'].value_counts().idxmax()
    print('Selected coin_id:', args.coin_id)
    X, y, feat_cols = prepare_ml_data(data, args.coin_id)
    results = train_and_save(X, y, feat_cols, args.coin_id, out_dir=args.out_dir)
    print('Training complete. Results:')
    for k,v in results.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
