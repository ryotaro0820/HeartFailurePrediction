"""
心不全患者の転帰先予測モデルの学習スクリプト
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import os

# データ読み込み
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart_failure_synthetic_500.csv')
df = pd.read_csv(data_path, encoding='utf-8-sig')

print("=" * 50)
print("データ概要")
print("=" * 50)
print(f"サンプル数: {len(df)}")
print(f"カラム数: {len(df.columns)}")
print()

# 転帰先の分布
print("転帰先の分布:")
print(df['転帰先'].value_counts().sort_index())
print("0: 自宅, 1: 転院, 2: 施設")
print()

# 入院時データのみを使用（予測に使う特徴量）
feature_columns = [
    '年齢', '性別', '体重', 'BMI', '介護保険認定', '独居・支援',
    '診断名',  # カテゴリカル変数
    'NYHA', '心不全入院歴', '原疾患',
    '高血圧', '糖尿病', '腎疾患', '運動器疾患', '脳血管疾患',
    'EF', 'LAD', '血清Cre', '血清Alb', '血清BUN', '血清eGFR',
    '血清CRP', '血清Hb', '血清Na', 'NT-proBNP',
    'βブロッカー', 'ACE阻害薬', 'SGLT2阻害薬', 'ARB', 'MRA', 'ARNI', 'デバイス挿入',
    'BI入院', 'FIM入院', 'バランス入院', '起立入院', '歩行入院', '合計入院',
    '4m歩行', '通常歩行速度', '握力',
    '認知機能低下', '基本チェックリスト'
]

target_column = '転帰先'

# 特徴量とターゲットの準備
X = df[feature_columns].copy()
y = df[target_column].copy()

# 診断名をエンコード
le_diagnosis = LabelEncoder()
X['診断名'] = le_diagnosis.fit_transform(X['診断名'])

# 診断名のマッピングを保存
diagnosis_mapping = {i: label for i, label in enumerate(le_diagnosis.classes_)}
print("診断名のエンコーディング:")
for k, v in diagnosis_mapping.items():
    print(f"  {k}: {v}")
print()

# 欠損値の確認
print("欠損値の確認:")
missing = X.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("欠損値なし")
print()

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"訓練データ: {len(X_train)}, テストデータ: {len(X_test)}")
print()

# スケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# モデル学習（RandomForest）
print("=" * 50)
print("RandomForest モデル学習")
print("=" * 50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced',  # クラス不均衡対策
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# 評価
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n精度（Accuracy）: {accuracy:.3f}")
print("\n分類レポート:")
print(classification_report(y_test, y_pred, target_names=['自宅', '転院', '施設']))

print("\n混同行列:")
print(confusion_matrix(y_test, y_pred))

# クロスバリデーション
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"\n5-Fold CV スコア: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 特徴量の重要度
print("\n特徴量の重要度 (Top 15):")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# モデルと関連ファイルの保存
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True)

# モデル保存
joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
joblib.dump(le_diagnosis, os.path.join(models_dir, 'label_encoder_diagnosis.pkl'))

# 特徴量リストと設定を保存
config = {
    'feature_columns': feature_columns,
    'target_column': target_column,
    'target_labels': {0: '自宅', 1: '転院', 2: '施設'},
    'diagnosis_mapping': diagnosis_mapping,
    'accuracy': float(accuracy),
    'cv_score': float(cv_scores.mean())
}

with open(os.path.join(models_dir, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print("モデル保存完了")
print("=" * 50)
print(f"保存先: {models_dir}")
print("  - rf_model.pkl")
print("  - scaler.pkl")
print("  - label_encoder_diagnosis.pkl")
print("  - config.json")
