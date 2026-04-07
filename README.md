# EC決済における不正取引検知

東京大学「データサイエンティスト養成講座 (DSS) 2025年度」にて、**GMOペイメントゲートウェイ株式会社**と共同で取り組んだ、EC決済データにおける不正取引検知の機械学習パイプラインです。

> [English README](README_EN.md)

## 課題背景

GMOペイメントゲートウェイは年間決済処理金額約16兆円超、稼働店舗数156,575店以上を擁する決済インフラ企業です。不正取引の早期検知は財務損失を防ぐ一方、誤検知（偽陽性）は正規顧客の利便性を損ないます。GMOからの要件: **検出した不正は確実に本物の不正であること**（高い適合率）を維持しつつ、一定の再現率を確保すること。

**主な課題:**

- **クラス不均衡** -- 不正取引率は約10%（約65,000件中約6,800件）
- **時間的分布シフト** -- 学習期間の不正率13%に対し、テスト期間は5%に低下
- **データリーケージ防止** -- 全ての特徴量は過去のデータのみを使用して算出

## データセット

> データセットは機密情報のため、本リポジトリには含まれていません。

- EC取引データ 約65,000件（397日間）
- 16個の匿名化された初期特徴量: 時系列、金額、デバイスOS、店舗、地域、メールドメイン、住所等
- 目的変数: 正常取引 (0) / 不正取引 (1)

## プロジェクト構成

```
src/
  preprocessing.py    -- 特徴量エンジニアリング（5カテゴリ 65特徴量）
  validation.py       -- Rolling / Expanding ウォークフォワード分割
  models.py           -- LightGBM, XGBoost, CatBoost, Random Forest
  rfe.py              -- 再帰的特徴量削減（RFE）
  evaluation.py       -- 閾値最適化、評価指標

experiments/
  run_preprocessing.py          -- 生データ → 65特徴量生成
  run_model_comparison.py       -- 4モデル比較（rolling/expanding）
  run_rfe.py                    -- 特徴量選択 → 47個のロバスト特徴量
  run_threshold_optimization.py -- 最小再現率制約下での適合率最適化
  run_shap_analysis.py          -- SHAP特徴量重要度解析
```

## 特徴量エンジニアリング（65特徴量）

16個の初期特徴量から、5つのカテゴリで合計65個の特徴量を追加生成しました:

| カテゴリ | 個数 | 主な特徴量 |
|---|---|---|
| **ベース統計** | 5 | `purchase_count_cumulative`, `past_mean_amount`, `past_std_amount`, `mu_personal`, `sigma_personal` |
| **特徴構造** | 10 | カテゴリ出現頻度（`f9_count`, `f12_count`等）、共起頻度（`f6xf9_count`, `f6xf9xf12_count`等） |
| **ルールベース** | 7 | 金額末尾パターン（`f5_end_with_00`）、カテゴリ変更（`f6_changed`）、**`f7xf5`**（入金履歴なし×金額） |
| **特徴強調** | 10 | `f5_x_{count}` = 金額 / log(出現頻度)（各カウント特徴量に対して） |
| **統計モデル** | 33 | グループ別・時間窓別の不正率、対数正規分布LLR（金額・取引間隔）、過去平均との乖離度 |

全特徴量は expanding window や累積カウント、shift による遅延集計で算出し、**データリーケージを完全に防止**しています。

## 検証戦略

`[Train=50%, Val=10%, Test=10%, Step=5%]` の設定で2つのウォークフォワード戦略を採用:

- **Rolling Window** -- 固定サイズの学習窓が前方にスライド（7 fold）。直近の不正パターンへの適応力を重視。
- **Expanding Window** -- 学習窓がデータ先頭から拡大（7 fold）。長期的パターンの安定性を重視。

また、単純な時系列分割（学習: Day 0-325、テスト: Day 326-396）により、不正率の分布シフト（学習期間13% → テスト期間5%）を確認しました。

## 特徴量選択（RFE）

LightGBMのFeature Importance (gain) を用いた再帰的特徴量削減:

1. 全特徴量でLightGBMを訓練
2. 特徴量重要度が最も低い特徴量を1個削除、検証AUCを記録
3. 特徴量が1つになるまで繰り返し
4. 最大の検証AUCを達成した特徴量セットを選択

7つのRolling Splitで実行し、**半数以上（4 fold以上）で選択されたロバスト特徴量**として**47個**を最終採用。全foldで選択された特徴量には `f7`（入金履歴）、`f9`/`f10`/`f11`/`f12`（カテゴリ系）、`f7xf5`（交互作用）、`fraud_ratio_store_7d`（店舗不正率）等が含まれます。

## モデル評価結果

| モデル | テストAUC | F1 | 適合率 | 再現率 |
|---|---|---|---|---|
| **LightGBM** | **~0.94** | **~0.54** | - | **~0.64** |
| CatBoost | ~0.94 | ~0.52 | 高い | 低い |
| XGBoost | ~0.93 | ~0.50 | - | - |
| Random Forest | ~0.91 | ~0.48 | - | - |

Rolling Window検証下で最もF1/再現率のバランスが優れる**LightGBM**を最終モデルとして選定しました。

### 最小再現率制約下での適合率最適化

| 最小再現率 | 最良モデル | 適合率 | 再現率 |
|---|---|---|---|
| 10% | XGBoost | 74% | 13% |
| 30% | XGBoost | 60% | 33% |
| 50% | Random Forest | 51% | 52% |

## SHAP特徴量重要度解析

SHAP値による主な知見:

1. **`f7`（入金履歴）** -- 最重要特徴量。入金履歴なしが不正方向に強く寄与
2. **`f7xf5`（入金履歴なし×購入金額）** -- 非常に高い重要度を持つ交互作用項。「取引履歴のないユーザによる高額取引」が最強の不正シグナル
3. **`fraud_ratio_store_7d`** -- 直近7日間の店舗別不正取引割合。不正が多発する店舗での新規取引はリスクが高い

## 実行方法

```bash
pip install -r requirements.txt

# 1. 特徴量エンジニアリング
python -m experiments.run_preprocessing --input data/raw.csv --output data/processed.csv

# 2. モデル比較（rolling / expanding）
python -m experiments.run_model_comparison --data data/processed.csv --strategy rolling

# 3. RFEによる特徴量選択
python -m experiments.run_rfe --data data/processed.csv

# 4. 適合率-再現率の閾値分析
python -m experiments.run_threshold_optimization --data data/processed.csv

# 5. SHAP解析
python -m experiments.run_shap_analysis --data data/processed.csv
```
