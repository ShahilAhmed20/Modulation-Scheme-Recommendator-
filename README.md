# 📡 Modulation Recommendation System (Adaptive Modulation Classifier)

ML project that recommends the optimal **digital modulation scheme** for a wireless link based on channel and service conditions — mimicking link adaptation in 4G/5G systems.

## 📁 Project Files

| File | Description |
|---|---|
| `AMP.ipynb` | Original notebook (untouched) — dataset generation, training, interactive recommender |
| `modulation_dataset_balanced.csv` | Generated dataset — ~60,000 rows, 5 balanced classes (15,000 each) |
| `evaluate_modulation.py` | Evaluation script — generates confusion / precision / recall matrices |
| `modulation_tuning.py` | Fine-tuning script — copies the original pipeline and tunes models |
| `confusion_matrix.png` / `precision_matrix.png` / `recall_matrix.png` | Baseline (XGBoost 300/8/0.1) metric plots |
| `tuned_confusion_matrix.png` / `tuned_precision_recall_matrix.png` | Best tuned model metric plots |
| `evaluation_results.txt` | Baseline metrics text report |
| `tuned_results.txt` | All candidate-model comparison + best model metrics |
| `modulation_model_tuned.pkl` | Saved tuned model bundle (`model` + `scaler` + `label_encoder`) |

## 📊 Dataset

Synthetic, class-balanced (20% each): **BPSK, QPSK, 16QAM, 64QAM, 256QAM**.

**7 input features:** `snr_db`, `distance_m`, `tx_power_dbm`, `bandwidth_mhz`, `frequency_ghz`, `target_rate_mbps`, `velocity_kmh`
**3 derived features (engineering):** `path_loss_db`, `effective_snr_db`, `spectral_efficiency_needed` → 9 total features.

## 🔧 Pipeline

1. Dataset generation — per-class uniform ranges over SNR (−15…35 dB), rate, distance, power
2. Derived features — free-space path loss model (32.4 + 20·log₁₀f + 30·log₁₀d), effective SNR, spectral efficiency
3. Label-encode target, train/test split (80/20, `random_state=42`)
4. StandardScaler → XGBoost (300 trees, depth 8, lr 0.1) → evaluate
5. Fine-tuning — early-stopped XGBoost configs + Random Forest + HistGradientBoosting

## 📈 Results

| Metric (macro) | Baseline XGB *(original)* | ✅ HistGradientBoosting |
|---|---|---|
| Accuracy | 0.8510 | **0.8593** |
| Precision | 0.8517 | **0.8634** |
| Recall | 0.8504 | **0.8588** |
| F1 Score | 0.8499 | **0.8575** |

Tuned runs: 4 early-stopping XGBoost configs (50-round patience on a 15% validation holdout), plus Random Forest and HistGradientBoosting. Full comparison in `tuned_results.txt`.

**Error structure:** confusion is concentrated between adjacent schemes (64QAM↔16QAM, QPSK↔BPSK) whose SNR ranges physically overlap — expected behaviour for link-adaptation data.

**Interactive demo:** the third cell of `AMP.ipynb` is an input-prompt recommender printing the chosen scheme and prediction confidence.

## 🚀 How to Run

```powershell
cd "F:\Mini project"
python evaluate_modulation.py    # baseline metric plots (~results identical to notebook)
python modulation_tuning.py      # fine-tuned comparison + tuned .pkl
```

## 💡 Load the Saved Model

```python
import pandas as pd, numpy as np, joblib
b = joblib.load(r"F:\Mini project\modulation_model_tuned.pkl")

df = pd.DataFrame([{ 'snr_db': 28, 'distance_m': 300, 'tx_power_dbm': 22,
                     'bandwidth_mhz': 20, 'frequency_ghz': 2.4,
                     'target_rate_mbps': 120, 'velocity_kmh': 0 }])
df['path_loss_db'] = 32.4 + 20*np.log10(df['frequency_ghz']) + 30*np.log10(df['distance_m']/1000)
df['effective_snr_db'] = df['tx_power_dbm'] - df['path_loss_db'] - 5
df['spectral_efficiency_needed'] = df['target_rate_mbps'] / df['bandwidth_mhz']

X = df[b['feature_cols']]                    # correct feature order
pred = b['label_encoder'].inverse_transform(b['model'].predict(b['scaler'].transform(X)))
# -> "256QAM"  (confidence via model.predict_proba)
```

> ⚠️ **Note:** the tuned model learned the derived features *with generator noise* (±6 dB in path loss, ±3 dB noise in effective SNR — as in the dataset generator in cell 1 of `AMP.ipynb`). The code above uses mean values (no noise), same as the interactive recommender in `AMP.ipynb`. For exact parity, reproduce those random offsets, or refit the scaler/model on clean derived features.

## 🛠 Requirements

Python 3.13+ · pandas · numpy · scikit-learn · xgboost · matplotlib · seaborn · joblib

---

Made by ShahilAhmed20 · Adaptive Modulation ML Project (Mini Project)
