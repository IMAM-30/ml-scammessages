# Validasi Inference Lokal

- Model: `hybrid`
- Artifact: `hasil/scam_hybrid_indobert_bilstm`
- Device: `cpu`
- Jumlah test split: 92

## Manual Examples

- scam_bonus: `SCAM/SPAM` (prob_scam=0.9559, risk=tinggi)
- normal_kampus: `NORMAL/HAM` (prob_scam=0.0211, risk=sangat rendah)

## Test Split Metrics

- Accuracy: 0.978261
- Precision: 0.978261
- Recall: 0.978261
- F1: 0.978261
- Confusion: TP=45, TN=45, FP=1, FN=1
