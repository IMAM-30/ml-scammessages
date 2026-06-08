"""Validate local inference against manual examples and the saved test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import DEFAULT_ARTIFACT_DIR, DependencyMissingError, load_engine


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    total = max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_test_split(path: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_markdown_report(output_path: Path, payload: dict) -> None:
    manual_rows = "\n".join(
        f"- {item['name']}: `{item['prediction']}` "
        f"(prob_scam={item['prob_scam']:.4f}, risk={item['risk_level']})"
        for item in payload["manual_examples"]
    )
    test = payload["test_split_metrics"]
    output_path.write_text(
        "\n".join(
            [
                "# Validasi Inference Lokal",
                "",
                f"- Model: `{payload['model_type']}`",
                f"- Artifact: `{payload['artifact_dir']}`",
                f"- Device: `{payload['device']}`",
                f"- Jumlah test split: {payload['test_count']}",
                "",
                "## Manual Examples",
                "",
                manual_rows,
                "",
                "## Test Split Metrics",
                "",
                f"- Accuracy: {test['accuracy']:.6f}",
                f"- Precision: {test['precision']:.6f}",
                f"- Recall: {test['recall']:.6f}",
                f"- F1: {test['f1']:.6f}",
                f"- Confusion: TP={test['tp']}, TN={test['tn']}, FP={test['fp']}, FN={test['fn']}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validasi inference lokal dari artifact training.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--model", choices=["hybrid", "indobert"], default="hybrid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    try:
        engine = load_engine(artifact_dir, model_type=args.model, device=args.device)
    except DependencyMissingError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    manual_inputs = [
        (
            "scam_bonus",
            "Selamat Anda memenangkan hadiah 100 juta. Transfer biaya admin ke rekening 1234 sekarang.",
        ),
        (
            "normal_kampus",
            "Besok kita kumpul jam 10 di kampus untuk diskusi tugas akhir.",
        ),
    ]
    manual_predictions = engine.predict_many([text for _, text in manual_inputs], batch_size=args.batch_size)

    test_texts, test_labels = load_test_split(artifact_dir / "test_split.csv")
    test_predictions = engine.predict_many(test_texts, batch_size=args.batch_size)
    test_pred_labels = [item.label_id for item in test_predictions]
    metrics = binary_metrics(test_labels, test_pred_labels)

    payload = {
        "model_type": args.model,
        "artifact_dir": project_relative(artifact_dir),
        "device": str(engine.device),
        "manual_examples": [
            {"name": name, **asdict(prediction)}
            for (name, _), prediction in zip(manual_inputs, manual_predictions)
        ],
        "test_count": len(test_labels),
        "test_split_metrics": metrics,
    }

    json_path = PROJECT_ROOT / "hasil" / f"inference_validation_{args.model}.json"
    md_path = PROJECT_ROOT / "hasil" / f"INFERENCE_VALIDATION_{args.model.upper()}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(md_path, payload)

    print("Validasi inference selesai.")
    print(f"Model: {args.model}")
    print(f"Device: {engine.device}")
    for item in payload["manual_examples"]:
        print(
            f"- {item['name']}: {item['prediction']} "
            f"prob={item['prob_scam']:.4f} risk={item['risk_level']}"
        )
    print(
        "Test split: "
        f"accuracy={metrics['accuracy']:.6f}, "
        f"precision={metrics['precision']:.6f}, "
        f"recall={metrics['recall']:.6f}, "
        f"f1={metrics['f1']:.6f}"
    )
    print(f"Report: {project_relative(md_path)}")
    print(f"JSON: {project_relative(json_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
