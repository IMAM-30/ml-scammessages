"""Local inference engine for Indonesian scam/spam detection.

The engine loads Kaggle artifacts from ``hasil/scam_hybrid_indobert_bilstm``:
model checkpoint, tokenizer, preprocessing config, and final threshold.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "hasil" / "scam_hybrid_indobert_bilstm"


class DependencyMissingError(RuntimeError):
    """Raised when inference dependencies are not installed locally."""


def _load_ml_dependencies():
    try:
        import torch
        import torch.nn as nn
        from transformers import BertConfig, BertModel, BertTokenizerFast
    except ImportError as exc:
        raise DependencyMissingError(
            "Dependency inference belum terpasang. Jalankan: "
            "python3 -m pip install -r requirements-inference.txt"
        ) from exc
    return torch, nn, BertConfig, BertModel, BertTokenizerFast


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
REPEATED_SPACE_PATTERN = re.compile(r"\s+")


def normalize_repeated_chars(word: str) -> str:
    """Limit extreme repeated characters: hadiahhhh -> hadiahh."""
    return re.sub(r"(.)\1{2,}", r"\1\1", word)


@dataclass(frozen=True)
class PredictionResult:
    raw_text: str
    clean_text: str
    prediction: str
    label_id: int
    prob_scam: float
    threshold: float
    risk_level: str
    model: str


def _risk_level(prob_scam: float, threshold: float) -> str:
    if prob_scam >= max(0.85, threshold):
        return "tinggi"
    if prob_scam >= threshold:
        return "sedang"
    if prob_scam >= 0.35:
        return "rendah"
    return "sangat rendah"


def _read_tokenizer_vocab_size(tokenizer_json: Path) -> int:
    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = data.get("model", {}).get("vocab", {})
    if not vocab:
        raise ValueError(f"Tokenizer vocab tidak ditemukan: {tokenizer_json}")
    return len(vocab)


def _load_checkpoint(torch, checkpoint_path: Path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _positive_probs(torch, logits):
    if logits.ndim == 2 and logits.shape[-1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits.view(-1))


class ScamInferenceEngine:
    """Load model artifact and run local scam/spam inference."""

    def __init__(
        self,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        model_type: Literal["hybrid", "indobert"] = "hybrid",
        device: str = "cpu",
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.model_type = model_type
        self.device_name = device
        self.torch, self.nn, self.BertConfig, self.BertModel, self.BertTokenizerFast = _load_ml_dependencies()

        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"Folder artifact tidak ditemukan: {self.artifact_dir}")

        self.preprocessing_path = self.artifact_dir / "preprocessing_config.json"
        self.tokenizer_dir = self.artifact_dir / "tokenizer"
        self.tokenizer_json = self.tokenizer_dir / "tokenizer.json"

        if model_type == "hybrid":
            self.checkpoint_path = self.artifact_dir / "hybrid_indobert_bilstm_scam_final.pth"
        elif model_type == "indobert":
            self.checkpoint_path = self.artifact_dir / "indobert_classifier_scam_baseline.pth"
        else:
            raise ValueError("model_type harus 'hybrid' atau 'indobert'")

        self.preprocessing = json.loads(self.preprocessing_path.read_text(encoding="utf-8"))
        self.config = self.preprocessing["config"]
        self.slang_dict = self.preprocessing["slang_dict"]
        self.keep_numbers = bool(self.config.get("keep_numbers", True))
        self.max_length = int(self.config.get("max_length", 128))

        self.tokenizer = self.BertTokenizerFast(
            tokenizer_file=str(self.tokenizer_json),
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )

        self.device = self._resolve_device(device)
        self.checkpoint = _load_checkpoint(self.torch, self.checkpoint_path)
        self.threshold = float(self.checkpoint.get("threshold", 0.5))
        self.architecture = str(self.checkpoint.get("architecture", ""))
        self.model = self._build_model()
        self._load_state_dict()
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, device: str):
        if device != "auto":
            return self.torch.device(device)
        if self.torch.cuda.is_available():
            return self.torch.device("cuda")
        if hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
            return self.torch.device("mps")
        return self.torch.device("cpu")

    def _bert_config(self):
        state_dict = self.checkpoint.get("state_dict", {})
        embedding_weight = state_dict.get("bert.embeddings.word_embeddings.weight")
        if embedding_weight is not None:
            vocab_size = int(embedding_weight.shape[0])
        else:
            vocab_size = _read_tokenizer_vocab_size(self.tokenizer_json)
        return self.BertConfig(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            pad_token_id=self.tokenizer.pad_token_id or 0,
        )

    def _build_model(self):
        torch = self.torch
        nn = self.nn
        BertModel = self.BertModel
        bert_config = self._bert_config()
        engine_config = self.config
        dropout = float(self.config.get("dropout", 0.3))

        class AttentionPooling(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                self.score = nn.Sequential(
                    nn.Linear(input_size, input_size // 2),
                    nn.Tanh(),
                    nn.Linear(input_size // 2, 1, bias=False),
                )

            def forward(self, sequence_output, attention_mask):
                scores = self.score(sequence_output).squeeze(-1)
                scores = scores.masked_fill(attention_mask == 0, -1e4)
                weights = torch.softmax(scores, dim=1)
                return torch.sum(sequence_output * weights.unsqueeze(-1), dim=1)

        class IndoBertClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel(bert_config)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(bert_config.hidden_size, 2)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                return self.classifier(self.dropout(cls_embedding))

        class HybridIndoBertBiLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel(bert_config)
                lstm_hidden_size = int(engine_config.get("lstm_hidden_size", 256))
                lstm_layers = int(engine_config.get("lstm_layers", 2))
                self.lstm = nn.LSTM(
                    input_size=bert_config.hidden_size,
                    hidden_size=lstm_hidden_size,
                    num_layers=lstm_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if lstm_layers > 1 else 0.0,
                )
                bilstm_output_size = lstm_hidden_size * 2
                self.attention_pooling = AttentionPooling(bilstm_output_size)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(bilstm_output_size, 1)

            def forward(self, input_ids, attention_mask):
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                sequence_output = bert_outputs.last_hidden_state
                lstm_output, _ = self.lstm(sequence_output)
                pooled = self.attention_pooling(lstm_output, attention_mask)
                return self.classifier(self.dropout(pooled)).squeeze(-1)

        if self.model_type == "hybrid":
            return HybridIndoBertBiLSTM()
        return IndoBertClassifier()

    def _load_state_dict(self) -> None:
        state_dict = self.checkpoint.get("state_dict")
        if state_dict is None:
            raise ValueError(f"Checkpoint tidak memiliki state_dict: {self.checkpoint_path}")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        allowed_buffer_keys = {"bert.embeddings.position_ids", "bert.embeddings.token_type_ids"}
        important_missing = [key for key in missing if key not in allowed_buffer_keys]
        important_unexpected = [key for key in unexpected if key not in allowed_buffer_keys]
        if important_missing or important_unexpected:
            raise RuntimeError(
                "State dict tidak cocok dengan arsitektur lokal. "
                f"missing={important_missing}, unexpected={important_unexpected}"
            )

    def clean_text(self, text: str) -> str:
        value = str(text).lower()
        value = URL_PATTERN.sub(" link ", value)
        value = MENTION_PATTERN.sub(" user ", value)
        value = HASHTAG_PATTERN.sub(r"\1", value)

        if self.keep_numbers:
            value = re.sub(r"[^a-z0-9\s]", " ", value)
        else:
            value = re.sub(r"[^a-z\s]", " ", value)

        words: list[str] = []
        for word in value.split():
            word = normalize_repeated_chars(word)
            word = self.slang_dict.get(word, word)
            words.extend(word.split())

        cleaned = " ".join(words)
        return REPEATED_SPACE_PATTERN.sub(" ", cleaned).strip()

    def predict(self, text: str) -> PredictionResult:
        return self.predict_many([text])[0]

    def predict_many(self, texts: Iterable[str], batch_size: int = 16) -> list[PredictionResult]:
        raw_texts = [str(text) for text in texts]
        clean_texts = [self.clean_text(text) for text in raw_texts]
        results: list[PredictionResult] = []

        with self.torch.no_grad():
            for start in range(0, len(clean_texts), batch_size):
                raw_batch = raw_texts[start : start + batch_size]
                clean_batch = clean_texts[start : start + batch_size]
                encoded = self.tokenizer(
                    clean_batch,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = _positive_probs(self.torch, logits).detach().cpu().tolist()

                for raw_text, clean_text, prob in zip(raw_batch, clean_batch, probs):
                    prob_value = float(prob)
                    label_id = int(prob_value >= self.threshold)
                    prediction = "SCAM/SPAM" if label_id == 1 else "NORMAL/HAM"
                    results.append(
                        PredictionResult(
                            raw_text=raw_text,
                            clean_text=clean_text,
                            prediction=prediction,
                            label_id=label_id,
                            prob_scam=prob_value,
                            threshold=self.threshold,
                            risk_level=_risk_level(prob_value, self.threshold),
                            model=self.architecture or self.model_type,
                        )
                    )
        return results


def load_engine(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    model_type: Literal["hybrid", "indobert"] = "hybrid",
    device: str = "cpu",
) -> ScamInferenceEngine:
    return ScamInferenceEngine(artifact_dir=artifact_dir, model_type=model_type, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediksi scam/spam dari teks bahasa Indonesia.")
    parser.add_argument("--text", required=True, help="Teks pesan yang akan diprediksi.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR), help="Folder artifact hasil training.")
    parser.add_argument("--model", choices=["hybrid", "indobert"], default="hybrid", help="Model checkpoint yang dipakai.")
    parser.add_argument("--device", default="cpu", help="Device torch: cpu, cuda, mps, atau auto.")
    parser.add_argument("--json", action="store_true", help="Cetak output JSON.")
    args = parser.parse_args()

    engine = load_engine(args.artifact_dir, model_type=args.model, device=args.device)
    result = engine.predict(args.text)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(f"Prediksi : {result.prediction}")
        print(f"Prob scam: {result.prob_scam:.4f}")
        print(f"Threshold: {result.threshold:.2f}")
        print(f"Risiko   : {result.risk_level}")
        print(f"Clean    : {result.clean_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
