"""Fine-tune DistilBERT for 5-way ticket classification.

DistilBERT is a 6-layer, 66M-parameter student of BERT-base (12 layers, 110M) trained with
knowledge distillation: ~40% smaller and ~60% faster while keeping ~97% of BERT's GLUE
score - a good fit for a laptop GPU/CPU.  Fine-tuning updates ALL weights plus a new
classification head on top of the [CLS] representation.

Training details
* input: lightly cleaned text (text_light), max_len tokens (default 128)
* AdamW, linear warm-up/decay, weight decay (regularisation)
* evaluation every epoch on the VALIDATION split; best checkpoint chosen by val macro-F1;
  early stopping if val macro-F1 does not improve for `patience` epochs (overfitting guard)
* fixed seed; GPU used automatically if available (fp16 on GPU)
* --fast: small training subset + 1 epoch for a quick CPU smoke test

Usage:  python -m src.models.train_transformer [--fast] [--epochs 4]
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed as hf_set_seed)

from src.data.load_dataset import load_processed
from src.evaluation.metrics import compute_metrics
from src.utils import ensure_dir, get_logger, load_config, resolve, save_json, set_seed

log = get_logger("train_transformer")


class TicketDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.enc = tokenizer(texts, truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="subset of training data + 1 epoch (CPU smoke test)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    tc, labels, seed = cfg["transformer"], cfg["labels"], cfg["seed"]
    set_seed(seed)
    hf_set_seed(seed)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    out_dir = resolve(f"{cfg['paths']['artifacts_dir']}/distilbert_helpdesk")
    ckpt_dir = resolve(f"{cfg['paths']['artifacts_dir']}/_distilbert_checkpoints")

    train, val = load_processed("train", cfg), load_processed("val", cfg)
    epochs = args.epochs or tc["epochs"]
    if args.fast:
        train = train.groupby("category", group_keys=False).apply(
            lambda g: g.sample(min(len(g), tc["fast_train_subset"] // len(labels)), random_state=seed))
        epochs = args.epochs or tc["fast_epochs"]
    use_gpu = torch.cuda.is_available()
    log.info("Device: %s | train=%d val=%d | epochs=%d", torch.cuda.get_device_name(0) if use_gpu else "CPU",
             len(train), len(val), epochs)

    tok = AutoTokenizer.from_pretrained(tc["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(tc["model_name"], num_labels=len(labels),
                                                               id2label=id2label, label2id=label2id)
    ds_tr = TicketDataset(train["text_light"].tolist(), [label2id[l] for l in train["category"]], tok, tc["max_len"])
    ds_va = TicketDataset(val["text_light"].tolist(), [label2id[l] for l in val["category"]], tok, tc["max_len"])

    def hf_metrics(eval_pred):
        logits, y = eval_pred
        pred = np.argmax(logits, axis=-1)
        return compute_metrics([id2label[i] for i in y], [id2label[i] for i in pred], labels)

    targs = TrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=epochs, learning_rate=tc["learning_rate"],
        per_device_train_batch_size=tc["batch_size"], per_device_eval_batch_size=64,
        weight_decay=tc["weight_decay"], warmup_ratio=tc["warmup_ratio"], lr_scheduler_type="linear",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, logging_steps=50, seed=seed,
        fp16=use_gpu, report_to=[], dataloader_num_workers=0,
    )
    from transformers import DataCollatorWithPadding

    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=hf_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=tc["early_stopping_patience"])])
    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0
    ev = trainer.evaluate()
    val_metrics = {k.replace("eval_", ""): v for k, v in ev.items() if k.replace("eval_", "") in
                   ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1")}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    save_json({"label2id": label2id, "id2label": id2label, "max_len": tc["max_len"]}, out_dir / "label_map.json")
    history = [h for h in trainer.state.log_history if "eval_macro_f1" in h or "loss" in h]
    info = {"model": "distilbert", "base_model": tc["model_name"], "fast_mode": args.fast, "epochs_run": trainer.state.epoch,
            "best_params": {"lr": tc["learning_rate"], "batch_size": tc["batch_size"], "max_len": tc["max_len"],
                            "weight_decay": tc["weight_decay"]},
            "val": val_metrics, "train_time_sec": round(train_time, 2), "device": "cuda" if use_gpu else "cpu",
            "n_train": len(train), "history": history}
    save_json(info, out_dir / "metrics.json")
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    log.info("DistilBERT valF1=%.4f  (%.1fs) saved -> %s", val_metrics.get("macro_f1", float("nan")), train_time, out_dir)


if __name__ == "__main__":
    main()
