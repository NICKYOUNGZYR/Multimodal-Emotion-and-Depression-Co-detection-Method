from __future__ import annotations
import os
import argparse
import yaml
import torch
from torch.optim import AdamW

from src.data.datamodule import MultiModalDataModule, DataModuleConfig
from src.data.datasets import DatasetConfig
from src.data.audio_preprocess import AudioPreprocessConfig
from src.data.text_preprocess import TextPreprocessConfig
from src.models.full_model import MultiModalModel
from src.adaptation.sfda_trainer import SFDATrainer
from src.utils.seed import set_global_seed
from src.utils.logger import SimpleLogger
from src.utils.checkpoint import load_model_weights, save_model


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cfg", required=True)
    ap.add_argument("--model_cfg", required=True)
    ap.add_argument("--sfda_cfg", required=True)
    ap.add_argument("--source_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    data_cfg = load_yaml(args.data_cfg)
    model_cfg = load_yaml(args.model_cfg)
    sfda_cfg = load_yaml(args.sfda_cfg)

    cfg = {}
    cfg.update(data_cfg)
    cfg["model_cfg"] = model_cfg

    out_dir = data_cfg.get("exp", {}).get("output_dir", "runs/sfda")
    os.makedirs(out_dir, exist_ok=True)

    seed = int(data_cfg.get("train", {}).get("seed", 42))
    set_global_seed(seed)

    dcfg = DatasetConfig(**data_cfg["dataset"])
    acfg = AudioPreprocessConfig(**data_cfg["audio"])
    tcfg = TextPreprocessConfig(
        max_length=data_cfg["text"]["max_length"],
        backbone=model_cfg["encoders"]["text"]["backbone"],
    )
    mcfg = DataModuleConfig(**data_cfg["train"])

    dm = MultiModalDataModule(dcfg, acfg, tcfg, mcfg)
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # teacher = source model
    teacher = MultiModalModel(cfg).to(device)
    load_model_weights(teacher, args.source_ckpt, map_location=str(device))

    # student init from teacher
    student = MultiModalModel(cfg).to(device)
    load_model_weights(student, args.source_ckpt, map_location=str(device))

    optim = AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-2)
    logger = SimpleLogger(os.path.join(out_dir, "sfda"))

    trainer = SFDATrainer(student, teacher, optim, device, sfda_cfg)

    best_val = -1.0
    best_path = os.path.join(out_dir, "sfda_best.pt")
    last_path = os.path.join(out_dir, "sfda_last.pt")

    global_step = 0
    for epoch in range(args.epochs):
        logs = trainer.train_one_epoch(dm.target_loader(), epoch)
        logger.log(epoch, logs, prefix="sfda/")

        # optional: evaluate on val if labels exist (in synthetic they do)
        val_metrics = evaluate(student, dm.val_loader(), device, num_classes=dcfg.num_classes)
        logger.log(epoch, val_metrics, prefix="val/")

        save_model(last_path, student, extra={"epoch": epoch})

        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            save_model(best_path, student, extra={"epoch": epoch, "best_acc": best_val})

        global_step += 1

    logger.close()
    print("Done. Best SFDA ckpt:", best_path)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        pred = out.logits.argmax(dim=-1)
        ys.append(batch["label"].cpu())
        ps.append(pred.cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    from src.utils.metrics import compute_basic_metrics
    return compute_basic_metrics(y, p, num_classes=num_classes)


def to_device(batch, device):
    batch["eeg_grid"] = batch["eeg_grid"].to(device, non_blocking=True)
    batch["audio_spec"] = batch["audio_spec"].to(device, non_blocking=True)
    batch["audio_len"] = batch["audio_len"].to(device, non_blocking=True)
    batch["label"] = batch["label"].to(device, non_blocking=True)
    batch["is_labeled"] = batch["is_labeled"].to(device, non_blocking=True)
    batch["subject_id"] = batch["subject_id"].to(device, non_blocking=True)
    for k in batch["text"]:
        batch["text"][k] = batch["text"][k].to(device, non_blocking=True)
    return batch


if __name__ == "__main__":
    main()
