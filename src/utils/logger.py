from __future__ import annotations
import os
import csv
import time
from typing import Dict, Optional


class SimpleLogger:
    """
    - Console printing
    - CSV logging
    - TensorBoard
    """
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = None

        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(out_dir, "tb")
            self.tb = SummaryWriter(tb_dir)
        except Exception:
            self.tb = None

        self.start_time = time.time()

    def log(self, step: int, metrics: Dict[str, float], prefix: str = "") -> None:
        # console
        parts = [f"{prefix}{k}={v:.4f}" for k, v in metrics.items()]
        elapsed = time.time() - self.start_time
        print(f"[step {step}] ({elapsed:.1f}s) " + " ".join(parts))

        # csv
        row = {"step": step, **{f"{prefix}{k}": float(v) for k, v in metrics.items()}}
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            if os.stat(self.csv_path).st_size == 0:
                self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # tensorboard
        if self.tb is not None:
            for k, v in metrics.items():
                self.tb.add_scalar(f"{prefix}{k}", float(v), step)

    def close(self) -> None:
        try:
            self._csv_file.close()
        except Exception:
            pass
        if self.tb is not None:
            self.tb.close()
