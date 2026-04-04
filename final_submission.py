import os
import json
import math
import random
import argparse
import time
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ============================================================
# 0. 기본 유틸
# ============================================================

LABEL_MAP = {"stable": 0.0, "unstable": 1.0}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 완전 deterministic 보다 속도/안정성 균형을 우선
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False



def ensure_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def configure_torch_threads(num_threads: int = 1) -> None:
    num_threads = max(1, int(num_threads))
    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass



def binary_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float32), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def get_autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


# ============================================================
# 1. 전처리: 고정 ROI crop + resize
# ============================================================
# 핵심 변경점
# - pseudo-3D 제거
# - physics label / physics loss 제거
# - full image 대신 object가 있는 중심 영역만 잘라서 2-view CNN에 입력


def _clip_roi(roi: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = roi
    x1 = float(np.clip(x1, 0.0, 1.0))
    y1 = float(np.clip(y1, 0.0, 1.0))
    x2 = float(np.clip(x2, 0.0, 1.0))
    y2 = float(np.clip(y2, 0.0, 1.0))
    if x2 <= x1:
        x1, x2 = 0.25, 0.75
    if y2 <= y1:
        y1, y2 = 0.25, 0.75
    return x1, y1, x2, y2



def _jitter_roi(
    roi: Tuple[float, float, float, float],
    shift_ratio: float = 0.03,
    scale_ratio: float = 0.05,
) -> Tuple[float, float, float, float]:
    """ROI를 조금씩 흔들어서 과적합을 줄입니다."""
    x1, y1, x2, y2 = roi
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1)
    h = (y2 - y1)

    cx += np.random.uniform(-shift_ratio, shift_ratio)
    cy += np.random.uniform(-shift_ratio, shift_ratio)
    w *= 1.0 + np.random.uniform(-scale_ratio, scale_ratio)
    h *= 1.0 + np.random.uniform(-scale_ratio, scale_ratio)

    new_roi = (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)
    return _clip_roi(new_roi)



def _crop_resize_rgb(
    image_path: str,
    roi: Tuple[float, float, float, float],
    out_hw: Tuple[int, int],
    train: bool = False,
    roi_shift: float = 0.03,
    roi_scale: float = 0.05,
) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if train:
        roi = _jitter_roi(roi, shift_ratio=roi_shift, scale_ratio=roi_scale)
    else:
        roi = _clip_roi(roi)

    x1 = int(round(roi[0] * w))
    y1 = int(round(roi[1] * h))
    x2 = int(round(roi[2] * w))
    y2 = int(round(roi[3] * h))

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    crop = img[y1:y2, x1:x2]
    interp = cv2.INTER_AREA if (crop.shape[0] > out_hw[0] or crop.shape[1] > out_hw[1]) else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (out_hw[1], out_hw[0]), interpolation=interp)
    return crop



def _apply_pair_augment(front: np.ndarray, top: np.ndarray, enabled: bool) -> Tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return front, top

    # 좌우 반전은 물리적으로 동일 구조의 미러 케이스이므로 label-preserving 증강으로 사용
    if np.random.rand() < 0.5:
        front = np.ascontiguousarray(front[:, ::-1])
        top = np.ascontiguousarray(top[:, ::-1])

    # 두 뷰에 같은 photometric jitter 적용
    alpha = 1.0 + np.random.uniform(-0.10, 0.10)  # contrast
    beta = np.random.uniform(-12.0, 12.0)         # brightness
    front = np.clip(front.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    top = np.clip(top.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return front, top



def _to_tensor_uint8_rgb(img: np.ndarray) -> torch.Tensor:
    # [0,255] -> [0,1] -> [-1,1]
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    x = x * 2.0 - 1.0
    return x


# ============================================================
# 2. Dataset
# ============================================================

class TwoViewStructureDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        train: bool,
        front_roi: Tuple[float, float, float, float],
        top_roi: Tuple[float, float, float, float],
        front_hw: Tuple[int, int],
        top_hw: Tuple[int, int],
        has_label: bool = True,
        roi_shift: float = 0.03,
        roi_scale: float = 0.05,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        # df에 'image_root' 컬럼이 있으면 행별 경로를 사용하고,
        # 없으면 기존처럼 단일 image_root를 모든 행에 적용한다.
        # 이를 통해 train/dev처럼 이미지 디렉토리가 다른 샘플을 하나의 Dataset으로 처리한다.
        self._default_image_root = image_root
        self._per_row_root = 'image_root' in self.df.columns
        self.train = train
        self.front_roi = _clip_roi(front_roi)
        self.top_roi = _clip_roi(top_roi)
        self.front_hw = front_hw
        self.top_hw = top_hw
        self.has_label = has_label
        self.roi_shift = roi_shift
        self.roi_scale = roi_scale

        if self.has_label and "label" not in self.df.columns:
            raise ValueError("has_label=True 인데 dataframe에 'label' 컬럼이 없습니다.")

    def __len__(self) -> int:
        return len(self.df)

    def _load_pair(self, sample_id: str, image_root: str) -> Tuple[np.ndarray, np.ndarray]:
        front_path = os.path.join(image_root, sample_id, "front.png")
        top_path = os.path.join(image_root, sample_id, "top.png")

        front = _crop_resize_rgb(
            front_path,
            roi=self.front_roi,
            out_hw=self.front_hw,
            train=self.train,
            roi_shift=self.roi_shift,
            roi_scale=self.roi_scale,
        )
        top = _crop_resize_rgb(
            top_path,
            roi=self.top_roi,
            out_hw=self.top_hw,
            train=self.train,
            roi_shift=self.roi_shift,
            roi_scale=self.roi_scale,
        )
        front, top = _apply_pair_augment(front, top, enabled=self.train)
        return front, top

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        root = str(row["image_root"]) if self._per_row_root else self._default_image_root
        front, top = self._load_pair(sample_id, root)

        item = {
            "id": sample_id,
            "front": _to_tensor_uint8_rgb(front),
            "top": _to_tensor_uint8_rgb(top),
        }

        if self.has_label:
            label = LABEL_MAP[str(row["label"])]
            item["target"] = torch.tensor([label], dtype=torch.float32)

        return item


# ============================================================
# 3. 모델: 2-branch CNN + late fusion
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.GELU(),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ViewEncoder(nn.Module):
    def __init__(self, base_channels: int = 24, dropout: float = 0.05):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            ConvBNAct(3, c, k=3, s=1),
            nn.MaxPool2d(2),
            ConvBNAct(c, c * 2, k=3, s=1),
            SEBlock(c * 2),
            nn.MaxPool2d(2),
            ConvBNAct(c * 2, c * 4, k=3, s=1),
            SEBlock(c * 4),
            nn.MaxPool2d(2),
            ConvBNAct(c * 4, c * 4, k=3, s=1),
            nn.MaxPool2d(2),
            ConvBNAct(c * 4, c * 8, k=3, s=1),
            SEBlock(c * 8),
        )
        self.dropout = nn.Dropout(dropout)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.out_dim = c * 8 * 4 * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.spatial_pool(x).flatten(1)
        x = self.dropout(x)
        return x


class TwoViewCNN(nn.Module):
    def __init__(self, base_channels: int = 32, encoder_dropout: float = 0.05, head_dropout: float = 0.35):
        super().__init__()
        # 2-branch: front / top 각자 다른 encoder 사용
        self.front_encoder = ViewEncoder(base_channels=base_channels, dropout=encoder_dropout)
        self.top_encoder = ViewEncoder(base_channels=base_channels, dropout=encoder_dropout)

        feat_dim = self.front_encoder.out_dim
        # Projection for better feature alignment
        self.front_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU())
        self.top_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.GELU())

        fusion_dim = feat_dim * 4  # [f_front, f_top, |diff|, product]

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 1),
        )

    def forward(self, front: torch.Tensor, top: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_front = self.front_proj(self.front_encoder(front))
        f_top = self.top_proj(self.top_encoder(top))

        fused = torch.cat(
            [f_front, f_top, torch.abs(f_front - f_top), f_front * f_top],
            dim=1,
        )
        logit = self.head(fused)
        return {
            "logit": logit,
            "unstable_prob": torch.sigmoid(logit),
        }


# ============================================================
# 4. 학습 / 검증 / 추론
# ============================================================


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    tta_flip: bool = True,
) -> Tuple[list, np.ndarray, Optional[np.ndarray]]:
    model.eval()

    ids = []
    probs = []
    targets = []
    has_target = False

    for batch in loader:
        front = batch["front"].to(device, non_blocking=True)
        top = batch["top"].to(device, non_blocking=True)

        with get_autocast_context(device, amp_enabled):
            out = model(front, top)
            prob = torch.sigmoid(out["logit"]).squeeze(1)

            if tta_flip:
                # front / top 모두 동일하게 좌우 반전
                out_flip = model(torch.flip(front, dims=[3]), torch.flip(top, dims=[3]))
                prob_flip = torch.sigmoid(out_flip["logit"]).squeeze(1)
                prob = 0.5 * (prob + prob_flip)

        ids.extend(batch["id"])
        probs.extend(prob.detach().cpu().numpy().tolist())

        if "target" in batch:
            has_target = True
            targets.extend(batch["target"].squeeze(1).cpu().numpy().tolist())

    y_prob = np.asarray(probs, dtype=np.float32)
    y_true = np.asarray(targets, dtype=np.float32) if has_target else None
    return ids, y_prob, y_true



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_clip: float,
) -> float:
    model.train()
    running_loss = 0.0
    num_samples = 0

    for batch in loader:
        front = batch["front"].to(device, non_blocking=True)
        top = batch["top"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with get_autocast_context(device, amp_enabled):
            out = model(front, top)
            loss = criterion(out["logit"], target)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bsz = front.size(0)
        running_loss += loss.item() * bsz
        num_samples += bsz

    return running_loss / max(num_samples, 1)



def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_logloss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_logloss": float(best_logloss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


# ============================================================
# 5. 데이터 split 로딩
# ============================================================


def _tag_root(df: pd.DataFrame, root: str) -> pd.DataFrame:
    """df에 'image_root' 컬럼을 추가하여 Dataset이 행별로 경로를 참조할 수 있게 한다."""
    df = df.copy()
    df["image_root"] = root
    return df


def build_mixed_split(
    train_csv: str,
    train_dir: str,
    dev_csv: str,
    dev_dir: str,
    dev_val_n: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    train/dev 데이터를 혼합하여 학습셋과 검증셋을 구성한다.

    학습셋 구성
    -----------
    - train 전체 (고정 카메라, stable N개 + unstable N개)
    - dev에서 stratify 분리한 학습용 샘플 (dev_total - dev_val_n 개)

    검증셋 구성
    -----------
    - dev에서 stratify 분리한 검증용 샘플 (dev_val_n 개)
    - train 전체 (고정 카메라 도메인 성능 모니터링용)
    → 두 도메인 모두 Val LogLoss에 반영되므로 best model 기준이 균형 잡힌다.

    반환값
    ------
    (train_mixed_df, val_mixed_df): 각각 'image_root' 컬럼 포함
    """
    ensure_file(train_csv)
    ensure_file(dev_csv)

    train_df = pd.read_csv(train_csv)
    dev_df   = pd.read_csv(dev_csv)

    # dev를 stratify split → 학습용 / 검증용
    dev_train_df, dev_val_df = train_test_split(
        dev_df,
        test_size=dev_val_n,
        stratify=dev_df["label"],
        random_state=seed,
    )

    # 각 DataFrame에 이미지 루트 경로 태깅
    train_tagged     = _tag_root(train_df,     train_dir)
    dev_train_tagged = _tag_root(dev_train_df, dev_dir)
    dev_val_tagged   = _tag_root(dev_val_df,   dev_dir)

    # 학습셋: train 전체 + dev 학습용
    train_mixed = pd.concat([train_tagged, dev_train_tagged], ignore_index=True)

    # 검증셋: dev 검증용 + train 전체
    # (dev_val → 가변 카메라 성능 / train → 고정 카메라 성능 모두 반영)
    val_mixed = pd.concat([dev_val_tagged, train_tagged], ignore_index=True)

    return train_mixed.reset_index(drop=True), val_mixed.reset_index(drop=True)


def read_split(
    train_csv: str,
    train_dir: str,
    dev_csv: Optional[str],
    dev_dir: Optional[str],
    fallback_val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, str, pd.DataFrame, str]:
    """기존 단일-루트 split (하위 호환용으로 유지)."""
    ensure_file(train_csv)
    train_df_all = pd.read_csv(train_csv)

    if dev_csv is not None and os.path.exists(dev_csv):
        val_df = pd.read_csv(dev_csv)
        val_root = dev_dir if dev_dir is not None else train_dir
        return train_df_all, train_dir, val_df, val_root

    if "label" not in train_df_all.columns:
        raise ValueError("dev.csv가 없고 train.csv에도 label이 없어 split을 만들 수 없습니다.")

    train_df, val_df = train_test_split(
        train_df_all,
        test_size=fallback_val_ratio,
        stratify=train_df_all["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), train_dir, val_df.reset_index(drop=True), train_dir


# ============================================================
# 6. 메인 파이프라인
# ============================================================


def main(args: argparse.Namespace) -> None:
    total_start = time.time()
    set_seed(args.seed)
    configure_torch_threads(args.torch_threads)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    print(f"[*] device: {device}")
    print(f"[*] AMP: {'ON' if amp_enabled else 'OFF'}")
    print(f"[*] torch threads: {torch.get_num_threads()}")

    front_roi = tuple(args.front_roi)
    top_roi = tuple(args.top_roi)
    front_hw = (args.front_h, args.front_w)
    top_hw = (args.top_h, args.top_w)

    # ------------------------------------------------------------------
    # 혼합 split 모드: train 전체 + dev 학습용을 합쳐 학습,
    #                  dev 검증용 + train 전체를 합쳐 검증
    # ------------------------------------------------------------------
    train_df, val_df = build_mixed_split(
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        dev_csv=args.dev_csv,
        dev_dir=args.dev_dir,
        dev_val_n=args.dev_val_n,
        seed=args.seed,
    )

    print(f"[*] train samples: {len(train_df)} | val samples: {len(val_df)}")
    # 도메인별 구성 현황 출력
    for tag, df in [("train_mixed", train_df), ("val_mixed", val_df)]:
        domain_counts = df["image_root"].value_counts().to_dict()
        label_counts  = df["label"].value_counts().to_dict() if "label" in df.columns else {}
        print(f"[*] {tag} | domain: {domain_counts} | label: {label_counts}")

    # df에 'image_root' 컬럼이 있으므로 image_root 인수는 빈 문자열 전달
    # (Dataset 내부에서 _per_row_root=True 경로로 동작)
    train_ds = TwoViewStructureDataset(
        df=train_df,
        image_root="",
        train=True,
        front_roi=front_roi,
        top_roi=top_roi,
        front_hw=front_hw,
        top_hw=top_hw,
        has_label=True,
        roi_shift=args.roi_shift,
        roi_scale=args.roi_scale,
    )
    val_ds = TwoViewStructureDataset(
        df=val_df,
        image_root="",
        train=False,
        front_roi=front_roi,
        top_roi=top_roi,
        front_hw=front_hw,
        top_hw=top_hw,
        has_label=True,
        roi_shift=0.0,
        roi_scale=0.0,
    )

    train_loader = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TwoViewCNN(
        base_channels=args.base_channels,
        encoder_dropout=args.encoder_dropout,
        head_dropout=args.head_dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # ReduceLROnPlateau는 Val LogLoss가 에포크마다 크게 요동칠 때 잘못된 시점에 LR을 낮춰
    # 과적합을 더 심화시키는 문제가 있었음 → CosineAnnealingLR로 교체
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_logloss = float("inf")
    patience_counter = 0
    history = []
    ckpt_path = os.path.join(args.output_dir, "best_model.pth")

    training_start = time.time()
    best_epoch = 0
    time_budget = 3600  # 60 minutes

    for epoch in range(1, args.epochs + 1):
        # Check time budget
        if time.time() - training_start > time_budget:
            print(f"\n[*] Time budget exceeded ({time_budget}s). Stopping training.")
            break

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip=args.grad_clip,
        )

        _, val_prob, val_true = predict_loader(
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            tta_flip=args.val_tta,
        )
        val_logloss = binary_logloss(val_true, val_prob)
        scheduler.step()  # CosineAnnealingLR은 인자 없이 에포크 단위 호출
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_logloss": float(val_logloss),
                "lr": float(current_lr),
            }
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val LogLoss: {val_logloss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # 30배 rule: train_loss가 val_logloss의 30배보다 크면 저장을 건너뜀 (너무 큰 괴리 방지)
        if val_logloss < best_logloss - args.min_delta:
            if train_loss > 30 * val_logloss:
                print(f"  -> [!] Skip saving: train_loss ({train_loss:.4f}) is > 30x val_logloss ({val_logloss:.4f})")
            else:
                best_logloss = val_logloss
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_logloss=best_logloss,
                    args=args,
                )
                print(f"  -> [*] Best Model Saved. (Val LogLoss: {best_logloss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping:
                print(f"\n[*] Early Stopping: {args.early_stopping} 에포크 동안 개선 없음.")
                break

    training_seconds = time.time() - training_start
    total_seconds = time.time() - total_start

    # VRAM 사용량 (GPU가 없으면 0)
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    # 파라미터 수 (M 단위)
    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6

    history_path = os.path.join(args.output_dir, "history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # best model 로드 후 train/dev 전체로 최종 검증
    # ------------------------------------------------------------------
    print("\n[*] Best model로 train/dev 전체 최종 검증 중...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Train 전체 검증
    train_full_df = pd.read_csv(args.train_csv)
    train_full_df["image_root"] = args.train_dir
    train_full_ds = TwoViewStructureDataset(
        df=train_full_df,
        image_root="",
        train=False,
        front_roi=front_roi,
        top_roi=top_roi,
        front_hw=front_hw,
        top_hw=top_hw,
        has_label=True,
        roi_shift=0.0,
        roi_scale=0.0,
    )
    train_full_loader = make_dataloader(train_full_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    _, train_prob, train_true = predict_loader(model=model, loader=train_full_loader, device=device, amp_enabled=amp_enabled, tta_flip=args.val_tta)
    train_full_logloss = binary_logloss(train_true, train_prob)

    # Dev 전체 검증
    dev_full_df = pd.read_csv(args.dev_csv)
    dev_full_df["image_root"] = args.dev_dir
    dev_full_ds = TwoViewStructureDataset(
        df=dev_full_df,
        image_root="",
        train=False,
        front_roi=front_roi,
        top_roi=top_roi,
        front_hw=front_hw,
        top_hw=top_hw,
        has_label=True,
        roi_shift=0.0,
        roi_scale=0.0,
    )
    dev_full_loader = make_dataloader(
        dev_full_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    _, dev_prob, dev_true = predict_loader(
        model=model,
        loader=dev_full_loader,
        device=device,
        amp_enabled=amp_enabled,
        tta_flip=args.val_tta,
    )
    dev_full_logloss = binary_logloss(dev_true, dev_prob)
    print(f"[*] Train full LogLoss ({len(train_full_df)}개): {train_full_logloss:.6f}")
    print(f"[*] Dev full LogLoss ({len(dev_full_df)}개): {dev_full_logloss:.6f}")

    # ------------------------------------------------------------------
    # autoresearch 형식 최종 결과 출력
    # program.md 가 이 블록을 파싱하여 val_logloss 를 핵심 지표로 읽습니다.
    # val_logloss 는 dev 전체 기준으로 출력한다.
    # 형식을 변경하지 마세요.
    # ------------------------------------------------------------------
    print()
    print(f"--- val_logloss: {dev_full_logloss:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds: {total_seconds:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"num_epochs: {best_epoch}")
    print(f"num_params_M: {num_params_m:.1f}")
    print(f"base_channels: {args.base_channels}")
    print(f"encoder_dropout: {args.encoder_dropout}")
    print(f"head_dropout: {args.head_dropout}")
    print(f"lr: {args.lr}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"batch_size: {args.batch_size}")
    print()

    print(f"[*] History saved to: {history_path}")

    # --------------------------------------------------------
    # test inference
    # --------------------------------------------------------
    if args.test_csv is None or not os.path.exists(args.test_csv):
        print("[*] test_csv가 없어 submission 생성은 생략합니다.")
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"best checkpoint를 찾을 수 없습니다: {ckpt_path}")

    print("\n[*] 테스트 추론 시작")
    # best model은 위 dev 전체 검증 시 이미 로드됨
    model.eval()

    test_df = pd.read_csv(args.test_csv)
    test_ds = TwoViewStructureDataset(
        df=test_df,
        image_root=args.test_dir,
        train=False,
        front_roi=front_roi,
        top_roi=top_roi,
        front_hw=front_hw,
        top_hw=top_hw,
        has_label=False,
        roi_shift=0.0,
        roi_scale=0.0,
    )
    test_loader = make_dataloader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ids, test_prob, _ = predict_loader(
        model=model,
        loader=test_loader,
        device=device,
        amp_enabled=amp_enabled,
        tta_flip=args.test_tta,
    )
    test_prob = np.clip(test_prob, 1e-7, 1.0 - 1e-7)

    pred_df = pd.DataFrame(
        {
            "id": ids,
            "unstable_prob": test_prob,
            "stable_prob": 1.0 - test_prob,
        }
    )

    # sample_submission 순서 보존
    final_sub = test_df[["id"]].merge(pred_df, on="id", how="left")

    # 기존 파일이 있으면 덮어쓰지 않고 submission01.csv, submission02.csv ... 순으로 저장
    base_path = os.path.join(args.output_dir, "submission.csv")
    if not os.path.exists(base_path):
        sub_path = base_path
    else:
        n = 1
        while True:
            candidate = os.path.join(args.output_dir, f"submission{n:02d}.csv")
            if not os.path.exists(candidate):
                sub_path = candidate
                break
            n += 1

    final_sub.to_csv(sub_path, index=False)
    print(f"[*] 완료. submission 생성됨: {sub_path}")


# ============================================================
# 7. argparse
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-view CNN baseline for stability prediction")

    # path
    parser.add_argument("--train_csv", type=str, default="./train.csv")
    parser.add_argument("--dev_csv", type=str, default="./dev.csv")
    parser.add_argument("--test_csv", type=str, default="./sample_submission.csv")
    parser.add_argument("--train_dir", type=str, default="./train")
    parser.add_argument("--dev_dir", type=str, default="./dev")
    parser.add_argument("--test_dir", type=str, default="./test")
    parser.add_argument("--output_dir", type=str, default="./outputs_two_view_cnn")

    # crop / resize
    # 현재 렌더 포맷 기준으로 front/top의 중심 구조물이 들어오도록 잡은 ROI
    parser.add_argument("--front_roi", type=float, nargs=4, default=[0.24, 0.10, 0.76, 0.96])
    parser.add_argument("--top_roi", type=float, nargs=4, default=[0.28, 0.28, 0.72, 0.72])
    parser.add_argument("--front_h", type=int, default=256)
    parser.add_argument("--front_w", type=int, default=160)
    parser.add_argument("--top_h", type=int, default=160)
    parser.add_argument("--top_w", type=int, default=160)
    parser.add_argument("--roi_shift", type=float, default=0.015)   # 0.03 → 0.015: jitter 과도 시 Train/Val 분포 불일치 심화
    parser.add_argument("--roi_scale", type=float, default=0.03)    # 0.05 → 0.03: 동일 이유

    # train
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)

    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=4e-4)  # 1e-4 → 4e-4: L2 정규화 강화
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--base_channels", type=int, default=104)
    parser.add_argument("--encoder_dropout", type=float, default=0.15)   # 0.05 → 0.15: 인코더 정규화 강화
    parser.add_argument("--head_dropout", type=float, default=0.6)      # 0.35 → 0.6: head 과적합 억제 강화
    # ReduceLROnPlateau 관련 인수 제거 → CosineAnnealingLR 사용
    # (patience=4 기반 step-down 스케줄러는 요동치는 Val LogLoss에서 잘못된 시점에 LR을 낮춰 불안정을 심화함)
    parser.add_argument("--early_stopping", type=int, default=20)   # 12 → 20: cosine 스케줄에서는 충분한 patience 필요
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--dev_val_n", type=int, default=20,
        help="dev에서 검증용으로 홀드아웃할 샘플 수 (나머지는 학습에 포함)"
    )
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--val_tta", action="store_true", default=False)
    parser.add_argument("--test_tta", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
