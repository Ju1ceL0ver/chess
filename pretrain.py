from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

from models import ChessNNWithResiduals
from utitlities import PolicyIndexDataset, get_dicts


@dataclass
class PretrainConfig:
    data: List[Path]
    fen_column: str = "FEN"
    index_column: str = "Move"
    limit: Optional[int] = None
    epochs: int = 10
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    value_reg: float = 0.05
    val_split: float = 0.1
    seed: int = 69
    device: Optional[str] = None
    transformer: bool = False
    save_path: Path = Path("checkpoints/pretrained_policy.pt")
    load_checkpoint: Optional[Path] = None
    num_workers: int = 4
    pin_memory: bool = False
    topk: Sequence[int] = (1, 3)


# Настроить пути/параметры под свой датасет
CONFIG = PretrainConfig(data=[Path("/Users/aleksejzagorskij/chess/data/formatted.csv")])


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: Optional[str]) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        try:
            if mps_backend.is_available():  # type: ignore[union-attr]
                return torch.device("mps")
        except (AttributeError, RuntimeError):
            pass
    return torch.device("cpu")


def load_dataframe(
    paths: Sequence[Path],
    fen_column: str,
    index_column: str,
    limit: Optional[int],
    seed: int,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Файл {path} не найден")
        suffix = path.suffix.lower()
        if suffix in {".csv", ".csv.gz", ".csv.bz2", ".csv.zip", ".csv.xz"}:
            frame = pd.read_csv(path, usecols=[fen_column, index_column])
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path, columns=[fen_column, index_column])
        else:
            raise ValueError(
                f"Формат {path.suffix} не поддерживается. Используйте CSV или Parquet."
            )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[fen_column, index_column])

    if limit is not None and limit > 0 and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df[fen_column] = df[fen_column].astype(str)
    return df


def split_dataset(
    dataset: Dataset, val_split: float, seed: int
) -> Tuple[Dataset, Optional[Dataset]]:
    if val_split <= 0.0:
        return dataset, None

    val_size = int(len(dataset) * val_split)
    if val_size <= 0:
        return dataset, None

    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    return train_dataset, val_dataset


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    if logits.size(0) == 0:
        return 0.0
    k = max(1, min(k, logits.size(1)))
    _, indices = logits.topk(k, dim=1)
    correct = indices.eq(targets.view(-1, 1))
    return correct.any(dim=1).float().mean().item()


def train_epoch(
    model: ChessNNWithResiduals,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_reg: float,
    topk: Iterable[int],
) -> Tuple[float, float, float, dict]:
    topk = list(topk)
    primary_k = topk[0]
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_samples = 0
    acc_totals = {k: 0.0 for k in topk}

    progress = tqdm(loader, desc="train", leave=False, ncols=120)
    for features, targets in progress:
        batch_size = features.size(0)
        features = features.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        policy_logits, value = model(features)

        loss_policy = F.cross_entropy(policy_logits, targets)
        loss_value = value.pow(2).mean()
        loss = loss_policy + value_reg * loss_value
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            batch_metrics = {k: topk_accuracy(policy_logits, targets, k) for k in topk}

        total_loss += loss.item() * batch_size
        total_policy_loss += loss_policy.item() * batch_size
        total_value_loss += loss_value.item() * batch_size
        total_samples += batch_size
        for k in topk:
            acc_totals[k] += batch_metrics[k] * batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            top=f"{acc_totals[primary_k] / total_samples:.3f}",
        )

    mean_loss = total_loss / max(total_samples, 1)
    mean_policy = total_policy_loss / max(total_samples, 1)
    mean_value = total_value_loss / max(total_samples, 1)
    mean_acc = {k: acc_totals[k] / max(total_samples, 1) for k in topk}
    return mean_loss, mean_policy, mean_value, mean_acc


def evaluate(
    model: ChessNNWithResiduals,
    loader: Optional[DataLoader],
    device: torch.device,
    value_reg: float,
    topk: Iterable[int],
) -> Tuple[float, float, float, dict]:
    topk = list(topk)
    if loader is None:
        return 0.0, 0.0, 0.0, {k: 0.0 for k in topk}
    primary_k = topk[0]
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_samples = 0
    acc_totals = {k: 0.0 for k in topk}

    progress = tqdm(loader, desc="val", leave=False, ncols=120)
    with torch.no_grad():
        for features, targets in progress:
            batch_size = features.size(0)
            features = features.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.long)

            policy_logits, value = model(features)

            loss_policy = F.cross_entropy(policy_logits, targets)
            loss_value = value.pow(2).mean()
            loss = loss_policy + value_reg * loss_value

            batch_metrics = {k: topk_accuracy(policy_logits, targets, k) for k in topk}

            total_loss += loss.item() * batch_size
            total_policy_loss += loss_policy.item() * batch_size
            total_value_loss += loss_value.item() * batch_size
            total_samples += batch_size
            for k in topk:
                acc_totals[k] += batch_metrics[k] * batch_size

            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                top=f"{acc_totals[primary_k] / total_samples:.3f}",
            )

    mean_loss = total_loss / max(total_samples, 1)
    mean_policy = total_policy_loss / max(total_samples, 1)
    mean_value = total_value_loss / max(total_samples, 1)
    mean_acc = {k: acc_totals[k] / max(total_samples, 1) for k in topk}
    return mean_loss, mean_policy, mean_value, mean_acc


def load_model(config: PretrainConfig) -> ChessNNWithResiduals:
    model = ChessNNWithResiduals(use_transformer=config.transformer)
    if config.load_checkpoint is not None:
        if not config.load_checkpoint.exists():
            raise FileNotFoundError(f"Чекпоинт {config.load_checkpoint} не найден")
        state = torch.load(config.load_checkpoint, map_location="cpu")
        model.load_state_dict(state)
    return model


def main(config: PretrainConfig = CONFIG) -> None:
    set_seed(config.seed)

    topk = sorted({int(k) for k in config.topk if int(k) > 0})
    if not topk:
        topk = [1]

    move_to_index, _ = get_dicts()
    move_space_size = len(move_to_index)

    df = load_dataframe(
        paths=config.data,
        fen_column=config.fen_column,
        index_column=config.index_column,
        limit=config.limit,
        seed=config.seed,
    )

    dataset = PolicyIndexDataset(
        data=df,
        move_space_size=move_space_size,
        one_hot=False,
        fen_column=config.fen_column,
        index_column=config.index_column,
        move_to_index=move_to_index,
    )
    train_dataset, val_dataset = split_dataset(dataset, config.val_split, config.seed)

    device = resolve_device(config.device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=config.pin_memory,
            num_workers=config.num_workers,
        )
    else:
        val_loader = None

    model = load_model(config).to(device)
    print(f"Используем устройство: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    epoch_bar = tqdm(range(1, config.epochs + 1), desc="epochs", ncols=140)
    last_train_info = ""
    last_val_info = ""

    for epoch in epoch_bar:
        train_loss, train_policy, train_value, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            value_reg=config.value_reg,
            topk=topk,
        )
        train_info = (
            f"train_loss={train_loss:.4f} policy={train_policy:.4f} value={train_value:.4f} "
            + " ".join(f"top{k}={train_acc[k]:.3f}" for k in topk)
        )

        val_loss, val_policy, val_value, val_acc = evaluate(
            model,
            val_loader,
            device,
            value_reg=config.value_reg,
            topk=topk,
        )
        if val_loader is not None:
            val_info = (
                f"val_loss={val_loss:.4f} policy={val_policy:.4f} value={val_value:.4f} "
                + " ".join(f"top{k}={val_acc[k]:.3f}" for k in topk)
            )
        else:
            val_info = "val=NA"

        last_train_info = train_info
        last_val_info = val_info
        epoch_bar.set_postfix_str(f"{last_train_info} | {last_val_info}")

    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.save_path)
    print(f"Модель сохранена в {config.save_path}")


if __name__ == "__main__":
    main()
