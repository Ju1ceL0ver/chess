import json
import os
import random
from typing import Any, Dict, Optional, Tuple

import chess
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from models import ChessNNWithResiduals
from utitlities import PolicyDataset, _to_tensor_func, get_dicts


CONFIG = {
    "data_path": "data/processed.csv",
    "epochs": 5,
    "batch_size": 1024,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "test_split": 0.01,
    "seed": 42,
    "num_workers": 0,
    "device": None,
    "metrics_path": "metrics.json",
    "loss_plot_path": "loss_curve.png",
    "accuracy_plot_path": "accuracy_curve.png",
    "model_kwargs": {},
    "best_model_state_path": "checkpoints/best_model_state_dict.pt",
    "best_model_config_path": "checkpoints/best_model_config.json",
}


def resolve_device(configured: Optional[str] = None) -> torch.device:
    if configured is not None:
        if configured.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        if configured.startswith("mps") and not torch.backends.mps.is_available():
            print("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(configured)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_dataloaders(
    dataset: PolicyDataset,
    test_split: float,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Dataset must contain at least two samples to split.")
    if not (0.0 < test_split < 1.0):
        raise ValueError("test_split must be between 0 and 1.")

    test_size = int(dataset_size * test_split)
    if test_size == 0:
        test_size = 1
    if test_size >= dataset_size:
        test_size = dataset_size - 1

    train_size = dataset_size - test_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, test_subset = random_split(
        dataset,
        lengths=[train_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def _ensure_parent_dir(path: str) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_metric_plot(
    epochs: int,
    train_values: list,
    test_values: list,
    ylabel: str,
    output_path: Optional[str],
) -> None:
    if not output_path:
        return
    _ensure_parent_dir(output_path)
    x_axis = list(range(1, epochs + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, train_values, label="train", marker="o")
    plt.plot(x_axis, test_values, label="test", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} per epoch")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_model_artifacts(
    model: torch.nn.Module,
    model_kwargs: Dict[str, Any],
    state_path: Optional[str],
    config_path: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if not state_path and not config_path:
        return

    if state_path:
        _ensure_parent_dir(state_path)
        torch.save(model.state_dict(), state_path)

    if config_path:
        _ensure_parent_dir(config_path)
        payload = {
            "model_class": model.__class__.__name__,
            "model_kwargs": model_kwargs,
        }
        if metadata:
            payload["metadata"] = metadata
        with open(config_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    desc: Optional[str] = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    iterator = tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True)

    for features, targets in iterator:
        features = features.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits, _ = model(features)
                loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_examples += batch_size

    average_loss = total_loss / total_examples if total_examples else 0.0
    accuracy = total_correct / total_examples if total_examples else 0.0
    return average_loss, accuracy


def main(config: dict = CONFIG) -> None:
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = resolve_device(config.get("device"))
    pin_memory = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    df = pd.read_csv(config["data_path"])
    required_columns = {"FEN", "Move"}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(sorted(required_columns - set(df.columns)))
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.copy()
    df["FEN"] = df["FEN"].astype(str)
    df["Move"] = df["Move"].astype(str)

    move_to_idx, _ = get_dicts()
    dataset = PolicyDataset(df, ("FEN", "Move"), _to_tensor_func, move_to_idx)

    train_loader, test_loader = create_dataloaders(
        dataset=dataset,
        test_split=config["test_split"],
        batch_size=config["batch_size"],
        seed=config["seed"],
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )

    model_kwargs_config = config.get("model_kwargs")
    if model_kwargs_config is None:
        model_kwargs_config = {}
    if not isinstance(model_kwargs_config, dict):
        raise TypeError("config['model_kwargs'] must be a dictionary if provided.")
    model_kwargs: Dict[str, Any] = dict(model_kwargs_config)
    model = ChessNNWithResiduals(**model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_test_acc = float("-inf")
    best_test_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            desc=f"Epoch {epoch} [train]",
        )
        test_loss, test_acc = run_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            desc=f"Epoch {epoch} [test]",
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.2%} | "
            f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.2%}"
        )

        if test_acc > best_test_acc:
            previous_best = best_test_acc
            best_test_acc = test_acc
            best_test_loss = test_loss
            best_epoch = epoch

            if previous_best == float("-inf"):
                print(
                    f"Initial best test accuracy: {test_acc:.2%}. Saving model checkpoint."
                )
            else:
                print(
                    f"Improved test accuracy from {previous_best:.2%} to {test_acc:.2%}. Saving model checkpoint."
                )

            state_path = config.get("best_model_state_path")
            config_path = config.get("best_model_config_path")
            save_model_artifacts(
                model=model,
                model_kwargs=model_kwargs,
                state_path=state_path,
                config_path=config_path,
                metadata={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                },
            )
            saved_parts = [path for path in (state_path, config_path) if path]
            if saved_parts:
                print("Saved model artifacts to: " + ", ".join(saved_parts))

    save_metric_plot(
        epochs=config["epochs"],
        train_values=train_losses,
        test_values=test_losses,
        ylabel="Loss",
        output_path=config.get("loss_plot_path"),
    )
    save_metric_plot(
        epochs=config["epochs"],
        train_values=train_accuracies,
        test_values=test_accuracies,
        ylabel="Accuracy",
        output_path=config.get("accuracy_plot_path"),
    )

    if config.get("metrics_path"):
        metrics = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
        }
        with open(config["metrics_path"], "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"Metrics saved to {config['metrics_path']}")

    print("Training complete.")
    print("train_losses =", train_losses)
    print("test_losses =", test_losses)
    print("train_accuracies =", train_accuracies)
    print("test_accuracies =", test_accuracies)
    if best_epoch > 0:
        print(
            f"Best test accuracy {best_test_acc:.2%} "
            f"achieved at epoch {best_epoch} with loss {best_test_loss:.4f}."
        )


if __name__ == "__main__":
    main()
