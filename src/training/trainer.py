from pathlib import Path
import json
import time
import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def train(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device,
    epochs,
    experiment_name,
    output_dir,
):
    output_dir = Path(output_dir)

    checkpoint_dir = output_dir / "checkpoints"
    result_dir = output_dir / "results" / experiment_name

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    history = []

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):

        start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        epoch_time = time.perf_counter() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_time": epoch_time,
        })

        print(
            f"Epoch [{epoch:02d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            checkpoint_path = (
                checkpoint_dir /
                f"{experiment_name}_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

    history_path = result_dir / "history.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    best_result = min(
        history,
        key=lambda x: x["val_loss"]
    )

    metrics = {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_result["val_loss"],
        "best_val_accuracy": best_result["val_accuracy"],
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
    }

    metrics_path = result_dir / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return history, metrics