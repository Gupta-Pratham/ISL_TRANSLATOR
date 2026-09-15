from pathlib import Path
import sys
import time

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.dataset import create_dataloader
from src.models import (
    LSTMClassifier,
    GRUClassifier,
    TCNClassifier,
    TransformerClassifier,
)


# ============================================================
# PROJECT PATHS
# ============================================================

TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"
LANDMARK_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks_preprocessed"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

INPUT_SIZE = 150
NUM_CLASSES = 59
BATCH_SIZE = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WARMUP_RUNS = 20
NUM_TIMED_RUNS = 100


# ============================================================
# MODEL CHECKPOINTS
# ============================================================

MODEL_CONFIGS = {
    "BiLSTM": {
        "class": LSTMClassifier,
        "checkpoint": CHECKPOINT_DIR / "lstm_baseline_best.pt",
    },
    "BiGRU": {
        "class": GRUClassifier,
        "checkpoint": CHECKPOINT_DIR / "gru_baseline_best.pt",
    },
    "TCN": {
        "class": TCNClassifier,
        "checkpoint": CHECKPOINT_DIR / "tcn_baseline_best.pt",
    },
    "Transformer": {
        "class": TransformerClassifier,
        "checkpoint": CHECKPOINT_DIR / "transformer_best.pt",
    },
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model):
    """
    Approximate parameter storage size in MB.
    All our model parameters are float32.
    """
    param_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    buffer_bytes = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )

    total_bytes = param_bytes + buffer_bytes

    return total_bytes / (1024 ** 2)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return model


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# MEASURE INFERENCE LATENCY
# ============================================================

def benchmark_model(model, sequence, padding_mask):
    model.eval()

    sequence = sequence.to(DEVICE)
    padding_mask = padding_mask.to(DEVICE)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    with torch.no_grad():
        for _ in range(NUM_WARMUP_RUNS):
            _ = model(sequence, padding_mask)

    synchronize()

    # --------------------------------------------------------
    # Timed inference
    # --------------------------------------------------------

    timings = []

    with torch.no_grad():

        for _ in range(NUM_TIMED_RUNS):

            synchronize()

            start = time.perf_counter()

            _ = model(sequence, padding_mask)

            synchronize()

            end = time.perf_counter()

            timings.append((end - start) * 1000)

    timings = np.array(timings)

    mean_latency = timings.mean()
    median_latency = np.median(timings)
    std_latency = timings.std()
    min_latency = timings.min()
    max_latency = timings.max()

    return {
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ISL TEMPORAL MODEL EFFICIENCY BENCHMARK")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Input features/frame: {INPUT_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Warm-up runs: {NUM_WARMUP_RUNS}")
    print(f"Timed runs: {NUM_TIMED_RUNS}")
    print()

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    if not LANDMARK_DIR.exists():
        raise FileNotFoundError(
            f"Landmark directory not found:\n{LANDMARK_DIR}"
        )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    dataset, loader = create_dataloader(
        TEST_CSV,
        batch_size=BATCH_SIZE,
        shuffle=False,
        landmark_dir=LANDMARK_DIR,
    )

    print(f"Test samples: {len(dataset)}")
    print()

    # --------------------------------------------------------
    # Select one representative test sequence
    #
    # We benchmark all models on EXACTLY the same sequence.
    # --------------------------------------------------------

    sequence, label = dataset[0]

    sequence = sequence.unsqueeze(0)

    padding_mask = torch.zeros(
        1,
        sequence.shape[1],
        dtype=torch.bool
    )

    print(f"Benchmark sequence shape: {tuple(sequence.shape)}")
    print(f"Benchmark class ID: {label}")
    print()

    # --------------------------------------------------------
    # Benchmark every model
    # --------------------------------------------------------

    results = []

    for model_name, config in MODEL_CONFIGS.items():

        print("-" * 70)
        print(f"Benchmarking: {model_name}")
        print("-" * 70)

        checkpoint_path = config["checkpoint"]

        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Skipping...")
            print()
            continue

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = config["class"](
            input_size=INPUT_SIZE,
            num_classes=NUM_CLASSES,
        )

        model = load_checkpoint(
            model,
            checkpoint_path
        )

        model = model.to(DEVICE)
        model.eval()

        # ----------------------------------------------------
        # Parameter statistics
        # ----------------------------------------------------

        parameters = count_parameters(model)
        model_size = get_model_size_mb(model)

        # ----------------------------------------------------
        # GPU memory before benchmark
        # ----------------------------------------------------

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latency = benchmark_model(
            model,
            sequence,
            padding_mask
        )

        # ----------------------------------------------------
        # GPU peak memory
        # ----------------------------------------------------

        if DEVICE.type == "cuda":
            peak_memory_mb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )
        else:
            peak_memory_mb = np.nan

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        fps = 1000.0 / latency["mean_latency_ms"]

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "model": model_name,
            "parameters": parameters,
            "parameters_millions": parameters / 1e6,
            "model_size_mb": model_size,
            "mean_latency_ms": latency["mean_latency_ms"],
            "median_latency_ms": latency["median_latency_ms"],
            "std_latency_ms": latency["std_latency_ms"],
            "min_latency_ms": latency["min_latency_ms"],
            "max_latency_ms": latency["max_latency_ms"],
            "inference_fps": fps,
            "peak_gpu_memory_mb": peak_memory_mb,
        }

        results.append(result)

        print(f"Parameters       : {parameters:,}")
        print(f"Parameters (M)   : {parameters / 1e6:.3f}")
        print(f"Model size       : {model_size:.3f} MB")
        print(f"Mean latency     : {latency['mean_latency_ms']:.3f} ms")
        print(f"Median latency   : {latency['median_latency_ms']:.3f} ms")
        print(f"Std latency      : {latency['std_latency_ms']:.3f} ms")
        print(f"Min latency      : {latency['min_latency_ms']:.3f} ms")
        print(f"Max latency      : {latency['max_latency_ms']:.3f} ms")
        print(f"Inference FPS    : {fps:.2f}")

        if DEVICE.type == "cuda":
            print(f"Peak GPU memory  : {peak_memory_mb:.2f} MB")

        print()

        # Release model before next one
        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_csv = RESULTS_DIR / "model_efficiency_benchmark.csv"

    results_df.to_csv(
        output_csv,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("=" * 70)
    print("FINAL EFFICIENCY COMPARISON")
    print("=" * 70)

    if len(results_df) > 0:

        display_columns = [
            "model",
            "parameters_millions",
            "model_size_mb",
            "mean_latency_ms",
            "inference_fps",
            "peak_gpu_memory_mb",
        ]

        print(
            results_df[display_columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )

    print()
    print(f"Saved results to:")
    print(output_csv)
    print()
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()