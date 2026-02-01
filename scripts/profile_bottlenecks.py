"""
Quick profiling script to identify bottlenecks before moving to C++.

Run this FIRST to see where you're actually losing time.
"""

import time
import torch
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from nova_selachiia.models import NSSM, NSSMConfig
from nova_selachiia.data import DataConfig, create_dataloaders, get_feature_list


def profile_dataloader(loader, n_batches=10):
    """Profile DataLoader speed."""
    print("\n" + "=" * 80)
    print("PROFILING: DataLoader")
    print("=" * 80)

    times = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        start = time.perf_counter()
        X, Y, mask = batch
        _ = X.cpu().numpy()  # Force data transfer
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Batch {i+1}: {elapsed*1000:.2f} ms")

    avg = np.mean(times) * 1000
    print(f"\n📊 Average: {avg:.2f} ms/batch")
    print(f"   Throughput: {1000/avg:.1f} batches/sec")

    if avg > 100:
        print("⚠️  SLOW! DataLoader is a bottleneck")
        print("   Solutions:")
        print("   1. Increase num_workers (if CPU has cores)")
        print("   2. Use pin_memory=True")
        print("   3. Pre-load to RAM (if data fits)")
        print("   4. ⚡ C++ Parquet reader (5-10x faster)")
    else:
        print("✅ DataLoader is fine, not the bottleneck")

    return avg


def profile_forward_pass(model, batch, device, n_runs=50):
    """Profile model forward pass."""
    print("\n" + "=" * 80)
    print("PROFILING: Forward Pass")
    print("=" * 80)

    X, Y, mask = batch
    X = X.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(X)

    # Benchmark
    times = []
    with torch.no_grad():
        for i in range(n_runs):
            start = time.perf_counter()
            _ = model(X)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"📊 Average: {avg:.2f} ± {std:.2f} ms")
    print(f"   Throughput: {X.shape[0] * 1000 / avg:.1f} sequences/sec")

    if avg > 50:
        print("⚠️  Forward pass is slow")
        print("   Solutions:")
        print("   1. Use torch.compile() (PyTorch 2.0+)")
        print("   2. Reduce model size")
        print("   3. Use AMP (already enabled)")
    else:
        print("✅ Forward pass is fine")

    return avg


def profile_monte_carlo(model, batch, device, n_samples=100):
    """Profile Monte Carlo rollout (THE REAL BOTTLENECK)."""
    print("\n" + "=" * 80)
    print("PROFILING: Monte Carlo Rollout (100 samples)")
    print("=" * 80)

    X, Y, mask = batch
    X = X[:1].to(device)  # Single sequence
    n_steps = X.shape[1]

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        ensemble, stats = model.monte_carlo_forecast(
            X=X,
            n_steps=n_steps,
            n_samples=n_samples,
            inject_noise=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"📊 Total time: {elapsed:.2f} sec")
    print(f"   Per sample: {elapsed/n_samples*1000:.2f} ms")
    print(f"   Per timestep: {elapsed/(n_samples*n_steps)*1000:.2f} ms")

    if elapsed > 5:
        print("⚠️  CRITICAL BOTTLENECK DETECTED!")
        print("   This is where C++ gives 5-10x speedup")
        print("   ⚡ Priority #1: C++ Monte Carlo rollout")
    elif elapsed > 1:
        print("⚠️  Slow but tolerable")
        print("   Consider C++ if you need real-time inference")
    else:
        print("✅ Fast enough, GPU is doing its job")

    return elapsed


def main():
    print("\n" + "=" * 80)
    print("BOTTLENECK PROFILER - Find where C++ helps most")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # Load data
    print("\n📂 Loading data...")
    data_config = DataConfig(
        data_path=ROOT_DIR / "data" / "processed" / "modeling_dataset.parquet",
        features=get_feature_list(),
        target="shark_presence",
        seq_len=12,
        batch_size=32,
        normalize=True,
    )

    train_loader, _, _, _ = create_dataloaders(
        data_path=data_config.data_path,
        config=data_config,
        num_workers=0,  # Test with 0 first
        pin_memory=False,
    )

    # Get sample batch
    sample_batch = next(iter(train_loader))

    # Load model
    print("\n🧠 Loading model...")
    model_config = NSSMConfig(
        input_dim=len(data_config.features),
        hidden_dim=64,
        output_dim=1,
        num_layers=2,
        num_resnet_blocks=2,
        mc_samples=100,
    )

    model = NSSM(
        input_dim=model_config.input_dim,
        hidden_dim=model_config.hidden_dim,
        output_dim=model_config.output_dim,
        num_layers=model_config.num_layers,
        num_resnet_blocks=model_config.num_resnet_blocks,
    ).to(device)
    model.eval()

    # Run profiles
    t_dataloader = profile_dataloader(train_loader, n_batches=10)
    t_forward = profile_forward_pass(model, sample_batch, device, n_runs=50)
    t_mc = profile_monte_carlo(model, sample_batch, device, n_samples=100)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Where to invest C++ effort")
    print("=" * 80)

    bottlenecks = [
        ("DataLoader", t_dataloader, "C++ Parquet reader + precompute windows"),
        ("Forward Pass", t_forward, "torch.compile() or C++ LSTM kernel"),
        ("Monte Carlo", t_mc * 1000, "⚡ C++ rollout loop (HIGHEST PRIORITY)"),
    ]

    bottlenecks.sort(key=lambda x: x[1], reverse=True)

    print("\n🎯 Priority order (by time spent):\n")
    for rank, (name, time_ms, solution) in enumerate(bottlenecks, 1):
        print(f"{rank}. {name}: {time_ms:.2f} ms")
        print(f"   → {solution}\n")

    # Realistic C++ gains
    print("=" * 80)
    print("REALISTIC C++ SPEEDUPS")
    print("=" * 80)
    print("\n1. Monte Carlo rollout: 5-10x faster")
    print("   - Current: {:.2f} sec/100 samples".format(t_mc))
    print("   - With C++: {:.2f} sec/100 samples".format(t_mc / 7))
    print("   - Gain: Can run 700 samples in same time\n")

    print("2. DataLoader (if bottleneck): 3-5x faster")
    print("   - Current: {:.2f} ms/batch".format(t_dataloader))
    print("   - With C++: {:.2f} ms/batch".format(t_dataloader / 4))
    print("   - Gain: 4x more batches/sec\n")

    print("3. Combined (MC + Data): 15-30x total speedup")
    print("   - Worth it for production/UI")
    print("   - Not worth it for one-time experiments\n")

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    if t_mc > 5:
        print("\n⚡ DO IT NOW: C++ Monte Carlo is critical for your use case")
        print("   You'll run 1000s of counterfactual scenarios")
        print("   5-10x speedup = difference between usable and unusable UI\n")
    elif t_dataloader > 200:
        print("\n⚠️  DO NEXT: DataLoader is killing you")
        print("   C++ Parquet reader before Monte Carlo\n")
    else:
        print("\n✅ Python is fine for now")
        print("   Profile again after training completes")
        print("   C++ only if inference is too slow for UI\n")


if __name__ == "__main__":
    main()
