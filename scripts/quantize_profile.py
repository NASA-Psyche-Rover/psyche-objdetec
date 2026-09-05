"""
Benchmarks MiDaS depth estimation across three inference backends:
    - PyTorch FP32  (baseline)
    - ONNX Runtime  FP32  (graph-optimized, no accuracy loss; used in current perception)
    - OpenVINO INT8 (quantized, Intel-optimized)

Measures per-inference latency distributions (mean, P95, P99) and model file-size compression ratios to evaluate the trade-off between
inference speed and model size for edge deployment on rover hardware.

Dependencies:
    pip install torch torchvision timm onnx onnxruntime openvino numpy

Outputs:
    - Terminal summary of latency and compression metrics
    - quantize_results.json saved to project root
"""

import time
import json
import shutil
import subprocess
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).parent.parent   
MODELS    = ROOT / "models"
RESULTS_F = ROOT / "quantize_results.json"

MIDAS_ONNX = MODELS / "midas_small.onnx"
MIDAS_OV   = MODELS / "midas_small_openvino"

N_WARMUP = 5    # excluded from timing
N_BENCH  = 50   # timed iterations

# MiDaS small expects 3 × 256 × 256 input
DUMMY_INPUT = np.random.rand(1, 3, 256, 256).astype(np.float32)


# Helpers
def pct(arr, p):
    return float(np.percentile(arr, p))

def file_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2
    return path.stat().st_size / 1024**2

def bench(fn, n_warmup=N_WARMUP, n_bench=N_BENCH):
    """Warm up then time fn() for n_bench iterations. Returns latency array in ms."""
    for _ in range(n_warmup):
        fn()
    lats = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        fn()
        lats.append((time.perf_counter() - t0) * 1000)
    return np.array(lats)

def print_row(name, lats):
    if lats is None:
        return
    print(f"  {name:<24}  mean={np.mean(lats):6.1f} ms   "
          f"P95={pct(lats,95):6.1f} ms   P99={pct(lats,99):6.1f} ms")


# Load MiDAS + export to ONNX
def load_midas():
    import torch
    print("[setup] loading MiDaS_small from torch.hub …")
    device    = torch.device("cpu")
    model     = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.to(device).eval()
    return model, device

def export_onnx(model):
    import torch
    if MIDAS_ONNX.exists():
        print(f"[export] ONNX already exists: {MIDAS_ONNX}")
        return

    print("[export] exporting MiDaS to ONNX …")
    MODELS.mkdir(exist_ok=True)
    dummy = torch.from_numpy(DUMMY_INPUT)
    torch.onnx.export(
        model, dummy, str(MIDAS_ONNX),
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"[export] saved → {MIDAS_ONNX}")

def export_openvino():
    if MIDAS_OV.exists():
        print(f"[export] OpenVINO already exists: {MIDAS_OV}")
        return

    if not MIDAS_ONNX.exists():
        print("[export] ONNX model not found — cannot export to OpenVINO.")
        return

    print("[export] converting ONNX → OpenVINO INT8 …")
    try:
        result = subprocess.run(
            [
                "ovc", str(MIDAS_ONNX),
                "--output_model", str(MIDAS_OV / "midas_small"),
                "--compress_to_fp16", "False",
            ],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[export] ovc failed:\n{result.stderr}")
            return
        print(f"[export] saved → {MIDAS_OV}")
    except FileNotFoundError:
        print("[export] ovc not found — install openvino: pip install openvino")


# PyTorch benchmark
def bench_pytorch(model):
    import torch
    print("[bench] PyTorch FP32 …")
    dummy = torch.from_numpy(DUMMY_INPUT)

    def run():
        with torch.no_grad():
            model(dummy)

    return bench(run)


# ONNX Runtime benchmark
def bench_onnx():
    if not MIDAS_ONNX.exists():
        print("[bench] ONNX model not found — skipping.")
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        print("[bench] onnxruntime not installed — skipping.")
        return None

    print("[bench] ONNX Runtime FP32 …")
    sess  = ort.InferenceSession(str(MIDAS_ONNX), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    def run():
        sess.run(None, {iname: DUMMY_INPUT})

    return bench(run)


# OpenVino benchmark 
def bench_openvino():
    if not MIDAS_OV.exists():
        print("[bench] OpenVINO model not found — skipping.")
        return None
    try:
        from openvino import Core
    except ImportError:
        print("[bench] openvino not installed — skipping.")
        return None

    xml_f = next(MIDAS_OV.glob("*.xml"), None)
    if xml_f is None:
        print(f"[bench] no .xml found in {MIDAS_OV} — skipping.")
        return None

    print("[bench] OpenVINO FP16 …")
    ie        = Core()
    model     = ie.read_model(str(xml_f))
    compiled  = ie.compile_model(model, "CPU")
    infer_req = compiled.create_infer_request()
    inp       = compiled.input(0)

    def run():
        infer_req.infer({inp: DUMMY_INPUT})

    return bench(run)


def main():
    MODELS.mkdir(exist_ok=True)

    # Load and export
    model, device = load_midas()
    export_onnx(model)
    export_openvino()

    # File sizes
    pt_mb = None   # MiDaS loads from torch.hub cache, not a local .pt
    onnx_mb = file_mb(MIDAS_ONNX) if MIDAS_ONNX.exists() else None
    ov_mb   = file_mb(MIDAS_OV)   if MIDAS_OV.exists()   else None

    print()
    if onnx_mb:
        print(f"[sizes] ONNX          : {onnx_mb:.2f} MB")
    if ov_mb:
        print(f"[sizes] OpenVINO      : {ov_mb:.2f} MB")
    if onnx_mb and ov_mb:
        print(f"[sizes] compression   : {onnx_mb/ov_mb:.2f}× (ONNX → OpenVINO)")

    # Latency benchmarks
    pt_lats   = bench_pytorch(model)
    onnx_lats = bench_onnx()
    ov_lats   = bench_openvino()

    # Summary
    print(f"\n{'='*65}")
    print("  MiDaS INFERENCE BENCHMARK SUMMARY")
    print(f"{'='*65}")
    print_row("PyTorch FP32",   pt_lats)
    print_row("ONNX FP32",      onnx_lats)
    print_row("OpenVINO FP16",  ov_lats)

    if pt_lats is not None and ov_lats is not None:
        delta_mean = np.mean(pt_lats) - np.mean(ov_lats)
        delta_p99  = pct(pt_lats, 99) - pct(ov_lats, 99)
        speedup    = np.mean(pt_lats) / np.mean(ov_lats)
        print(f"\n  Mean latency reduction  : {delta_mean:.1f} ms  ({speedup:.1f}× speedup)")
        print(f"  P99  latency reduction  : {delta_p99:.1f} ms")
        print(f"\n  Rover impact:")
        print(f"    PyTorch  → depth assessment every {np.mean(pt_lats):.0f} ms")
        print(f"    OpenVINO → depth assessment every {np.mean(ov_lats):.0f} ms")
        print(f"    At 10 cm/s rover speed, that reduces blind distance from "
              f"{np.mean(pt_lats)*0.1/1000*100:.2f} cm to "
              f"{np.mean(ov_lats)*0.1/1000*100:.2f} cm per inference cycle")

    print(f"{'='*65}\n")

    # Save results
    results = {
        "model": "MiDaS_small",
        "n_bench_iterations": N_BENCH,
        "file_sizes_mb": {
            "onnx_fp32":      round(onnx_mb, 3) if onnx_mb else None,
            "openvino_fp16":  round(ov_mb,   3) if ov_mb   else None,
            "compression_ratio_onnx_vs_ov": round(onnx_mb / ov_mb, 2) if (onnx_mb and ov_mb) else None,
        },
        "latency_ms": {
            "pytorch_fp32": {
                "mean": round(float(np.mean(pt_lats)),   2),
                "p95":  round(pct(pt_lats, 95),          2),
                "p99":  round(pct(pt_lats, 99),          2),
            } if pt_lats is not None else None,
            "onnx_fp32": {
                "mean": round(float(np.mean(onnx_lats)), 2),
                "p95":  round(pct(onnx_lats, 95),        2),
                "p99":  round(pct(onnx_lats, 99),        2),
            } if onnx_lats is not None else None,
            "openvino_fp16": {
                "mean": round(float(np.mean(ov_lats)),   2),
                "p95":  round(pct(ov_lats, 95),          2),
                "p99":  round(pct(ov_lats, 99),          2),
            } if ov_lats is not None else None,
        },
        "speedup_pytorch_vs_openvino": round(
            float(np.mean(pt_lats)) / float(np.mean(ov_lats)), 2
        ) if (pt_lats is not None and ov_lats is not None) else None,
    }

    RESULTS_F.write_text(json.dumps(results, indent=2))
    print(f"[saved] results → {RESULTS_F}")


if __name__ == "__main__":
    main()