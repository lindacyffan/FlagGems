# Protenix Ascend 910C Optimization

This repository contains the NPU attention patches and evaluation workflow used
to run Protenix on Ascend 910C with an FP32 storage path.

## Optimization path

- `src/pse_patch_v1.py` routes TriangleAttention to NPU fusion attention with a position-specific scale/bias tensor and a boolean row mask.
- `src/apb_pse_patch_v2.py` routes the 4D and 3D attention primitives used by atom-pair bias to NPU flash attention.
- `src/triatt_npu_patch.py` adapts TriangleAttention layouts, row masking, and loop tiling for NPU execution.
- `src/tm_npu_patch.py` fuses parts of TriangleMultiplicativeUpdate while keeping BF16 matrix multiplication disabled for the FP32 comparison path.
- `src/flagos_910c_launcher.py` installs the selected patches before the Protenix CLI starts.

The implementation intentionally keeps model parameters in FP32 and disables TF32, AMP mixed precision, and shared diffusion caching for controlled cross-platform comparisons.

## Same-workload result

Both runs used protein `7WUX`, seed `101`, `N_msa=1302`, `N_token=1170`, `N_atom=9094`, cycle `10`, 200 diffusion steps, one sample, FP32 storage, TF32 disabled, and caching disabled.

### Performance

| Route | Forward | Total |
|---|---:|---:|
| H100 native FP32, TF32 off | 116.53 s | 124.31 s |
| Ascend 910C + PSE, historical idle-NPU three-run mean | 204.43 s | not recorded |

Under the idle-NPU timing condition, the 910C+PSE route is about 1.75x slower
than H100, or about 57% of H100 speed.

### Native TriAttention microbenchmark

Measured on Ascend 910C with `B=1, S=693, D=32`, FP16 Q/K/V/output, and FP32 biases. Efficiency is the CANN latency divided by the native-operator latency; lower latency is better.

| Heads | Native TriAttention | CANN | Efficiency vs CANN | Slowdown |
|---:|---:|---:|---:|---:|
| H1 | 0.060720 ms | 0.0490 ms | 80.7% | 1.24x |
| H2 | 0.075691 ms | 0.0435 ms | 57.5% | 1.74x |
| H4 | 0.093529 ms | 0.0473 ms | 50.6% | 1.98x |
| H8 | 0.122878 ms | 0.0563 ms | 45.8% | 2.18x |

This is an FP16 single-operator result only. Its real Protenix replay still fails the correctness gate (`max_abs_diff ~= 7.20`), so it has no valid end-to-end performance result.

### Accuracy

| Metric | H100 | Ascend 910C + PSE | Relative difference |
|---|---:|---:|---:|
| pLDDT | 90.6176 | 90.3770 | 0.266% |
| pTM | 0.925347 | 0.924468 | 0.095% |
| ipTM | 0.929381 | 0.928870 | 0.055% |
| Ranking | 0.928575 | 0.927990 | 0.063% |

All confidence differences are below 0.27%, satisfying the 5% target. Performance
and accuracy were collected from separate valid runs; a later resource-constrained
diagnostic timing of 368.39 s is not used as the headline result.

## Reproduce

Copy the files from `src/` into a Protenix checkout, then run:

```bash
PROTENIX_ROOT=/path/to/Protenix \
NPU_ID=8 \
OUTPUT_NAME=output_910c_same_msa_fp32_tf32off_pse \
bash benchmarks/run_same_config_pse.sh
```

The script expects the CANN 8.5 environment and the Protenix checkpoint and input file to already be installed. Use an idle NPU for performance measurements.
