# DGX Spark Qwen3-0.6B training speed comparison

Measured 2026-11-03, Asia/Shanghai. Both runs resumed the same step-50 state, used the same sample order, micro-batch 1, global batch 64, sequence limit 2048, NVFP4 quantization and distillation settings. One optimizer step was used as warmup; five paired steps (320 documents, 211,211 non-padding tokens) were timed including teacher/student forward, backward, gradient accumulation, optimizer and quantized-weight refresh. Startup, validation and checkpoint export were excluded.

| Metric | Original | Fast variant |
|---|---:|---:|
| Mean seconds per update | 118.4117 | 31.6955 |
| Effective tokens/second | 356.73 | 1332.76 |
| Speedup | 1.00x | 3.7359x |
| Peak PyTorch allocated memory | 16.29 GiB | 21.25 GiB |

Changes: remove right padding per microbatch; select supervised positions before the vocabulary projection/distillation loss; retain activations instead of using activation checkpointing. The 100M-token corpus budget, learning rate schedule and global batch remain configured as before.

This is a numerically different implementation variant. Removing padding changes which activations the running-max quantization observer sees. BF16 reduction grouping also changes. Loss compaction alone produced identical loss on one test document but a 0.9694% relative L2 difference across all 311 gradient tensors. Repeated original runs also diverged after their first update, so no claim of bit-identical trajectories is supported. Long-run task accuracy remains to be evaluated.

The five per-step pairs, memory and loss values are in `fast_paired_report.json`. Production fast continuation was started from step 50 in `/home/litto/work/disagg_quan/checkpoints-fast`, retaining the original checkpoint directory. Its first complete update took 34.78 seconds. Server logs: `/home/litto/work/disagg_quan/logs/train-fast-nvfp4.log`. Server configuration: `/home/litto/work/disagg_quan/configs/fast-continuation.json`.

The measured acceleration applies to the training updates in this sample, not a guarantee for total wall time including validation, export, I/O, other models or final accuracy.
