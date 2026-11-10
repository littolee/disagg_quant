# DGX Spark 分离式量化实验

本仓库保存 Qwen3-0.6B 的单台 DGX Spark 复现实验代码：QADD 训练适配与加速、NVFP4 PD shared 模型导出、双 vLLM 进程，以及通过 CUDA IPC 在同卡复制真实 paged KV 的实验连接器。研究背景见 [论文](https://arxiv.org/abs/2609.26333)和[作者仓库](https://github.com/IST-DASLab/disaggregated-quantization)。

## 结论和范围

- GB10、vLLM 0.33.0、CUDA Graph；280 条 MMLU-Pro 输入、并发 8、固定输出 128 token、预热后测三轮：PD + IPC 中位数 1433.05 token/s，BF16 1006.95 token/s，PD + NIXL 989.00 token/s。
- 这是特定负载的端到端吞吐：IPC 比 BF16 高约 42%，单请求 1 token 延迟仍更高。权重是 P 为 W4A4、D 为 W4A16 的 PD shared，**不是 LUT3**。
- 训练使用 fake quantization / STE 和 BF16 F.linear；没有声称训练 GEMM 全部执行原生 NVFP4。主要对照、完整精度、延迟和限制见 `reports/`。
- 本仓库是实验归档；脚本没有自动触发训练、停止进程或启动服务。部分脚本保留当次实验的固定端口、模型路径和输出路径，执行前请阅读。

## 本机环境与依赖

脚本中的项目路径为 `/home/litto/work/disagg_quant`，原始实验工作区为 `/home/litto/work/disagg_quan`；模型位于 `/home/litto/work/models/Qwen/Qwen3-0.6B`。模型、数据、检查点、日志、缓存、编译依赖、虚拟环境及作者上游源码均不在仓库内。

在当前 Spark 上执行 `bash scripts/link_local_runtime.sh`，会在仓库内建立指向旧工作区的本地链接；已有路径不会被覆盖，同时建立本地 `logs/`、`results/`。这些内容都被 Git 忽略。注意：训练脚本写入链接指向的原始缓存或检查点；测试副本时请先调整相应参数。

上游源码需位于 `disaggregated-quantization-main/`，与当前训练生成器适配的原始归档为旧工作区 `upstream.zip`，SHA-256：`eca1c7951249044f910a861ec82670541560cf47ae725063e715eb7f805f8641`。该目录上游 QAD Python/Shell 文件已与本机归档比对；不包含在本仓库。更换上游版本时应重新检查生成器的精确替换点和差异。环境通过旧工作区 `.venv`（训练）和 `.venv-eval`（评测）链接使用；`configs/requirements-lock.txt` 是历史环境快照，不保证从零安装可复现。

## 核心代码

- `scripts/train_memory.py`、`qad_memory_generated.py`、`cache_weight_views.py`、`reuse_distillation.py`：训练入口、实际训练版本及内存/损失优化。`make_fast_train.py` 和 `make_memory_train.py` 从上游源码重新生成对应实现，重新生成会覆盖仓库里的生成文件。
- `scripts/ipc_graph_servers.sh`、`ipc_live_proxy.py`：P/D vLLM 服务及代理；服务脚本只启动 P/D，代理需另行启动。请先检查 18140、18240、18041、5640、5641 和 19041 端口，避免干扰已有服务。
- `scripts/ipc_inspect_connector.py`、`ipc_nixl_connector.py`、`ipc_kv_copy.py`：真实 KV 张量句柄导出、NIXL 控制面/IPC 数据复制及完成通知。配置依赖单 GPU、TP=1、单 KV group 与相同 KV 布局；关闭 NIXL hash 后必须自行核实兼容性。IPC 本机 pickle 服务仅适用于可信的隔离实验。
- `scripts/check_ipc_kv_batch.py`、`check_real_vllm_ipc_kv.py`、`compare_pd_export_weights.py`：KV 复制、真实布局和共享权重检查。
- `scripts/ipc_graph_perf.py`、`bf16_graph_perf.py`、`nixl_graph_perf.py`：吞吐测试；`eval_ipc_vllm.py`、`eval_pd_vllm.py`、`eval_vllm_spark.py`：任务精度评测。
- `scripts/trace_latency_client.py`、`ipc_profile_connector.py`、`analyze_latency_trace.py`、`ipc_kv_copy_profile.py`：代理、KV 和单请求延迟剖析。历史 profile 脚本依赖对应的日志，报告保存在 `reports/latency-breakdown.md`。

## 安全检查和使用限制

先准备上述本机链接，再运行：

```bash
cd /home/litto/work/disagg_quant
python3 -m compileall -q scripts
.venv-eval/bin/python scripts/check_ipc_kv_batch.py
```

GPU IPC 检查会启动 CUDA 子进程；其他训练、评测与服务脚本会实际占用设备或写入运行目录。脚本中的 fallback、hash 绕过、KV 生命周期、取消及并发行为尚未达到通用生产连接器的保证，使用前应审查并验证。

## 实验记录

`reports/training-speed.md`、`reports/pd-throughput.md`、`reports/latency-breakdown.md` 是从旧工作区复制的历史报告，报告内的绝对路径及时间描述保留原状，指向旧实验位置，并非当前服务状态。小型 JSON 是对应的实验摘要；完整结果、检查点和论文源码需从本机工作区或原始数据恢复。
