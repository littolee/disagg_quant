# Qwen3-0.6B 单卡 PD 性能实验（2026-11-09）

实验条件：GB10，vLLM 0.33.0，MMLU-Pro 280 条问题，无思考模式；吞吐测试固定生成 128 token、ignore_eos、并发 8，两个预热批次和三轮正式测试。每条请求加入变化的 nonce；测量按端到端墙钟时间计算输出 token/s。各服务最大并发 8、2 GiB KV 缓存，均启用 CUDA Graph，BF16 与 PD 分开运行。PD 为两个进程：P 使用训练后的 NVFP4 W4A4，D 使用共享权重的 NVFP4 W4A16。以下为三轮中位数。

| 配置 | token/s | 280 请求耗时 | 单请求首 token p50（32 请求，各轮 p50 的中位数） |
| --- | ---: | ---: | ---: |
| BF16 | 1006.95 | 35.59 s | 13.46 ms |
| PD + NIXL | 989.00 | 36.24 s | 68.82 ms |
| PD + CUDA IPC | 1433.05 | 25.01 s | 34.72 ms |

PD + IPC 对 BF16 吞吐提高 42.3%，对 PD + NIXL 提高 44.9%；首 token 延迟仍高于 BF16。该结果适用于上述负载，不能外推至其他长度、并发和模型。

IPC 通过 NIXL 控制面协商与释放远端 block，在同一设备映射真正的 vLLM KV，并批量复制分页 block。固定长度性能实验 968 次 IPC_COPY_COMPLETED，未发现 IPC_COPY_FAILED。开启 IPC_VERIFY_KV 后，真实请求逐层比较 28 层 KV；测试及 280 题评测共 281 次验证成功，未见复制失败。此实验实现限定单卡、TP=1、单 KV group 和匹配布局；非支持情形会回退 NIXL，尚非通用或生产实现。P/D 使用相同量化权重，但激活量化配置不同，禁用了 NIXL 兼容性 hash，实验前需自行确认布局等兼容。

同一 280 题、5-shot MMLU-Pro no-think：Graph+IPC 53/280，重复一次及开启 KV 校验时相同；Graph+NIXL 52/280，二者提取答案 276/280 相同，正确性仅 1 题不同。旧 eager+IPC 60/280、eager+NIXL 62/280；Graph 和 eager 的答案存在变化，原因尚未确定，不能归结为 IPC，也不能宣称模型质量完全无损。重复运行可能命中前缀缓存；样本规模不足以作全量质量结论。

原始 JSON：results/ipc-graph-benchmark/、results/bf16-graph-benchmark/、results/nixl-graph-benchmark/ 和 results/ipc-vllm-eval/nothink/mmlu_pro/。日志：logs/ipc-graph-perf-decode.log、logs/ipc-graph-verified-decode.log、logs/nixl-graph-benchmark.log。服务脚本 scripts/ipc_graph_servers.sh；连接器 scripts/ipc_nixl_connector.py 与 scripts/ipc_kv_copy.py。
