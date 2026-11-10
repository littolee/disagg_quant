# 单请求 1 token 延迟分解（2026-11-10）

DGX Spark，Qwen3-0.6B，vLLM CUDA Graph。对两种方案使用完全相同的 96 条 MMLU-Pro 提示，先预热 16 次，再逐条串行发起、max_tokens=1、ignore_eos。每条提示加入变化的 nonce，避免整块前缀命中。独立的跟踪 D 服务与代理复用实际 PD Prefill 和 CUDA IPC 实现；BF16 为单进程直接 HTTP 服务。跟踪 P/D/客户端使用同机单调时钟，按 D 侧请求 ID 连接数据；96/96 成功匹配，无复制失败。测量会带来少量日志和计时开销。

客户端 PD 均值 35.187 ms、中位数 33.478 ms，BF16 均值 14.054 ms、中位数 13.768 ms；两者均值差 21.132 ms。每个阶段下表均为 96 条的算术平均，所有行互不重叠，可相加为总均值：

| 阶段 | 均值 ms | 说明 |
|---|---:|---|
| 客户端到代理 | 0.280 | 含本地 HTTP 与框架处理 |
| 代理准备 P 请求 | 0.061 | 请求解析、构建与会话 |
| P HTTP 往返 | 19.789 | 含 P 输入处理、排队、模型、序列化及返回，不是纯 GPU 执行 |
| 代理接收 P 到发起 D | 0.007 | KV 参数交接 |
| D 请求发出到 IPC 入口 | 2.787 | D HTTP、输入处理、调度等 |
| IPC 入口到复制准备 | 0.006 | 含映射查询 |
| 复制准备/提交 | 1.215 | CPU 指针与参数构建及提交 |
| 等待复制完成 | 0.841 | CPU 同步等待 |
| 复制结束到 D HTTP 返回 | 9.603 | 含 D 调度、模型生成、响应处理等 |
| 代理收到 D 到代理返回 | 0.136 | JSON 处理及关闭会话 |
| 代理返回到客户端收全 | 0.462 | 返回序列化与本地 HTTP |

P 往返比 BF16 客户端完整请求均值高 5.734 ms，D 相关区间合计约 14.45 ms，其余代理及边界约 0.95 ms，三者之和约为 21.13 ms。这里是墙钟区间的算术拆分，P 往返与 BF16 端到端的边界不完全相同，不能把 5.734 ms 全归因到 GPU 算子。

另一次 112 请求（含 16 预热）采集 vLLM 请求指标：P 调度到首 token 均值 16.715 ms，BF16 调度到首 token 均值 11.467 ms；D 排队到首次调度均值 5.039 ms，D 首次调度到首 token 均值 5.766 ms。vLLM 将这两个区间命名为 queue 和 prefill；单 token 请求的 decode 区间按定义为零。引擎区间包含 CPU、GPU、调度及同步，且与上表区间重叠，不可直接相加或称为纯 GPU 算子时长。

使用 CUDA Event 的另一次 96 请求中，复制设备事件区间 p50 1.518 ms、均值 1.617 ms、p95 2.231 ms；复制函数调用到同步完成 p50 2.015 ms、均值 2.107 ms。设备区间与 CPU 提交/等待重叠，事件时间可能包含流调度及干扰，不应相加。平均约 12.9 个逻辑 block。事件计时运行和墙钟拆分运行彼此独立，不应用一组的均值替换另一组。

结论：P/D 交接及 D 阶段带来大量额外固定开销；KV 复制自身的设备区间约 1.5 ms（中位数），并不是全部延迟差。P 本身的引擎区间也比 BF16 大约 5.2 ms，但还没有逐算子证据解释原因。当前代理会先让 P 生成一个 token，再将 KV 传给 D 生成最终的 token；直接返回 P 的 token 会改变首 token 所用的量化路径，需要先验证输出质量与一致性。

证据文件：joined-trace.json、pd-client.json、bf16-client.json、metrics-run/metric-deltas.json、gpu-run/summary.json；脚本在 ../../scripts/trace_latency_client.py、analyze_latency_trace.py、trace_latency_metrics_run.py、ipc_profile_connector.py 和 ipc_kv_copy_profile.py。原始日志在 ../../logs/ipc-profile-proxy-wall.log、ipc-profile-decode-wall.log 和 ipc-profile-decode.log。测试用服务已关闭，原 18140/18240 IPC 服务保留。
