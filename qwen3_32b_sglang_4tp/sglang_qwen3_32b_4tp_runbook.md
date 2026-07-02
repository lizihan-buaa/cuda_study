# Qwen3-32B BF16 4TP SGLang Runbook

本文记录在当前服务器上使用 4 张 RTX 5090 跑 Qwen3-32B dense BF16 的启动、停止、benchmark 和常见问题处理流程。

## 实验条件

- 服务器 GPU：8 x NVIDIA GeForce RTX 5090，每张约 32 GB 显存。
- 本次使用 GPU：`0,1,2,3`。
- 并行策略：Tensor Parallel，`TP=4`。
- 模型：`/DaTa/lizihan/weight/Qwen3-32B`。
- 推理框架：SGLang，环境 `/home/lizihan/sglang_cp_env`。
- Python：`/home/lizihan/sglang_cp_env/bin/python`。
- Torch：`2.11.0+cu128`，CUDA runtime 12.8。
- Nsight Systems：`/usr/local/cuda-12.8/bin/nsys`。
- 实际可跑通 backend：
  - `--attention-backend triton`
  - `--sampling-backend pytorch`
- 未采用默认 FlashInfer attention 的原因：
  - 当前 RTX 5090 / SM120 + CUDA 12.8 环境下，FlashInfer 会报 `SM 12.x requires CUDA >= 12.9` 或误报 `FlashInfer requires GPUs with sm75 or higher`。

## 启动服务

注意：必须清理代理环境，否则 SGLang 服务端内部访问 `127.0.0.1` 也可能走 Squid 代理并失败。

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
CC=/home/lizihan/bin/gcc-python312 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/lizihan/sglang_cp_env/bin/python -m sglang.launch_server \
  --model-path /DaTa/lizihan/weight/Qwen3-32B \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.88 \
  --context-length 32768 \
  --disable-radix-cache \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-custom-all-reduce \
  --skip-server-warmup \
  --log-level info
```

服务 ready 的标志：

```text
The server is fired up and ready to roll!
Uvicorn running on http://0.0.0.0:30000
```

本机验证：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
curl -sS http://127.0.0.1:30000/v1/models
```

预期返回中应包含：

```text
/DaTa/lizihan/weight/Qwen3-32B
```

## 杀服务

优先使用进程名查询后终止：

```bash
pgrep -af 'sglang.launch_server|sglang serve|/home/lizihan/sglang_cp_env/bin/python'
```

如果服务在当前终端前台运行，直接：

```bash
Ctrl-C
```

如果服务在后台运行，按查到的主进程 PID 终止：

```bash
kill <sglang_main_pid>
```

如果普通 `kill` 后还有残留子进程，再使用：

```bash
pkill -f 'sglang.launch_server'
pkill -f '/home/lizihan/sglang_cp_env/bin/python.*sglang'
```

确认 GPU 0-3 已释放：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

正常停止后 GPU 0-3 应回到约 `4 MiB`。

## Benchmark 命令

本次只测 prefill，对比单并发和 10 并发。使用 `random-ids`，避免联网下载 ShareGPT。

重要：`--random-range-ratio 1` 必须加，否则输入长度会在 `1..4096` 之间随机，并不是固定 4096。

单并发：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
/home/lizihan/sglang_cp_env/bin/python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random-ids \
  --random-input-len 4096 \
  --random-output-len 1 \
  --random-range-ratio 1 \
  --num-prompts 10 \
  --request-rate inf \
  --max-concurrency 1 \
  --warmup-requests 1 \
  --flush-cache
```

10 并发：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
/home/lizihan/sglang_cp_env/bin/python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random-ids \
  --random-input-len 4096 \
  --random-output-len 1 \
  --random-range-ratio 1 \
  --num-prompts 10 \
  --request-rate inf \
  --max-concurrency 10 \
  --warmup-requests 1 \
  --flush-cache
```

本次初步结果：

| 并发 | Total input tokens | Input token throughput | Mean TTFT | P99 TTFT |
| --- | ---: | ---: | ---: | ---: |
| 1 | 40960 | 6026.85 tok/s | 677.36 ms | 817.97 ms |
| 10 | 40960 | 6347.20 tok/s | 3807.79 ms | 6405.29 ms |

## Nsight Systems profiling 建议

如果要抓模型 GPU kernel timeline，`nsys` 需要包住服务端进程，而不是只包住 benchmark 客户端。

一个可行方式是让 `nsys` 直接启动 SGLang 服务：

```bash
mkdir -p /home/lizihan/cuda_study/qwen3_32b_sglang_4tp/nsys

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost \
CC=/home/lizihan/bin/gcc-python312 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/usr/local/cuda-12.8/bin/nsys profile \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=graph \
  --sample=none \
  --cpuctxsw=none \
  -o /home/lizihan/cuda_study/qwen3_32b_sglang_4tp/nsys/qwen3_32b_4tp_sglang \
  /home/lizihan/sglang_cp_env/bin/python -m sglang.launch_server \
    --model-path /DaTa/lizihan/weight/Qwen3-32B \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.88 \
    --context-length 32768 \
    --disable-radix-cache \
    --attention-backend triton \
    --sampling-backend pytorch \
    --disable-custom-all-reduce \
    --skip-server-warmup \
    --log-level info
```

缺点：会把权重加载、CUDA graph capture 也录进去，trace 文件会很大。更干净的方式是用 `--capture-range=cudaProfilerApi` 或 NVTX range 控制采集窗口，但需要服务端在 benchmark 前后调用对应 profiler start/stop API 或在代码里打 NVTX range。

## 本次遇到的问题和解决方法

### 1. localhost 请求被代理劫持

现象：

```text
Squid ERROR 403
http://127.0.0.1:30000/model_info
```

原因：环境中设置了 `http_proxy/https_proxy`，SGLang 服务端内部访问 `127.0.0.1` 也走了代理。

解决：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
NO_PROXY=127.0.0.1,localhost ...
```

同时启动服务时加：

```bash
--skip-server-warmup
```

### 2. FlashInfer 与 RTX 5090 / CUDA 12.8 兼容问题

现象：

```text
SM 12.x requires CUDA >= 12.9
FlashInfer requires GPUs with sm75 or higher
```

原因：RTX 5090 是 Blackwell/SM120，当前环境 torch 是 cu128，FlashInfer 的架构探测和 JIT 路径在此组合下不稳定。

解决：不要使用默认 FlashInfer attention，改用：

```bash
--attention-backend triton
--sampling-backend pytorch
```

### 3. sglang-kernel CUDA 版本不匹配

现象：

```text
ImportError: libnvrtc.so.13: cannot open shared object file
```

原因：原环境中的 `sglang-kernel` 依赖 CUDA 13 动态库，但机器上主要是 CUDA 12.8。

解决：安装适配 wheel，并避免改动 torch：

```bash
/home/lizihan/sglang_cp_env/bin/python -m pip install \
  --force-reinstall --no-deps \
  --index-url https://docs.sglang.ai/whl/cu129/ \
  'sglang-kernel==0.4.3+cu129'
```

### 4. 缺少 FlashInfer 包

现象：

```text
ModuleNotFoundError: No module named 'flashinfer'
```

解决：只安装包本身，不安装依赖，避免 pip 顺手把 torch 升到 CUDA 13：

```bash
/home/lizihan/sglang_cp_env/bin/python -m pip install --no-deps flashinfer-python
```

### 5. Triton 编译缺 Python headers

现象：

```text
fatal error: Python.h: No such file or directory
fatal error: x86_64-linux-gnu/python3.12/pyconfig.h: No such file or directory
```

原因：系统没有安装 `python3.12-dev`，且当前用户没有免密 sudo。

解决：把 `python3.12-dev` 和 `libpython3.12-dev` 的 deb 解包到用户目录，然后用 gcc wrapper 指向本地 headers。

当前 wrapper：

```bash
/home/lizihan/bin/gcc-python312
```

内容：

```bash
#!/bin/sh
exec /usr/bin/gcc \
  -I/home/lizihan/local_python_headers/usr/include \
  -I/home/lizihan/local_python_headers/usr/include/python3.12 \
  -I/home/lizihan/local_python_headers/usr/include/x86_64-linux-gnu/python3.12 \
  "$@"
```

启动服务时设置：

```bash
CC=/home/lizihan/bin/gcc-python312
```

### 6. `random` benchmark 会联网下载 ShareGPT

现象：

```text
Network is unreachable
HEAD https://huggingface.co/datasets/anon8231489123/ShareGPT...
```

原因：`--dataset-name random` 会用 ShareGPT 内容采样 token。

解决：使用纯本地 token id：

```bash
--dataset-name random-ids
```

### 7. `random-ids` 默认不是固定输入长度

现象：设置 `--random-input-len 4096` 后，总输入 token 不是 `num_prompts * 4096`。

原因：默认 `--random-range-ratio 0`，实际长度在 `1..4096` 中随机。

解决：

```bash
--random-range-ratio 1
```

### 8. PCIe 4TP 的性能特征

当前 4 张 5090 之间没有 NVLink，4TP 会产生跨卡通信开销。长 prefill 下，10 并发主要提升吞吐不明显，但 TTFT 明显升高，这是预期现象。做 profiling 时应重点关注：

- attention kernel 时间；
- GEMM kernel 时间；
- NCCL all-reduce/all-gather 时间；
- CPU 调度间隙；
- CUDA graph replay 是否生效；
- chunked prefill 的切分粒度。

