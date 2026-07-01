# Nsight Compute Reduce Benchmark

本文记录 `src/ampere/reduce/reduce.cu` 的编译、运行和 Nsight Compute Memory Throughput 测试流程。

## 1. 选择空闲 GPU

每次测试前先检查 GPU 进程占用：

```bash
nvidia-smi
```

选择 `Processes` 表中没有进程的 GPU。注意：设置 `CUDA_VISIBLE_DEVICES=<physical_gpu_id>` 后，程序内部看到的设备编号会被重映射成 `device=0`，这是正常现象。

本次测试时间：2026-07-01 16:13。`nvidia-smi` 显示 GPU 0-3 有 `sglang` 进程，GPU 4-7 空闲，因此选择物理 GPU 4：

```bash
export CUDA_VISIBLE_DEVICES=4
```

## 2. 编译

RTX 5090 是 Blackwell 架构，本机 CUDA 12.8 支持 `sm_120`：

```bash
mkdir -p build/ampere/reduce

/usr/local/cuda/bin/nvcc -O3 -lineinfo -arch=sm_120 \
  src/ampere/reduce/reduce.cu \
  -o build/ampere/reduce/reduce
```

`-lineinfo` 用于让 Nsight Compute 把指标关联回源码行，同时基本不影响优化。

## 3. 先做正确性和 CUDA Event 基线测试

运行全部版本：

```bash
CUDA_VISIBLE_DEVICES=4 ./build/ampere/reduce/reduce all 67108864 20
```

参数含义：

- `all`：依次运行 `v0/v1/v2/v3/v6/v7`
- `67108864`：输入元素个数，即 `1 << 26`
- `20`：每个 kernel 计时重复次数

本次输出：

```text
device=0 name=NVIDIA GeForce RTX 5090 sms=170 n=67108864 repeat=20 cpu_sum=67108864
v0   grid=262144 block= 256 avg_ms=  0.2751 effective_read_GB/s=   975.93 gpu_sum=67108864 cpu_sum=67108864 abs_err=0 rel_err=0 PASS
v1   grid=262144 block= 256 avg_ms=  0.2790 effective_read_GB/s=   962.19 gpu_sum=67108864 cpu_sum=67108864 abs_err=0 rel_err=0 PASS
v2   grid=262144 block= 256 avg_ms=  0.2720 effective_read_GB/s=   986.95 gpu_sum=67108864 cpu_sum=67108864 abs_err=0 rel_err=0 PASS
v3   grid=131072 block= 256 avg_ms=  0.1618 effective_read_GB/s=  1659.06 gpu_sum=67108864 cpu_sum=67108864 abs_err=0 rel_err=0 PASS
v6   grid=131072 block= 256 avg_ms=  0.1612 effective_read_GB/s=  1664.75 gpu_sum=67108864 cpu_sum=67108864 abs_err=0 rel_err=0 PASS
v7   grid=   680 block= 256 avg_ms=  0.1618 effective_read_GB/s=  1658.63 gpu_sum=67108864 cpu_sum=67108864 abs_err=0 rel_err=0 PASS
```

这里的 `effective_read_GB/s` 是程序按输入读取字节数估算的有效带宽：

```text
effective_read_GB/s = n * sizeof(float) / avg_kernel_time_seconds / 1e9
```

它不是 Nsight Compute 的硬件计数器结果，只用于快速确认趋势。

## 4. Nsight Compute Memory Throughput 指标

逐版本单独 profile，避免一次运行多个 kernel 时报告混在一起。

推荐指标：

```text
dram__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum
dram__bytes_write.sum
gpu__time_duration.sum
lts__throughput.avg.pct_of_peak_sustained_elapsed
```

单个版本示例：

```bash
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/ncu \
  --kernel-name regex:reduce_v7 \
  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv \
  ./build/ampere/reduce/reduce v7 67108864 1
```

所有版本分别执行：

```bash
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/ncu --kernel-name regex:reduce_v0 --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed --csv ./build/ampere/reduce/reduce v0 67108864 1
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/ncu --kernel-name regex:reduce_v1 --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed --csv ./build/ampere/reduce/reduce v1 67108864 1
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/ncu --kernel-name regex:reduce_v2 --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed --csv ./build/ampere/reduce/reduce v2 67108864 1
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/ncu --kernel-name regex:reduce_v3 --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed --csv ./build/ampere/reduce/reduce v3 67108864 1
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/ncu --kernel-name regex:reduce_v6 --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed --csv ./build/ampere/reduce/reduce v6 67108864 1
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/ncu --kernel-name regex:reduce_v7 --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed --csv ./build/ampere/reduce/reduce v7 67108864 1
```

也可以用 section 生成更完整的 Memory Workload 报告：

```bash
CUDA_VISIBLE_DEVICES=4 /usr/local/cuda/bin/ncu \
  --kernel-name regex:reduce_v7 \
  --section MemoryWorkloadAnalysis \
  ./build/ampere/reduce/reduce v7 67108864 1
```

## 5. 本机当前 Nsight Compute 权限状态

本次在空闲物理 GPU 4 上执行 Nsight Compute 时，程序能启动并运行，但硬件性能计数器被系统权限拒绝：

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

因此当前无法拿到 `dram__throughput...` 等 Nsight Compute 硬件计数器指标。需要管理员开启 NVIDIA performance counter 权限，或者用有权限的账号运行 `ncu`。

开启权限后，重新执行第 4 节的逐版本命令即可得到 Memory Throughput 数据。

## 6. 记录表格模板

开启 Nsight Compute 权限后，建议记录：

```text
version | n | grid | block | avg_ms | effective_read_GB/s | dram_throughput_%peak | dram_read_bytes | dram_write_bytes | lts_throughput_%peak
v0      |   |      |       |        |                     |                       |                 |                  |
v1      |   |      |       |        |                     |                       |                 |                  |
v2      |   |      |       |        |                     |                       |                 |                  |
v3      |   |      |       |        |                     |                       |                 |                  |
v6      |   |      |       |        |                     |                       |                 |                  |
v7      |   |      |       |        |                     |                       |                 |                  |
```

