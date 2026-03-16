#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// 如果每个block有多个warp，这会导致全局内存高重复访问，可以通过smem进行优化
// Block内所有thread协同搬数到smem，然后由warp 加载fragment

// Tensor Core 对 FP16 的标准形状要求通常是 16x16x16
// 假设一个 Block 处理64*64的区域，由4*4个 Warp 组成（每个 Warp 处理16*16）
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define S_M 64
#define S_N 64
#define S_K 64

#define M_GLOBAL 1024
#define N_GLOBAL 512
#define K_GLOBAL 1024

// 使用 managed 内存简化数据传输
__managed__ half a[M_GLOBAL * N_GLOBAL];
__managed__ half b[N_GLOBAL * K_GLOBAL];
__managed__ float c_gpu[M_GLOBAL * K_GLOBAL];
__managed__ float c_cpu[M_GLOBAL * K_GLOBAL];

__global__ void gpu_matmul_fp16(const half *a, const half *b, float *c, int m, int n, int k)
{
    __shared__ half sub_a[S_M][S_N];
    __shared__ half sub_b[S_N][S_K];

    // 计算当前 Warp 在 Block 内的位置
    int warpId = threadIdx.x / 32;
    int warpM = (warpId / 4) * WMMA_M; // Block 内纵向偏移
    int warpK = (warpId % 4) * WMMA_K; // Block 内横向偏移

    // 计算当前 Block 在全局矩阵中的起始位置
    int blockM = blockIdx.y * S_M;
    int blockK = blockIdx.x * S_K;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // 沿内积维度 N 循环
    for (int i = 0; i < n; i += S_N) {
        // --- 1. 协同搬运数据到 Shared Memory ---
        // 使用整个 Block 的线程(512个)分摊搬运任务
        for (int t = threadIdx.x; t < S_M * S_N; t += blockDim.x) {
            int r = t / S_N;
            int c_idx = t % S_N;
            sub_a[r][c_idx] = a[(blockM + r) * n + (i + c_idx)];
        }
        for (int t = threadIdx.x; t < S_N * S_K; t += blockDim.x) {
            int r = t / S_K;
            int c_idx = t % S_K;
            sub_b[r][c_idx] = b[(i + r) * k + (blockK + c_idx)];
        }

        __syncthreads(); // 确保搬运完成

        // --- 2. 使用 Tensor Core 进行计算 ---
        // 在当前的 S_N 范围内，Warp 需要进一步循环读取 Smem
        for (int j = 0; j < S_N; j += WMMA_N) {
            wmma::load_matrix_sync(a_frag, (half*)&sub_a[warpM][j], S_N);
            wmma::load_matrix_sync(b_frag, (half*)&sub_b[j][warpK], S_K);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads(); // 确保计算完成，准备加载下一块 Smem
    }

    // --- 3. 写回结果 ---
    int g_m = blockM + warpM;
    int g_k = blockK + warpK;
    if (g_m < m && g_k < k) {
        wmma::store_matrix_sync(c + g_m * k + g_k, acc_frag, k, wmma::mem_row_major);
    }
}

void cpu_matmul(const half *a, const half *b, float *c, int m, int n, int k)
{
    for(int y = 0; y < m; y++)
    {
        for(int x = 0; x < k; x++)
        {
            float tmp = 0.0f;
            for(int step = 0; step < n; step++)
            {
                // 将 half 转为 float 计算
                tmp += __half2float(a[y * n + step]) * __half2float(b[step * k + x]);
            }
            c[y * k + x] = tmp;
        }
    }
}

int main()
{
    // 初始化数据
    for(int i = 0; i < M_GLOBAL * N_GLOBAL; i++)
    {
        a[i] = __float2half((float)(rand() % 10) / 10.0f);
    }

    for(int i = 0; i < N_GLOBAL * K_GLOBAL; i++)
    {
        b[i] = __float2half((float)(rand() % 10) / 10.0f);
    }
        
    // 每个 block 配置 4 * 4 个 warp (512 threads)
    dim3 dimBlock(512, 1);
    // 每个 block 处理 4 个 16x16 tile (纵向排列)
    dim3 dimGrid((K_GLOBAL / WMMA_K) / 4, (M_GLOBAL / WMMA_M) / 4);

    gpu_matmul_fp16<<<dimGrid, dimBlock>>>(a, b, c_gpu, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    cudaDeviceSynchronize();

    // CPU 验证
    cpu_matmul(a, b, c_cpu, M_GLOBAL, N_GLOBAL, K_GLOBAL);

    // 浮点数验证（使用较小的阈值，因为 half 存在精度损失）
    bool errors = false;
    for(int i = 0; i < M_GLOBAL * K_GLOBAL; i++)
    {
        if(abs(c_cpu[i] - c_gpu[i]) > 0.1f) 
        {
            errors = true;
            break;
        }
    }

    printf("Result: %s\n", errors ? "Failed" : "Passed");

    return 0;
}