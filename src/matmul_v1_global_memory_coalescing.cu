#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// 全局内存合并
// 理解warp在其中的作用:naive方法的warp访问的全局内存不连续
// 属于同一warp的线程的顺序内存访问可以合并并作为一个整体执行。这被称为全局内存合并（GMEM）
// 线程id计算方法：threadId = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z)
// 注意这里不再区分线程在块内的维度，而是计算一个id，这个id用于做warp的划分
// threadIdx.x 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
// threadIdx.y 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
// threadIdx.z 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
// threadIdx   0 1 2 3 4 5 6 7 8 9 ...

// 本方法和v0没有区别，原因是：
// 对于A阵，连续的thread访问同一内存；
// 对于B矩阵连续的thread之间访问连续的首地址，thread内带stride，这样的访存请求已是合并的

#define M 1000
#define N 500
#define K 1000
#define BLOCK_SIZE 16

__managed__ int a[M * N]; // 统一内存
__managed__ int b[N * K];
__managed__ int c_gpu[M * K];
__managed__ int c_cpu[M * K];

__global__ void gpu_matmul1(int *a, int *b, int *c, int m, int n, int k)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    // blockDim：Block 的维度（每个维度上的线程数)
    // blockIdx：当前 Block 在 Grid 中的索引（从 0 开始）
    // threadIdx：当前 Thread 在 Block 中的索引（从 0 开始）
    int tmp = 0;
    if(x < k && y < m) //否则出越界报错
    {
        for(int step=0; step<n; ++step) // 该方法存在重复访存的问题
        {
            tmp += a[y*n + step] * b[step * k + x];
        }
        c[y * k + x] = tmp;
    }
}

void cpu_matmul(int *a, int *b, int *c, int m, int n, int k)
{
    for(int y=0; y<m; ++y)
    {
        for(int x=0; x<k; ++x)
        {
            int tmp = 0;
            for(int step=0; step<n; step++)
            {
                tmp += a[y * n + step] * b[step * k + x];
            }
            c[y * k + x] = tmp;
        }
    }
}

int main()
{
    // y表示行，x表示列
    // 数据初始化
    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            a[y * N + x] = rand() % 1024;
        }
    }

    for(int y=0; y<N; ++y)
    {
        for(int x=0; x<K; ++x)
        {
            b[y * K + x] = rand() % 1024;
        }
    }

    unsigned int grid_x = (K + BLOCK_SIZE - 1) / BLOCK_SIZE; // 列数
    unsigned int grid_y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE; // 行数

    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matmul1<<<dimGrid, dimBlock>>>(a, b, c_gpu, M, N, K);
    cpu_matmul(a, b, c_cpu, M, N, K);

    bool errors = false;
    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<K; ++x)
        {
            if(fabs(c_cpu[y * K + x] - c_gpu[y * K + x]) > (1.0e-10))
            {
                errors = true;
            }
        }
    }

    printf("Result: %s\n", errors?"Failed":"Passed");

    return 0;
}