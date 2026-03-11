#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// 全局内存合并
// 理解warp在其中的作用：
// 属于同一warp的线程的顺序内存访问可以合并并作为一个整体执行。这被称为全局内存合并（GMEM）
// 线程id计算方法：threadId = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z)
// 注意这里不再区分线程在块内的维度，而是计算一个id，这个id用于做warp的划分

#define M 1000
#define N 500
#define K 1000
#define BLOCK_SIZE 16

__managed__ int a[M * N]; // 统一内存
__managed__ int b[N * K];
__managed__ int c_gpu[M * K];
__managed__ int c_cpu[M * K];

void gpu_matmul1(int *a, int *b, int *c, int m, int n, int k)
{

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