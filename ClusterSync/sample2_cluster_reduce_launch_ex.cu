#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// 1クラスタあたりの block 数
constexpr int BLOCKS_PER_CLUSTER  = 8;
// 1 block あたりの thread 数
constexpr int THREADS_PER_BLOCK   = 128;
// grid あたりのクラスタ数
constexpr int NUM_CLUSTERS        = 2;

// 合計スレッド数 = NUM_CLUSTERS * BLOCKS_PER_CLUSTER * THREADS_PER_BLOCK

__global__
void cluster_reduce(const float* __restrict__ in,
                    float*       __restrict__ out,
                    int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float v = 0.0f;
    if (gid < n) {
        v = in[gid];           // 中身は全部 1.0 の想定
    }

    // block 内 reduce
    sdata[tid] = v;
    __syncthreads();

    for (int ofs = blockDim.x >> 1; ofs > 0; ofs >>= 1) {
        if (tid < ofs) {
            sdata[tid] += sdata[tid + ofs];
        }
        __syncthreads();
    }

    // cluster 取得
    cg::cluster_group cluster = cg::this_cluster();

    // cluster 全体で共有する変数
    // 実体は各 block にあるが、rank 0 block のものを代表として使う
    __shared__ float cluster_sum;
    if (tid == 0) {
        cluster_sum = 0.0f;
    }
    __syncthreads();

    // 初期化完了をクラスタ全体でそろえる
    cluster.sync();

    // 各 block の部分和 (sdata[0]) を rank 0 block の cluster_sum に加算
    if (tid == 0) {
        float* remote = cluster.map_shared_rank(&cluster_sum, /*rank=*/0);
        atomicAdd(remote, sdata[0]);
    }

    // 全 block が atomicAdd し終わるのを待つ
    cluster.sync();

    // rank 0 block だけが global メモリ out に書き出す
    if (tid == 0 && cluster.block_rank() == 0) {
        atomicAdd(out, cluster_sum);
    }
}

int main()
{
    // 全スレッド数 = 2 * 8 * 128 = 2048
    const int N = NUM_CLUSTERS * BLOCKS_PER_CLUSTER * THREADS_PER_BLOCK;

    float* h_in = new float[N];
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;
    }

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    // 起動設定
    cudaLaunchConfig_t config = {};
    config.gridDim          = dim3(NUM_CLUSTERS * BLOCKS_PER_CLUSTER); // 総 block 数
    config.blockDim         = dim3(THREADS_PER_BLOCK);
    config.dynamicSmemBytes = THREADS_PER_BLOCK * sizeof(float);
    config.stream           = 0;

    // クラスタサイズを属性で指定
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = BLOCKS_PER_CLUSTER; // 1クラスタあたりの block 数
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;

    config.attrs   = attr;
    config.numAttrs = 1;

    // カーネル引数（シグネチャ: (const float*, float*, int)）
    cudaError_t err = cudaSuccess;
    err = cudaLaunchKernelEx(&config, cluster_reduce, d_in, d_out, N);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchKernelEx error: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("GPU sum = %.0f  (expect %d)\n", h_out, N);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;

    return 0;
}
