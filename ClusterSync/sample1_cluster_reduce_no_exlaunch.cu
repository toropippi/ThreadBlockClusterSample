#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
constexpr int BLOCKS_PER_CLUSTER  = 8;// cluster あたりの block 数
constexpr int THREADS_PER_BLOCK   = 128;// block あたりの thread 数
constexpr int NUM_CLUSTERS        = 2;// grid あたりの cluster 数
// __cluster_dims__ には「cluster あたりの block 数」を指定
__global__ __cluster_dims__(BLOCKS_PER_CLUSTER, 1, 1)
void cluster_reduce(const float* __restrict__ in, float* __restrict__ out)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    // 範囲外アクセスチェックはしてないことに注意
    float v = in[gid];
    // block 内での reduce
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
    // 実体は各 block にあるが、rank 0 のものを代表として使う
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
    const int N = NUM_CLUSTERS * BLOCKS_PER_CLUSTER * THREADS_PER_BLOCK;
    float* h_in = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));
    // grid.x = クラスタ数 * 1クラスタあたりの block 数であることに注意
    dim3 grid(NUM_CLUSTERS * BLOCKS_PER_CLUSTER);
    dim3 block(THREADS_PER_BLOCK);
    size_t shmem = THREADS_PER_BLOCK * sizeof(float); // sdata 用
    // カーネル実行
    cluster_reduce<<<grid, block, shmem>>>(d_in, d_out);
    
    cudaDeviceSynchronize();
    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU sum = %.0f  (expect %d)\n", h_out, N);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    return 0;
}