// cluster_wang_hash_histogram.cu
//
// RTX 5090 (Blackwell, sm_120) 向け Thread Block Cluster / DSMEM + Wang hash ヒストグラムベンチ。
// - Baseline: Wang hash → global histogram へ直接 atomicAdd
// - TBC版: Wang hash → DSMEM(分散共有メモリ) 上の histogram に atomicAdd → 最後だけ global へ flush
//
// コンパイル例:
//   nvcc -arch=sm_120 -std=c++17 -rdc=true cluster_wang_hash_histogram.cu -o cluster_wang_hash_histogram
//
// 実行例:
//   ./cluster_wang_hash_histogram

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

//--------------------------------------
// パラメータ
//--------------------------------------

// Wang hash の出力を 0〜262143 に縮約 → 2^18 bin
constexpr int BINS_PER_BLOCK      = 16384;           // 1 block が担当する bin 数 (64KB / 4B)
constexpr int BLOCKS_PER_CLUSTER  = 16;              // 1 クラスタあたり block 数
constexpr int BINS_TOTAL          = BINS_PER_BLOCK * BLOCKS_PER_CLUSTER; // 262144
static_assert(BINS_TOTAL == 262144, "BINS_TOTAL must be 262144");

// グリッド構成
constexpr int NUM_CLUSTERS        = 10;   // クラスタ数（環境に応じて調整可）
constexpr int THREADS_PER_BLOCK   = 128;  // 1 block あたりスレッド数

// 1 thread が生成する hash 値の個数
constexpr int SAMPLES_PER_THREAD  = 4096*24;

// 計測ループ回数（ウォームアップ 1 回 + 本番 NUM_LOOPS 回）
constexpr int NUM_LOOPS           = 3;

//--------------------------------------
// エラーチェックマクロ
//--------------------------------------

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::fprintf(stderr,                                           \
                         "CUDA error at %s:%d: %s (code=%d)\n",            \
                         __FILE__, __LINE__, cudaGetErrorString(err__),    \
                         static_cast<int>(err__));                         \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

//--------------------------------------
// Wang hash 実装 (32bit)
//--------------------------------------

__device__ __forceinline__
unsigned int wang_hash(unsigned int x)
{
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2d;
    x = x ^ (x >> 15);
    return x;
}

//--------------------------------------
// Baseline: Wang hash → global histogram (global atomic)
//--------------------------------------

__global__ void histogram_global_hash_kernel(unsigned int* __restrict__ global_hist,
                                             int samples_per_thread)
{
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // スレッドごとに異なるシードを用意（簡易）
    unsigned int seed = static_cast<unsigned int>(global_tid * 747796405u + 2891336453u);

    // 全体で同じ総サンプル数になるようにするなら、tid/strideを使う設計もあるが、
    // ここでは単純に「各 thread が samples_per_thread 回 hash を生成」
    for (int i = 0; i < samples_per_thread; ++i) {
        seed = wang_hash(seed);
        unsigned int bin = seed & (BINS_TOTAL - 1); // 0〜(BINS_TOTAL-1)

        // グローバルヒストグラムへ atomicAdd
        atomicAdd(&global_hist[bin], 1u);
    }

    // total_threads などは未使用だが、将来拡張用に残しておいてもよい
    (void) total_threads;
}

//--------------------------------------
// TBC + DSMEM 版: Wang hash → 分散共有メモリ histogram → global flush
//--------------------------------------

__global__ void histogram_cluster_hash_kernel(unsigned int* __restrict__ global_hist,
                                              int samples_per_thread)
{
    extern __shared__ unsigned int s_hist[]; // 各 block に 64KB (16384 bin) 割り当て

    cg::cluster_group cluster = cg::this_cluster();
    const int rank = cluster.block_rank(); // 0 .. BLOCKS_PER_CLUSTER-1

    // s_hist のゼロクリア
    for (int i = threadIdx.x; i < BINS_PER_BLOCK; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // cluster 内でゼロクリア完了を同期
    cluster.sync();

    const int global_tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads  = gridDim.x * blockDim.x;

    // シード
    unsigned int seed = static_cast<unsigned int>(global_tid * 747796405u + 2891336453u);

    // Wang hash を samples_per_thread 回生成
    for (int i = 0; i < samples_per_thread; ++i) {
        seed = wang_hash(seed);
        unsigned int bin = seed & (BINS_TOTAL - 1); // 0〜(BINS_TOTAL-1)

        // bin を担当する block と offset を決定
        const int owner_rank   = static_cast<int>(bin / BINS_PER_BLOCK);
        const int owner_offset = static_cast<int>(bin % BINS_PER_BLOCK);

        // owner_rank 番目の block の shared histogram を DSMEM 経由で取得
        unsigned int* remote_hist = cluster.map_shared_rank(s_hist, owner_rank);

        // DSMEM 上で atomicAdd（グローバルではない）
        atomicAdd(&remote_hist[owner_offset], 1u);
    }

    // すべての DSMEM 更新が完了するまで同期
    cluster.sync();

    // 各 block は自分の担当区間 [rank * BINS_PER_BLOCK, (rank+1)*BINS_PER_BLOCK) を global に flush
    const int bin_base = rank * BINS_PER_BLOCK;
    for (int i = threadIdx.x; i < BINS_PER_BLOCK; i += blockDim.x) {
        unsigned int c = s_hist[i];
        if (c != 0u) {
            atomicAdd(&global_hist[bin_base + i], c);
        }
    }

    (void) total_threads;
}

//--------------------------------------
// ベンチマーク用ヘルパ
//--------------------------------------

float benchmark_global(unsigned int* d_hist,
                       dim3 grid,
                       dim3 block,
                       int samples_per_thread)
{
    // ウォームアップ 1 回
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS_TOTAL * sizeof(unsigned int)));
    histogram_global_hash_kernel<<<grid, block>>>(d_hist, samples_per_thread);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;

    for (int iter = 0; iter < NUM_LOOPS; ++iter) {
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS_TOTAL * sizeof(unsigned int)));

        CUDA_CHECK(cudaEventRecord(start));
        histogram_global_hash_kernel<<<grid, block>>>(d_hist, samples_per_thread);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / NUM_LOOPS;
}

float benchmark_cluster(unsigned int* d_hist,
                        dim3 grid,
                        dim3 block,
                        int samples_per_thread,
                        std::size_t shared_bytes)
{
    // cudaLaunchKernelEx 用設定
    cudaLaunchConfig_t config{};
    config.gridDim          = grid;
    config.blockDim         = block;
    config.dynamicSmemBytes = static_cast<unsigned int>(shared_bytes);
    config.stream           = 0;

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = BLOCKS_PER_CLUSTER;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;

    config.attrs    = attr;
    config.numAttrs = 1;

    // dynamic shared memory / carveout / 非ポータブルクラスタサイズ属性を設定
    CUDA_CHECK(cudaFuncSetAttribute(
        histogram_cluster_hash_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));

    CUDA_CHECK(cudaFuncSetAttribute(
        histogram_cluster_hash_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100)); // shared 優先

    CUDA_CHECK(cudaFuncSetAttribute(
        histogram_cluster_hash_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1));   // clusterDim.x=16 を許可

    // ウォームアップ 1 回
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS_TOTAL * sizeof(unsigned int)));
    CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        histogram_cluster_hash_kernel,
        d_hist,
        samples_per_thread
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 計測ループ
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;

    for (int iter = 0; iter < NUM_LOOPS; ++iter) {
        CUDA_CHECK(cudaMemset(d_hist, 0, BINS_TOTAL * sizeof(unsigned int)));

        CUDA_CHECK(cudaEventRecord(start));

        CUDA_CHECK(cudaLaunchKernelEx(
            &config,
            histogram_cluster_hash_kernel,
            d_hist,
            samples_per_thread
        ));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / NUM_LOOPS;
}

//--------------------------------------
// メイン
//--------------------------------------

int main()
{
    std::printf("Wang-hash histogram benchmark with Thread Block Cluster (RTX 5090)\n");
    std::printf("  BINS_PER_BLOCK      = %d\n", BINS_PER_BLOCK);
    std::printf("  BLOCKS_PER_CLUSTER  = %d\n", BLOCKS_PER_CLUSTER);
    std::printf("  BINS_TOTAL          = %d\n", BINS_TOTAL);
    std::printf("  NUM_CLUSTERS        = %d\n", NUM_CLUSTERS);
    std::printf("  THREADS_PER_BLOCK   = %d\n", THREADS_PER_BLOCK);
    std::printf("  SAMPLES_PER_THREAD  = %d\n", SAMPLES_PER_THREAD);
    std::printf("  NUM_LOOPS           = %d (first launch is warm-up)\n", NUM_LOOPS);

    // デバイス情報
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("\nDevice: %s\n", prop.name);
    std::printf("  Compute Capability        : %d.%d\n", prop.major, prop.minor);
    std::printf("  Shared mem per SM (bytes) : %zu\n", prop.sharedMemPerMultiprocessor);
    std::printf("  Shared mem per block (max, opt-in bytes): %zu\n",
                prop.sharedMemPerBlockOptin);

    // グリッド構成
    const int blocks_per_grid   = NUM_CLUSTERS * BLOCKS_PER_CLUSTER;
    const int threads_per_block = THREADS_PER_BLOCK;
    const int total_threads     = blocks_per_grid * threads_per_block;
    const unsigned long long total_samples =
        static_cast<unsigned long long>(total_threads) *
        static_cast<unsigned long long>(SAMPLES_PER_THREAD);

    std::printf("\nLaunch configuration:\n");
    std::printf("  blocks_per_grid   = %d\n", blocks_per_grid);
    std::printf("  threads_per_block = %d\n", threads_per_block);
    std::printf("  total_threads     = %d\n", total_threads);
    std::printf("  total_samples     = %llu\n", total_samples);

    // デバイスメモリ（ヒストグラム）確保
    unsigned int* d_hist_global  = nullptr;
    unsigned int* d_hist_cluster = nullptr;

    CUDA_CHECK(cudaMalloc(&d_hist_global,  BINS_TOTAL * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_hist_cluster, BINS_TOTAL * sizeof(unsigned int)));

    dim3 grid(blocks_per_grid);
    dim3 block(threads_per_block);

    // 共有メモリ (per block) = 64KB
    const std::size_t shared_bytes = static_cast<std::size_t>(BINS_PER_BLOCK) *
                                     sizeof(unsigned int); // 16384 * 4 = 65536

    //----------------------------------
    // ベンチマーク
    //----------------------------------

    std::printf("\nRunning benchmarks...\n");

    float avg_ms_global = benchmark_global(
        d_hist_global,
        grid,
        block,
        SAMPLES_PER_THREAD);

    float avg_ms_cluster = benchmark_cluster(
        d_hist_cluster,
        grid,
        block,
        SAMPLES_PER_THREAD,
        shared_bytes);

    std::printf("\nResults (average over %d runs, first run is warm-up):\n",
                NUM_LOOPS);
    std::printf("  Global-only (one atomic per hash) : %.3f ms\n", avg_ms_global);
    std::printf("  TBC+DSMEM (local DSMEM hist)      : %.3f ms\n", avg_ms_cluster);

    //----------------------------------
    // 結果の一致確認と総和チェック
    //----------------------------------

    std::vector<unsigned int> h_hist_global(BINS_TOTAL);
    std::vector<unsigned int> h_hist_cluster(BINS_TOTAL);

    CUDA_CHECK(cudaMemcpy(h_hist_global.data(), d_hist_global,
                          BINS_TOTAL * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hist_cluster.data(), d_hist_cluster,
                          BINS_TOTAL * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    unsigned long long sum_global  = 0;
    unsigned long long sum_cluster = 0;
    int mismatch = 0;

    for (int i = 0; i < BINS_TOTAL; ++i) {
        sum_global  += h_hist_global[i];
        sum_cluster += h_hist_cluster[i];
        if (h_hist_global[i] != h_hist_cluster[i]) {
            ++mismatch;
            if (mismatch <= 10) {
                std::printf("  MISMATCH at bin %d: global=%u, cluster=%u\n",
                            i, h_hist_global[i], h_hist_cluster[i]);
            }
        }
    }

    std::printf("\nHistogram total counts:\n");
    std::printf("  sum(global)  = %llu\n", sum_global);
    std::printf("  sum(cluster) = %llu\n", sum_cluster);
    std::printf("  expected     = %llu\n", total_samples);

    if (mismatch == 0 && sum_global == total_samples && sum_cluster == total_samples) {
        std::printf("  Output check: OK (no mismatch, totals match)\n");
    } else {
        std::printf("  Output check: %d mismatches\n", mismatch);
    }

    //----------------------------------
    // 後始末
    //----------------------------------

    CUDA_CHECK(cudaFree(d_hist_global));
    CUDA_CHECK(cudaFree(d_hist_cluster));
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
