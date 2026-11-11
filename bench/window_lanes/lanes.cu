// Microbenchmark of the searchChi2 inner loop with G lanes per window (G = 1 is the current
// kernel's scheme, G = 32 one warp per window). One duration per launch, so the per-window
// cost depends only on (L, stride, G) and on the device.
__device__ __forceinline__ unsigned int orderedKey(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <int G, bool FP64DIV>
__device__ void searchG(unsigned long long* best, const double* cumResidual, const float2* weighted,
                        double base, const float* s, float overshoot, int L, int stride, int nWin,
                        int patchedSize, float transitDepthMin) {
    int z = blockIdx.y;
    int sub = threadIdx.x % G;
    int win = (blockIdx.x * blockDim.x + threadIdx.x) / G;
    unsigned long long key = 0xffffffffffffffffull;
    float ar = 0.f, a2 = 0.f, meanDepth = 0.f;
    bool active = false;
    int start = 0;
    if (win < nWin) {
        start = win * stride;
        const double* C = cumResidual + (long long)z * patchedSize;
        double windowSum = C[start + L - 1] - (start > 0 ? C[start - 1] : 0.0);
        meanDepth = FP64DIV ? (float)(windowSum / L) : (float)windowSum * __frcp_rn((float)L);
        active = meanDepth > transitDepthMin;
        if (active) {
            const float2* x = weighted + (long long)z * patchedSize + start;
            for (int i = sub; i < L; i += G) {
                float si = s[i];
                float2 v = x[i];
                ar = fmaf(si, v.x, ar);
                a2 = fmaf(si * si, v.y, a2);
            }
        }
    }
    #pragma unroll
    for (int o = G / 2; o > 0; o >>= 1) {
        ar += __shfl_xor_sync(0xffffffff, ar, o);
        a2 += __shfl_xor_sync(0xffffffff, a2, o);
    }
    if (active && sub == 0) {
        float k = meanDepth * overshoot * 2.0f;
        float chi2 = (float)(base + (double)(k * (k * a2 - 2.0f * ar)));
        key = ((unsigned long long)orderedKey(chi2) << 32) | (unsigned int)start;
    }
    for (int o = 16; o > 0; o >>= 1) {
        unsigned long long other = __shfl_down_sync(0xffffffff, key, o);
        key = other < key ? other : key;
    }
    __shared__ unsigned long long warpMin[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warpMin[warp] = key;
    __syncthreads();
    if (warp == 0) {
        key = lane < (int)(blockDim.x >> 5) ? warpMin[lane] : 0xffffffffffffffffull;
        for (int o = 16; o > 0; o >>= 1) {
            unsigned long long other = __shfl_down_sync(0xffffffff, key, o);
            key = other < key ? other : key;
        }
        if (lane == 0 && key != 0xffffffffffffffffull) atomicMin(best + z, key);
    }
}

#define DEF(G) \
extern "C" __global__ void search_G##G(unsigned long long* b, const double* c, const float2* w, double base, \
    const float* s, float ov, int L, int st, int n, int p, float dmin) { searchG<G, true>(b, c, w, base, s, ov, L, st, n, p, dmin); } \
extern "C" __global__ void search_G##G##_f32(unsigned long long* b, const double* c, const float2* w, double base, \
    const float* s, float ov, int L, int st, int n, int p, float dmin) { searchG<G, false>(b, c, w, base, s, ov, L, st, n, p, dmin); }
DEF(1) DEF(2) DEF(4) DEF(8) DEF(16) DEF(32)
