#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tiny_attn(const float* Q, const float* K, const float* V, const int N, const int d, const int Tc, const int Tr, const int Bc, const int Br, const float scale, float* l, float *m, float* O) {
    extern __shared__ float sram[]; float* Qi = sram, *Kj = &sram[Bc*d], *Vj = &sram[2*Bc*d], *S = &sram[3*Bc*d];
    int tx = threadIdx.x, qkv_idx = (blockIdx.x*gridDim.y + blockIdx.y)*N*d, lm_idx = qkv_idx/d;
    for (int j = 0; j < Tc; j++) {
        for (int x = 0; x < d; x++) Kj[tx*d + x] = K[qkv_idx + j*Bc*d + tx*d + x], Vj[tx*d + x] = V[qkv_idx + j*Bc*d + tx*d + x];
        __syncthreads();
        for (int i = 0; i < Tr; i++) {
            for (int x = 0; x < d; x++) Qi[tx*d + x] = Q[qkv_idx + i*Bc*d + tx*d + x];
            float mi = m[lm_idx + i*Br + tx], li = l[lm_idx + i*Br + tx], row_m = -INFINITY, row_l = 0;
            for (int y = 0; y < Bc; y++) { float sum = 0;
                for (int x = 0; x < d; x++) sum += Qi[tx*d + x]*Kj[y*d + x];
                row_m = max(row_m, S[Bc*tx + y] = sum*scale);
            }
            for (int y = 0; y < Bc; y++) row_l += S[Bc*tx + y] = __expf(S[Bc*tx + y] - row_m);
            float mi_new = max(mi, row_m), li_new = __expf(mi - mi_new)*li + __expf(row_m - mi_new)*row_l;
            for (int x = 0; x < d; x++) { float pv = 0;
                for (int y = 0; y < Bc; y++) pv += S[Bc*tx + y]*Vj[y*d + x];
                O[qkv_idx + i*Bc*d + tx*d + x] = (1/li_new)*(__expf(mi - mi_new)*li*O[qkv_idx + i*Bc*d + tx*d + x] + __expf(row_m - mi_new)*pv);
            }
            m[lm_idx + i*Br + tx] = mi_new, l[lm_idx + i*Br + tx] = li_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B = Q.size(0), nh = Q.size(1), N = Q.size(2), d = Q.size(3), Bc = 32, Br = 32; const int Tc = ceil((float)N / Bc), Tr = ceil((float)N / Br); const float scale = 1.0f / sqrt(d);
    auto O = torch::zeros_like(Q); auto l = torch::zeros({B, nh, N}).to(Q.device()), m = torch::full({B, nh, N}, -INFINITY).to(Q.device());
    const int sram_size = (3 * Bc * d + Bc * Br) * sizeof(float);
    dim3 grid(B, nh), block(Bc);
    tiny_attn<<<grid, block, sram_size>>>(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc, Tr, Bc, Br, scale, l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>());
    return O;
}