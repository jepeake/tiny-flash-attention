import math, torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'], extra_cuda_cflags=['-O2'])
batch_size, n_head, seq_len, head_embd = 16, 12, 64, 64
q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('--- Profiling Manual Flash Attention ---')
def manual_attn(q, k, v): return (F.softmax((q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))), dim=-1) @ v)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('--- Profiling Tiny Flash Attention ---')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))