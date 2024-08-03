import torch

print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

q = torch.randn(2, 3, 8, device='cuda')
k = torch.randn(2, 3, 8, device='cuda')
v = torch.randn(2, 3, 8, device='cuda')
mask = torch.ones(2, 1, 3, device='cuda', dtype=torch.bool)

try:
    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    print("Flash Attention is working.")
except RuntimeError as e:
    print(f"Flash Attention test failed: {e}")