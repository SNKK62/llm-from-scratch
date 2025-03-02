import torch
import torch.nn as nn
from .self_attention_v1 import SelfAttention_v1


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your
            [0.55, 0.87, 0.66],  # journey
            [0.57, 0.85, 0.64],  # starts
            [0.22, 0.58, 0.33],  # with
            [0.77, 0.25, 0.10],  # one
            [0.05, 0.80, 0.55],  # step
        ]
    )

    sa_v2 = SelfAttention_v2(d_in=3, d_out=2)
    print(sa_v2(inputs))

    # copy wights of SelfAttention_v2 to SelfAttention_v1 to check the logic between v1 and v2 are same
    sa_v1 = SelfAttention_v1(d_in=3, d_out=2)
    sa_v1.W_query = nn.Parameter(sa_v2.W_query.weight.T)
    sa_v1.W_key = nn.Parameter(sa_v2.W_key.weight.T)
    sa_v1.W_value = nn.Parameter(sa_v2.W_value.weight.T)
    print(sa_v1(inputs))
