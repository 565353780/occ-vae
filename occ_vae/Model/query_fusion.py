import torch
import torch.nn as nn


class QueryFusion(nn.Module):
    """
    Fuse variable-length token features per batch into fixed number of M output tokens per batch.

    Input:
        feat_all: Tensor of shape (T, C)     -- 所有 batch 拼接后的 token 特征
        batch_idx: Tensor of shape (T,)      -- 每个 token 属于哪个 batch (0..B-1)
    Output:
        fused: Tensor of shape (B, M, K)
    """

    def __init__(
        self, input_dim: int, M: int, K: int, num_heads: int = 4, dropout: float = 0.0
    ):
        """
        Args:
            input_dim: C, 输入 token 的特征维度。
            M: 每个 batch 想要输出多少个 token/features。
            K: 输出 token 的特征维度。
            num_heads: 注意力头数。
            dropout: 注意力中的 dropout。
        """
        super(QueryFusion, self).__init__()
        self.input_dim = input_dim
        self.M = M
        self.K = K

        # learnable query tokens (共有 M 个)，初始维度为 input_dim，以便做 attention query
        self.queries = nn.Parameter(
            torch.randn(1, M, input_dim)
        )  # 将来 expand 到 batch 维度

        # MultiheadAttention 要求某种 (seq, batch, dim) 或 (batch, seq, dim) 格式
        # 我们这里使用 batch_first=True 模式
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # 最终将 attention 的输出 proj 到 K 维
        self.out_proj = nn.Linear(input_dim, K)

    def forward(
        self,
        feat_all: torch.Tensor,
        batch_idx: torch.Tensor,
    ):
        """
        Args:
            feat_all: (T, C)
            batch_idx: (T,), long, in [0, B-1]
        Returns:
            fused: Tensor of shape (B, M, K)
        """
        device = feat_all.device
        dtype = feat_all.dtype
        B = int(batch_idx.max().item() + 1)

        outputs = []
        for b in range(B):
            # 获取 batch b 的所有 token 特征
            mask_b = batch_idx == b
            feats_b = feat_all[mask_b]  # (N_b, C)

            N_b = feats_b.shape[0]
            if N_b == 0:
                queries = self.queries.expand(1, self.M, self.input_dim)  # (1, M, C)
                dummy_key = torch.zeros(
                    (1, 1, self.input_dim), device=device, dtype=dtype
                )
                dummy_value = torch.zeros_like(dummy_key)
                attn_out, _ = self.attn(queries, dummy_key, dummy_value)  # (1, M, C)
                attn_out = attn_out.squeeze(0)
                out_b = self.out_proj(attn_out)

                # 如果这个 batch 没有 token（极端情况），填零
                # out_b = torch.zeros((self.M, self.K), device=device, dtype=dtype)
            else:
                # 扩展 queries 到 batch size 1，为这个 batch 用
                # queries: (1, M, C)
                queries = self.queries.expand(1, self.M, self.input_dim)  # batch size 1

                # key 和 value 都是 feats_b
                # 但是 MultiheadAttention 的输入格式是 (batch, seq_len, dim)，这里 batch=1，seq_len=N_b
                keys = feats_b.unsqueeze(0)  # (1, N_b, C)
                values = feats_b.unsqueeze(0)  # (1, N_b, C)

                # 注意：因为我们希望 query attend 到全部 feats_b，所以 query shape (1, M, C)
                # 如果 N_b < some threshold，不用截断
                attn_out, _ = self.attn(queries, keys, values)  # attn_out: (1, M, C)

                attn_out = attn_out.squeeze(0)  # (M, C)
                out_b = self.out_proj(attn_out)  # (M, K)

            outputs.append(out_b)

        fused = torch.stack(outputs, dim=0)  # (B, M, K)
        return fused
