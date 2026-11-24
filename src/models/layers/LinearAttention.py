import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    def __init__(
        self, d_q, d_kv, d_att, mem_size=16, eps=1e-6, mem_mode="append", mem_gain=1.0
    ):
        super().__init__()
        assert mem_mode in ("append", "slots")
        self.mem_mode = mem_mode
        self.proj_Q = nn.Linear(d_q, d_att)
        self.proj_K = nn.Linear(d_kv, d_att)
        self.proj_V = nn.Linear(d_kv, d_att)
        self.eps = eps

        self.k_mem = nn.Parameter(torch.empty(mem_size, d_att))
        self.v_mem = nn.Parameter(torch.empty(mem_size, d_att))

        nn.init.orthogonal_(self.k_mem, gain=mem_gain)
        nn.init.orthogonal_(self.v_mem)

        self._collect = None  # dict -> forward stores Q, w and the fraction of memory mass (for logging)

    def elu_feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, q, k, v):
        Q = self.elu_feature_map(self.proj_Q(q))  # (n, l, d)
        k = self.elu_feature_map(self.proj_K(k))  # (n, s, d)
        v = self.proj_V(v)

        if self.mem_mode == "slots":
            KV = torch.einsum("nsd,nse->nde", k, v)
            K_sum = k.sum(dim=1)
            w = F.softplus(Q @ self.k_mem.t())  # (n, l, M), positive
            num = torch.einsum("nld,nde->nle", Q, KV) + w @ self.v_mem
            den = torch.einsum("nld,nd->nl", Q, K_sum) + w.sum(-1) + self.eps
            if self._collect is not None:
                self._collect["q"].append(Q.detach().reshape(-1, Q.size(-1)))
                self._collect["w"].append(w.detach().reshape(-1, w.size(-1)))
                self._collect["mass"].append(
                    (w / den.unsqueeze(-1)).detach().reshape(-1, w.size(-1))
                )
            return (num / den.unsqueeze(-1)).contiguous()

        # Append memory vectors to K and V
        K = torch.cat([k, self.k_mem.expand(Q.size(0), -1, -1)], dim=1)
        V = torch.cat([v, self.v_mem.expand(Q.size(0), -1, -1)], dim=1)

        # (n, d, d)
        KV = torch.einsum("nsd,nse->nde", K, V)

        # (n, d)
        K_sum = K.sum(dim=1)

        # (n, l)
        Z = 1.0 / (torch.einsum("nld,nd->nl", Q, K_sum) + self.eps)

        # (n, l, d)
        out = torch.einsum("nld,nde,nl->nle", Q, KV, Z)

        return out.contiguous()
