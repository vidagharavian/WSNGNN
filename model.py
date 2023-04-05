import torch



class DIMPA(torch.nn.Module):
    r"""The directed mixed-path aggregation model.

    Args:
        hop (int): Number of hops to consider.
    """

    def __init__(self, hop: int):
        super(DIMPA, self).__init__()
        self._hop = hop
        self._w_s = Parameter(torch.FloatTensor(hop + 1, 1))
        self._w_t = Parameter(torch.FloatTensor(hop + 1, 1))

        self._reset_parameters()

    def _reset_parameters(self):
        self._w_s.data.fill_(1.0)
        self._w_t.data.fill_(1.0)

    def forward(self, x_s: torch.FloatTensor, x_t: torch.FloatTensor,
                A: Union[torch.FloatTensor, torch.sparse_coo_tensor],
                At: Union[torch.FloatTensor, torch.sparse_coo_tensor]) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Making a forward pass of DIMPA.

        Arg types:
            * **x_s** (PyTorch FloatTensor) - Source hidden representations.
            * **x_t** (PyTorch FloatTensor) - Target hidden representations.
            * **A** (PyTorch FloatTensor or PyTorch sparse_coo_tensor) - Row-normalized adjacency matrix.
            * **At** (PyTorch FloatTensor or PyTorch sparse_coo_tensor) - Tranpose of column-normalized adjacency matrix.

        Return types:
            * **feat** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*input_dim).
        """
        feat_s = self._w_s[0]*x_s
        feat_t = self._w_t[0]*x_t
        curr_s = x_s.clone()
        curr_t = x_t.clone()
        for h in range(1, 1+self._hop):
            curr_s = torch.matmul(A, curr_s)
            curr_t = torch.matmul(At, curr_t)
            feat_s += self._w_s[h]*curr_s
            feat_t += self._w_t[h]*curr_t

        feat = torch.cat([feat_s, feat_t], dim=1)  # concatenate results

        return feat
