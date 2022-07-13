import torch
import utils.utils as utils
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MetaBaseline(nn.Module):
    def __init__(self, encoder, method="cos", temp=10.0, temp_learnable=False):
        super().__init__()
        self.encoder = encoder
        self.method = method
        self.momentum = 0.9
        self.center = torch.zeros(1, 1, 1, 768).cuda()
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        """
        x_shot-[B, K, S, Img]
        x_query-[B, K, Q, Img]
        """
        shot_shape = x_shot.shape[:3]
        query_shape = x_query.shape[:3]
        img_shape = x_shot.shape[3:]

        x_shot = x_shot.view(-1, *img_shape)  # [BxKxS, Img]
        x_query = x_query.view(-1, *img_shape)  # [BxKxQ, Img]

        x_tot = self.encoder.forward_features(
            torch.cat([x_shot, x_query], dim=0)
        )  # [BxKx(S+Q), D]

        temp_center = torch.mean(x_tot, dim=0).detach()
        temp_center = repeat(temp_center, "D -> i j k D", i=1, j=1, k=1)
        x_shot, x_query = x_tot[: len(x_shot)], x_tot[-len(x_query) :]

        x_shot = x_shot.view(*shot_shape, -1)  # [B, K, S, D]
        x_query = x_query.view(*query_shape, -1)  # [B, K, Q, D]

        if self.method == "cos":
            prototype = x_shot.mean(dim=-2)  # [B, K, D]
            prototype = F.normalize(prototype, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = "dot"

        elif self.method == "sqr":
            x_shot = F.normalize(x_shot - self.center, dim=-1)
            prototype = x_shot.mean(dim=-2)  # [B, K, D]
            x_query = F.normalize(x_query - self.center, dim=-1)
            self.center = (
                self.momentum * self.center + (1 - self.momentum) * temp_center
            )
            metric = "sqr"

        else:
            assert False

        x_query = rearrange(x_query, "B K Q D -> B (K Q) D")
        logits = utils.compute_logits(x_query, prototype, metric=metric, temp=self.temp)

        return logits
