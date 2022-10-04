from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from src.model.news_encoder import NewsEncoder
from src.model.user_encoder import UserEncoder


class NRMS(nn.Module):
    def __init__(self, user_encoder: UserEncoder, news_encoder: NewsEncoder):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder

    def forward(self, title: Tensor, title_mask: Tensor, his_title: Tensor, his_title_mask: Tensor,
                his_mask: Tensor, sapo: Union[Tensor, None] = None, sapo_mask: Union[Tensor, None] = None,
                his_sapo: Union[Tensor, None] = None, his_sapo_mask: Union[Tensor, None] = None,
                category: Union[Tensor, None] = None, his_category: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            title: tensor of shape ``(batch_size, num_candidates, title_length)``.
            title_mask: tensor of shape ``(batch_size, num_candidates, title_length)``.
            his_title: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_title_mask: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_mask: tensor of shape ``(batch_size, num_clicked_news)``.
            sapo: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            sapo_mask: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            his_sapo: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            his_sapo_mask: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            category: tensor of shape ``(batch_size, num_candidates)``.
            his_category: tensor of shape ``(batch_size, num_clicked_news)``.
        Returns:
            Tensor of shape ``(batch_size, num_candidates)``
        """
        # Representation of the candidate news
        batch_size = title.shape[0]
        num_candidates = title.shape[1]
        title = title.view(batch_size * num_candidates, -1)
        title_mask = title_mask.view(batch_size * num_candidates, -1)
        sapo = sapo.view(batch_size * num_candidates, -1)
        sapo_mask = sapo_mask.view(batch_size * num_candidates, -1)
        category = category.view(batch_size * num_candidates)
        candidate_news_repr = self.news_encoder(title_encoding=title, title_attn_mask=title_mask, sapo_encoding=sapo,
                                                sapo_attn_mask=sapo_mask, category_encoding=category)
        candidate_news_repr = candidate_news_repr.view(batch_size, num_candidates, -1)

        # Representation of the users
        user_repr = self.user_encoder(title_encoding=his_title, title_attn_mask=his_title_mask, history_mask=his_mask,
                                      sapo_encoding=his_sapo, sapo_attn_mask=his_sapo_mask,
                                      category_encoding=his_category)

        # Click predictor
        logits = torch.bmm(candidate_news_repr, user_repr.unsqueeze(dim=2)).squeeze(dim=2)

        return logits
