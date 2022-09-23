from typing import Union

from torch import Tensor
import torch.nn as nn

from src.model.additive_attention import AdditiveAttention
from src.model.news_encoder import NewsEncoder


class UserEncoder(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, num_heads: int, attn_dropout: int, query_dim: int):
        super().__init__()
        self.news_encoder = news_encoder
        self.multi_head_self_attn = nn.MultiheadAttention(embed_dim=news_encoder.embed_dim, num_heads=num_heads,
                                                          dropout=attn_dropout, batch_first=True)
        self.additive_attn = AdditiveAttention(query_dim=query_dim, embed_dim=news_encoder.embed_dim)

    def forward(self, title_encoding: Tensor, title_attn_mask: Tensor, history_mask: Tensor,
                sapo_encoding: Union[Tensor, None] = None, sapo_attn_mask: Union[Tensor, None] = None,
                category_encoding: Union[Tensor, None] = None):
        r"""
        Forward propagation
        Args:
            title_encoding: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            history_mask: tensor of shape ``(batch_size, num_clicked_news)``.
            sapo_encoding: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            category_encoding: tensor of shape ``(batch_size, num_clicked_news)``.
        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        batch_size = title_encoding.shape[0]
        num_clicked_news = title_encoding.shape[1]

        title_encoding = title_encoding.view(batch_size * num_clicked_news, -1)
        title_attn_mask = title_attn_mask.view(batch_size * num_clicked_news, -1)
        sapo_encoding = sapo_encoding.view(batch_size * num_clicked_news, -1)
        sapo_attn_mask = sapo_attn_mask.view(batch_size * num_clicked_news, -1)
        category_encoding = category_encoding.view(batch_size * num_clicked_news)

        news_repr = self.news_encoder(title_encoding=title_encoding, title_attn_mask=title_attn_mask,
                                      sapo_encoding=sapo_encoding, sapo_attn_mask=sapo_attn_mask,
                                      category_encoding=category_encoding)
        news_repr = news_repr.view(batch_size, num_clicked_news, -1)
        news_attn, _ = self.multi_head_self_attn(query=news_repr, key=news_repr, value=news_repr,
                                                 key_padding_mask=~history_mask)
        user_repr = self.additive_attn(news_attn, history_mask)

        return user_repr
    