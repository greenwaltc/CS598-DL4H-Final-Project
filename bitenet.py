from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from enum import Enum
from functools import reduce
from operator import mul

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskDirection(Enum):
    FORWARD = 'forward'
    BACKWARD = 'backward'
    DIAGONAL = 'diagonal'
    NONE = 'none'


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.scale = nn.parameter.Parameter(torch.ones(normalized_shape, dtype=torch.float32, device=device))
        self.bias = nn.parameter.Parameter(torch.zeros(normalized_shape, dtype=torch.float32, device=device))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor, eps=1e-5):
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.mean(torch.square(x - mean), dim=-1, keepdim=True)
        norm_x = (x - mean) * torch.rsqrt(variance + eps)
        return norm_x * self.scale + self.bias


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, keep: int):
        fixed_shape = list(x.size())
        start = len(fixed_shape) - keep
        left = reduce(mul, [fixed_shape[i] or x.shape[i] for i in range(start)])
        out_shape = [left] + [fixed_shape[i] or x.shape[i] for i in range(start, len(fixed_shape))]
        return torch.reshape(x, out_shape)


class Unflatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v: torch.Tensor, ref: torch.Tensor, embedding_dim):
        batch_size = ref.shape[0]
        n_visits = ref.shape[1]
        out = torch.reshape(v, [batch_size, n_visits, embedding_dim])
        return out


class AttentionPooling(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        )

    def forward(self, inputs):
        x, mask = inputs
        x = self.fc(x)
        x[~mask] = VERY_NEGATIVE_NUMBER
        soft = F.softmax(x, dim=1)
        x[~mask] = 0
        attn_output = torch.sum(soft * x, 1)
        return attn_output


class MultiHeadAttention(nn.Module):
    def __init__(self, direction, dropout, n_units, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.direction = direction
        self.n_units = n_units
        self.q_linear = nn.Linear(n_units, n_units, bias=False)
        self.k_linear = nn.Linear(n_units, n_units, bias=False)
        self.v_linear = nn.Linear(n_units, n_units, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):

        # because of self-attention, queries and keys is equal to inputs
        input_tensor, input_mask = inputs
        queries = input_tensor
        keys = input_tensor

        # Linear projections
        Q = self.q_linear(queries)  # (N, L_q, d)
        K = self.k_linear(keys)  # (N, L_k, d)
        V = self.v_linear(keys)  # (N, L_k, d)

        # Split and concat
        assert self.n_units % self.n_heads == 0
        Q_ = torch.cat(torch.split(Q, self.n_units // self.n_heads, dim=2), dim=0)  # (h*N, L_q, d/h)
        K_ = torch.cat(torch.split(K, self.n_units // self.n_heads, dim=2), dim=0)  # (h*N, L_k, d/h)
        V_ = torch.cat(torch.split(V, self.n_units // self.n_heads, dim=2), dim=0)  # (h*N, L_k, d/h)

        # Multiplication
        outputs = torch.matmul(Q_, torch.permute(K_, [0, 2, 1]))  # (h*N, L_q, L_k)

        # Scale
        outputs = outputs / (list(K_.shape)[-1] ** 0.5)  # (h*N, L_q, L_k)

        # Key Masking
        key_masks = torch.sign(torch.sum(torch.abs(K_), dim=-1))  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1)  # (h*N, 1, T_k)
        key_masks = torch.tile(key_masks, [1, list(Q_.shape)[1], 1])  # (h*N, T_q, T_k)

        # Apply masks to outputs
        paddings = torch.ones_like(outputs, device=device) * (-2 ** 32 + 1)  # exp mask
        outputs = torch.where(key_masks == 0, paddings, outputs)  # (h*N, T_q, T_k)

        n_visits = list(input_tensor.shape)[1]
        sw_indices = torch.arange(0, n_visits, dtype=torch.int32, device=device)
        sw_col, sw_row = torch.meshgrid(sw_indices, sw_indices)
        if self.direction == MaskDirection.DIAGONAL:
            # shape of (n_visits, n_visits)
            attention_mask = (torch.diag(- torch.ones([n_visits], dtype=torch.int32, device=device)) + 1).bool()
        elif self.direction == MaskDirection.FORWARD:
            attention_mask = torch.greater(sw_row, sw_col)  # shape of (n_visits, n_visits)
        else:  # MaskDirection.BACKWARD
            attention_mask = torch.greater(sw_col, sw_row)  # shape of (n_visits, n_visits)
        adder = (1.0 - attention_mask.type(outputs.dtype)) * -10000.0
        outputs += adder

        # softmax
        outputs = F.softmax(outputs, -1)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.sum(torch.abs(Q_), dim=-1))  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, -1)  # (h*N, T_q, 1)
        query_masks = torch.tile(query_masks, [1, 1, list(K_.shape)[1]])  # (h*N, T_q, T_k)

        # Apply masks to outputs
        outputs = outputs * query_masks

        # Dropouts
        outputs = self.dropout(outputs)
        # Weighted sum
        outputs = torch.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = torch.cat(torch.split(outputs, outputs.shape[0] // self.n_heads, dim=0), dim=2)  # (N, L_q, d)

        # input padding
        val_mask = torch.unsqueeze(input_mask, -1)
        outputs = torch.multiply(outputs, val_mask.float())

        return outputs


class PrePostProcessingWrapper(nn.Module):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, module: nn.Module, normalized_shape: int):
        super().__init__()
        self.module = module
        self.layer_norm = LayerNorm(normalized_shape)

    def forward(self, inputs):
        """Calls wrapped layer with same parameters."""

        x, mask = inputs
        try:
            y = self.module((x, mask))
        except:
            y = self.module(x)
        return self.layer_norm(x + y)


class MaskEnc(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            n_heads: int,
            dropout: float = 0.1,
            temporal_mask_direction: MaskDirection = MaskDirection.NONE,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.temporal_mask_direction = temporal_mask_direction

        self.attention = PrePostProcessingWrapper(
            module=MultiHeadAttention(
                direction=temporal_mask_direction,
                dropout=dropout,
                n_units=embedding_dim,
                n_heads=n_heads
            ),
            normalized_shape=embedding_dim
        )

        self.fc = PrePostProcessingWrapper(
            module=nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim)
            ),
            normalized_shape=embedding_dim
        )

        self.output_normalization = LayerNorm(embedding_dim)

    def forward(self, inputs):
        x, mask = inputs

        out = self.attention((x, mask))
        out = self.fc((out, mask))
        out = self.output_normalization(out)
        return out, mask

    def _make_temporal_mask(self, n: int) -> Optional[torch.Tensor]:
        if self.temporal_mask_direction == MaskDirection.NONE:
            return None
        if self.temporal_mask_direction == MaskDirection.FORWARD:
            return torch.tril(torch.full((n, n), -10000, device=device)).fill_diagonal_(0).float()
        if self.temporal_mask_direction == MaskDirection.BACKWARD:
            return torch.triu(torch.full((n, n), -10000, device=device)).fill_diagonal_(0).float()
        if self.temporal_mask_direction == MaskDirection.DIAGONAL:
            return torch.zeros(n, n, device=device).fill_diagonal_(-10000).float()


class _BiteNet(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 128,
            n_heads: int = 4,
            dropout: float = 0.1,
            n_mask_enc_layers: int = 2,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.flatten = Flatten()
        self.unflatten = Unflatten()

        def _make_mask_enc_block(temporal_mask_direction: MaskDirection = MaskDirection.NONE):
            return MaskEnc(
                embedding_dim=embedding_dim,
                n_heads=n_heads,
                dropout=dropout,
                temporal_mask_direction=temporal_mask_direction,
            )

        self.code_attn = nn.Sequential()
        self.visit_attn_fw = nn.Sequential()
        self.visit_attn_bw = nn.Sequential()
        for _ in range(n_mask_enc_layers):
            self.code_attn.append(_make_mask_enc_block(MaskDirection.DIAGONAL))
            self.visit_attn_fw.append(_make_mask_enc_block(MaskDirection.FORWARD))
            self.visit_attn_bw.append(_make_mask_enc_block(MaskDirection.BACKWARD))

        # Attention pooling layers
        self.code_attn.append(AttentionPooling(embedding_dim))
        self.visit_attn_fw.append(AttentionPooling(embedding_dim))
        self.visit_attn_bw.append(AttentionPooling(embedding_dim))

        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(
            self,
            embedded_codes: torch.Tensor,
            codes_mask: torch.Tensor,
            visits_mask: torch.Tensor,
            embedded_intervals: torch.Tensor = None,
    ) -> torch.Tensor:

        # input tensor, reshape 4 dimension to 3
        flattened_codes = self.flatten(embedded_codes, 2)

        # input mask, reshape 3 dimension to 2
        flattened_codes_mask = self.flatten(codes_mask, 1)

        code_attn = self.code_attn((flattened_codes, flattened_codes_mask))
        code_attn = self.unflatten(code_attn, embedded_codes, self.embedding_dim)

        if embedded_intervals is not None:
            code_attn += embedded_intervals

        u_fw = self.visit_attn_fw((code_attn, visits_mask))
        u_bw = self.visit_attn_bw((code_attn, visits_mask))
        u_bi = torch.cat([u_fw, u_bw], dim=-1)

        s = self.fc(u_bi)
        return s


class BiteNet(BaseModel):
    def __init__(
            self,
            dataset: SampleDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            embedding_dim: int = 128,
            n_mask_enc_layers: int = 1,
            n_heads: int = 4,
            dropout: float = 0.1,
            **kwargs
    ):
        super().__init__(dataset, feature_keys, label_key, mode)

        # Any BaseModel should have these attributes, as functions like add_feature_transform_layer uses them
        self.feat_tokenizers = {}
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embedding_dim = embedding_dim

        # self.add_feature_transform_layer will create a transformation layer for each feature
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            self.add_feature_transform_layer(
                feature_key, input_info, special_tokens=["<pad>", "<unk>"]
            )

        # final output layer
        output_size = self.get_output_size(self.label_tokenizer)
        self.bite_net = _BiteNet(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
            n_mask_enc_layers=n_mask_enc_layers,
        )

        self.fc = nn.Linear(self.embedding_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:

        embeddings = []
        masks = []
        intervals_embeddings = None
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]

            # each patient's feature is represented by [[code1, code2],[code3]]
            assert input_info["dim"] == 3 and input_info["type"] == str
            feature_vals = kwargs[feature_key]

            x = self.feat_tokenizers[feature_key].batch_encode_3d(feature_vals, truncation=(False, False))
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            pad_idx = self.feat_tokenizers[feature_key].vocabulary("<pad>")

            # Create the mask
            mask = (x != pad_idx).long()
            embeds = self.embeddings[feature_key](x)

            if feature_key == "intervals":
                intervals_embeddings = embeds
            else:
                embeddings.append(embeds)
                masks.append(mask)

        code_embeddings = torch.cat(embeddings, dim=2)
        codes_mask = torch.cat(masks, dim=2)
        visits_mask = torch.where(torch.sum(codes_mask, dim=-1) != 0, 1, 0)

        output = self.bite_net(code_embeddings, codes_mask, visits_mask, intervals_embeddings.squeeze(2))
        logits = self.fc(output)

        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}
