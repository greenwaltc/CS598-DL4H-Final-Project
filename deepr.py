from pyhealth.models import DeeprLayer
from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
import torch
import torch.nn as nn
from typing import List, Dict

class Deepr(BaseModel):
    def __init__(
            self,
            dataset: SampleDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            embedding_dim: int = 128
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
        self.deepr = DeeprLayer(
            feature_size=self.embedding_dim,
            hidden_size=self.embedding_dim
        )

        self.fc = nn.Linear(self.embedding_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:

        embeddings = []
        masks = []
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

            embeddings.append(embeds)
            masks.append(mask)

        code_embeddings = torch.cat(embeddings, dim=2)
        visit_embeddings = torch.sum(code_embeddings, dim=2)
        codes_mask = torch.cat(masks, dim=2)
        visits_mask = torch.where(torch.sum(codes_mask, dim=-1) != 0, 1, 0)

        output = self.deepr(visit_embeddings, visits_mask)

        logits = self.fc(output)

        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}