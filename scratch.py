# ! pip install pyhealth
from pyhealth.datasets import MIMIC3Dataset, SampleDataset
from pyhealth.data import Visit
import pandas as pd
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import BaseModel
from pyhealth.trainer import Trainer
from pyhealth.metrics.binary import binary_metrics_fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from enum import Enum
from functools import reduce
from operator import mul

# Set this to the directory with all MIMIC-3 dataset files
data_root = "./data"


# Load the dataset

mimic3_ds = MIMIC3Dataset(
        root=data_root,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD"],
        dev=True
)


# Find all diagnoses codes
# Find all procedure codes
# Remove diagnoses codes with fewer than 5 occurences in the dataset

all_diag_codes = []
for patient_id, patient in mimic3_ds.patients.items():
  for i in range(len(patient)):
    visit: Visit = patient[i]
    conditions = visit.get_code_list(table="DIAGNOSES_ICD")
    all_diag_codes.extend(conditions)

codes = pd.Series(all_diag_codes)
diag_code_counts = codes.value_counts()
filtered_diag_codes = diag_code_counts[diag_code_counts > 4].index.values
num_unique_diag_codes = len(filtered_diag_codes)


# Define the tasks

DIAGNOSES_KEY = "conditions"
PROCEDURES_KEY = "procedures"
INTERVAL_DAYS_KEY = "days_since_first_visit"

def flatten(l: List):
    return [item for sublist in l for item in sublist]

def patient_level_readmission_prediction(patient, time_window=30):
    """
    patient is a <pyhealth.data.Patient> object
    """
    samples = []

    # if the patient only has one visit, we drop it
    if len(patient) <= 2:
        return []

    sorted_visits = sorted(patient, key=lambda visit: visit.encounter_time)

    # step 1: define label
    idx_last_visit = len(sorted_visits)-1
    last_visit: Visit = sorted_visits[idx_last_visit]
    second_to_last_visit: Visit = sorted_visits[idx_last_visit - 1]
    first_visit: Visit = sorted_visits[0]

    time_diff = (last_visit.encounter_time - second_to_last_visit.encounter_time).days
    readmission_label = 1 if time_diff < time_window else 0

    # step 2: obtain features
    visits_conditions = []
    visits_procedures = []
    visits_intervals = []
    for idx, visit in enumerate(sorted_visits):
        if idx == len(sorted_visits) - 1: break
        conditions = [c for c in visit.get_code_list(table="DIAGNOSES_ICD") if c in filtered_diag_codes]
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        time_diff_from_first_visit = (visit.encounter_time - first_visit.encounter_time).days

        if len(conditions) * len(procedures) == 0:
            continue

        visits_conditions.append(conditions)
        visits_procedures.append(procedures)
        visits_intervals.append([str(time_diff_from_first_visit)])

    unique_conditions = list(set(flatten(visits_conditions)))
    unique_procedures = list(set(flatten(visits_procedures)))

    # step 3: exclusion criteria
    if len(unique_conditions) * len(unique_procedures) == 0:
        return []

    # step 4: assemble the sample
    samples.append(
        {
            "patient_id": patient.patient_id,
            "visit_id": visit.visit_id,
            "conditions": visits_conditions,
            "procedures": visits_procedures,
            "intervals": visits_intervals,
            "label": readmission_label,
        }
    )
    return samples


# Create the task datasets
mimic3_dxtx = mimic3_ds.set_task(task_fn=patient_level_readmission_prediction)


BATCH_SIZE = 32
train, val, test = split_by_patient(mimic3_dxtx, [0.8, 0.1, 0.1])

train_loader = get_dataloader(train, batch_size=BATCH_SIZE, shuffle=False) # Switch back
val_loader = get_dataloader(val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = get_dataloader(test, batch_size=BATCH_SIZE, shuffle=False)

# Define the models
VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

class MaskDirection(Enum):
    FORWARD = 'forward'
    BACKWARD = 'backward'
    DIAGONAL = 'diagonal'
    NONE = 'none'

class MaskedLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.scale = nn.parameter.Parameter(torch.ones(normalized_shape, dtype=torch.float32))
        self.bias = nn.parameter.Parameter(torch.zeros(normalized_shape, dtype=torch.float32))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor, eps = 1e-5):
        mean = torch.nanmean(x, dim=-1, keepdim=True)
        variance = torch.nanmean(torch.square(x - mean), dim=-1, keepdim=True)
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

    def forward(self, input):
        x = self.fc(input)
        x = torch.nan_to_num(x, nan=VERY_NEGATIVE_NUMBER)
        soft = F.softmax(x, dim=1)
        attn_output = torch.nansum(soft * input, 1)
        return attn_output


class MaskEnc(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            dropout: float = 0.1,
            batch_first: bool = True,
            temporal_mask_direction: MaskDirection = MaskDirection.NONE,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.temporal_mask_direction = temporal_mask_direction

        self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=batch_first
            )

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.layer_norm1 = MaskedLayerNorm(embedding_dim)
        self.layer_norm2 = MaskedLayerNorm(embedding_dim)

    def forward(self, inputs):
        x, key_padding_mask = inputs
        attn_mask = self._make_temporal_mask(x.shape[1])
        x = torch.nan_to_num(x, nan=0)

        attn_output, attn_output_weights = self.attention(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        attn_output = self.layer_norm1(x + attn_output)
        out = self.fc(attn_output)
        out = self.layer_norm2(out + attn_output)

        return out, key_padding_mask

    def _make_temporal_mask(self, n: int) -> Optional[torch.Tensor]:
        if self.temporal_mask_direction == MaskDirection.NONE:
            return None
        if self.temporal_mask_direction == MaskDirection.FORWARD:
            return torch.tril(torch.ones(n,n)).bool()
        if self.temporal_mask_direction == MaskDirection.BACKWARD:
            return torch.triu(torch.ones(n,n)).bool()
        if self.temporal_mask_direction == MaskDirection.DIAGONAL:
            return torch.ones(n,n).fill_diagonal_(0).bool()


class BiteNet(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 128,
            num_heads: int = 4,
            dropout: float = 0.1,
            batch_first: bool = True,
            n_mask_enc_layers: int = 2,
            use_procedures: bool = True,
            use_intervals: bool = True,
    ):
        super().__init__()

        self.use_intervals = use_intervals
        self.use_procedures = use_procedures
        self.embedding_dim = embedding_dim

        self.flatten = Flatten()
        self.unflatten = Unflatten()

        def _make_mask_enc_block(temporal_mask_direction: MaskDirection = MaskDirection.NONE):
            return MaskEnc(
                embedding_dim = embedding_dim,
                num_heads = num_heads,
                dropout = dropout,
                batch_first = batch_first,
                temporal_mask_direction = temporal_mask_direction,
            )

        self.code_attn = nn.Sequential()
        self.visit_attn_fw = nn.Sequential()
        self.visit_attn_bw = nn.Sequential()
        for _ in range(n_mask_enc_layers):
            self.code_attn.append(_make_mask_enc_block(MaskDirection.DIAGONAL))
            self.visit_attn_fw.append(_make_mask_enc_block(MaskDirection.FORWARD))
            self.visit_attn_bw.append(_make_mask_enc_block(MaskDirection.BACKWARD))

        # Attention pooling layers
        self.code_attn_pooling = AttentionPooling(embedding_dim)
        self.visit_attn_bw_pooling = AttentionPooling(embedding_dim)
        self.visit_attn_fw_pooling = AttentionPooling(embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(
            self,
            embedded_codes: torch.Tensor,
            embedded_intervals: torch.Tensor,
            codes_mask: torch.Tensor,
            visits_mask: torch.Tensor,
    ) -> torch.Tensor:

        codes_mask = ~(codes_mask.bool())

        # input tensor, reshape 4 dimension to 3
        flattened_codes = self.flatten(embedded_codes, 2)

        # input mask, reshape 3 dimension to 2
        flattened_codes_mask = self.flatten(codes_mask, 1)

        code_attn, _ = self.code_attn((flattened_codes, flattened_codes_mask))
        code_attn = self.code_attn_pooling(code_attn)
        code_attn = self.unflatten(code_attn, embedded_codes, self.embedding_dim)

        if self.use_intervals:
            code_attn += embedded_intervals

        visits_mask = ~(visits_mask.bool())

        u_fw, _ = self.visit_attn_fw((code_attn, visits_mask))
        u_fw = self.visit_attn_fw_pooling(u_fw)

        u_bw, _ = self.visit_attn_bw((code_attn, visits_mask))
        u_bw = self.visit_attn_bw_pooling(u_bw)

        u_bi = torch.cat([u_fw, u_bw], dim=-1)
        s = self.fc(u_bi)
        return s

class PyHealthBiteNet(BaseModel):
    def __init__(
            self,
            dataset: SampleDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            embedding_dim: int = 128,
            n_mask_enc_layers: int = 2,
            use_intervals: bool = True,
            use_procedures: bool = True,
            num_heads: int = 4,
            dropout: float = 0.1,
            batch_first: bool = True,
            **kwargs
    ):
        super().__init__(dataset, feature_keys, label_key, mode)

        self.use_intervals = use_intervals
        self.use_procedures = use_procedures

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
        self.bite_net = BiteNet(
            embedding_dim = embedding_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = batch_first,
            use_intervals=use_intervals,
            use_procedures=use_procedures,
            n_mask_enc_layers=n_mask_enc_layers
        )

        self.fc = nn.Linear(self.embedding_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:

        embeddings = {}
        masks = {}
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]

            # each patient's feature is represented by [[code1, code2],[code3]]
            assert input_info["dim"] == 3 and input_info["type"] == str
            feature_vals = kwargs[feature_key]

            x = self.feat_tokenizers[feature_key].batch_encode_3d(feature_vals, truncation=(False, False))
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            pad_idx = self.feat_tokenizers[feature_key].vocabulary("<pad>")
            #create the mask
            mask = (x != pad_idx).long()
            embeds = self.embeddings[feature_key](x)
            embeddings[feature_key] = embeds
            masks[feature_key] = mask

        embedded_codes = embeddings['conditions']
        codes_mask = masks['conditions']
        if self.use_procedures:
            embedded_codes = torch.cat((embedded_codes, embeddings['procedures']), dim=2)
            codes_mask = torch.cat((codes_mask, masks['procedures']), dim=2)

        output = self.bite_net(embedded_codes, embeddings['intervals'].squeeze(2), codes_mask, masks['intervals'].squeeze(-1))
        logits = self.fc(output)

        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        for p, name in self.named_parameters():
            print(name)
            print(p)

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}

model_dxtx = PyHealthBiteNet(
    dataset = mimic3_dxtx,
    feature_keys = ['procedures', 'conditions', 'intervals'],
    label_key = "label",
    mode = "binary",
    embedding_dim=4
)

data = next(iter(train_loader))
model_dxtx(**data)


trainer_dxtx = Trainer(model=model_dxtx)
trainer_dxtx.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=1,
    monitor="pr_auc",
    optimizer_class=torch.optim.RMSprop
)

while True:
    data = next(iter(train_loader))
    model_dxtx(**data)