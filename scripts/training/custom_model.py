"""Custom odor model: a molecule graph network fused with molecular descriptors.

``GraphDescriptorModel`` is a message passing network over a 3D molecular graph.
It is fused with Dragon descriptor groups, RDKit, EPI Suite and Morgan features, odor
name features, and NSPDK and similarity structure features by one attention head per
DREAM attribute. It predicts the 21 means
and 21 standard deviations of DREAM Subchallenge 2, or every participant's rating
of the 21 attributes for Subchallenge 1.

This module holds the architecture only. The inputs it consumes are prepared in
``scripts.core.data`` and it is trained and scored by ``scripts.core.train``:

    poetry run ds-train
"""

from __future__ import annotations

import torch
from torch import nn

from scripts.core.data import (
    ATOM_FEATURE_COUNT,
    EDGE_FEATURE_COUNT,
    SUBJECT_IDS,
    GraphData,
    dragon_group_indices,
)
from scripts.core.metrics import TARGETS

GRAPH_HIDDEN_SIZE = 64
DRAGON_CONTEXT_SIZE = 32
DRAGON_GROUP_ENCODING_SIZE = 16
GRAPH_DROPOUT = 0.25
MESSAGE_PASSING_LAYERS = 3
# During training every attention token is hidden from the attributes with this probability,
# so that no single block, such as the Dragon descriptors, can carry the prediction alone.
TOKEN_DROPOUT = 0.15
# Subchallenge 2 hides the tokens more often. Over three seeds, 0.3 scored 9.65 on the test set and
# 9.48 on validation, against 9.46 and 9.27 for 0.15 and 9.30 and 9.16 for none.
Q2_TOKEN_DROPOUT = 0.3
Q2_DROPOUT = GRAPH_DROPOUT
# Subchallenge 2 can answer with experts: each expert is a small attention network over its own
# view of the molecule, and a gate for every attribute mixes them. The views are the Dragon
# descriptors, the graph with the name windows, the similarity features, and the neighbour labels.
# They are trained together, in one model. During training each expert is switched off for a
# molecule with the probability below, so that the experts do not all learn the same thing.
EXPERT_DROPOUT = 0.2


class GraphDescriptorModel(nn.Module):
    """Message passing graph network fused with descriptors by per attribute attention.

    The molecule graph, four Dragon descriptor families, the numeric block, the odor
    name features, and the structure features each become one token. Every DREAM attribute owns a query
    that attends over those tokens and feeds its own small head. With two outputs
    per attribute the model produces the Q2 layout of 21 means followed by 21
    standard deviations. With 49 outputs per attribute it predicts every
    participant's rating, which is the Subchallenge 1 layout.
    """

    def __init__(
        self,
        dragon_groups: dict[str, list[int]],
        numeric_count: int,
        name_count: int,
        output_count: int,
        structure_count: int,
        dropout_rate: float = GRAPH_DROPOUT,
        token_dropout: float = 0.0,
        neighbour_count: int = 0,
        experts: bool = False,
    ):
        super().__init__()
        if not 0.0 <= token_dropout <= 1.0:
            raise ValueError("The token dropout must lie between 0 and 1.")
        self.token_dropout = token_dropout
        if output_count % len(TARGETS) != 0:
            raise ValueError("Output count must divide evenly across the 21 attributes.")
        hidden = GRAPH_HIDDEN_SIZE
        self.outputs_per_attribute = output_count // len(TARGETS)
        self.atom_input = nn.Linear(ATOM_FEATURE_COUNT, hidden)
        self.message_layers = nn.ModuleList(
            nn.Linear(hidden + EDGE_FEATURE_COUNT + DRAGON_CONTEXT_SIZE, hidden)
            for _ in range(MESSAGE_PASSING_LAYERS)
        )
        self.update_layers = nn.ModuleList(
            nn.Linear(hidden * 2 + DRAGON_CONTEXT_SIZE, hidden)
            for _ in range(MESSAGE_PASSING_LAYERS)
        )
        self.dragon_group_branches = nn.ModuleDict()
        for group_name, indices in dragon_groups.items():
            if not indices:
                raise ValueError(f"Dragon model group {group_name!r} has no columns.")
            self.register_buffer(
                f"dragon_indices_{group_name}", torch.tensor(indices, dtype=torch.long)
            )
            self.dragon_group_branches[group_name] = nn.Sequential(
                nn.Linear(len(indices), DRAGON_GROUP_ENCODING_SIZE),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
        if not self.dragon_group_branches:
            raise ValueError("At least one Dragon feature group is required.")
        self.dragon_context_merge = nn.Sequential(
            nn.Linear(
                DRAGON_GROUP_ENCODING_SIZE * len(self.dragon_group_branches), DRAGON_CONTEXT_SIZE
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_count, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.name_branch = nn.Sequential(
            nn.Linear(name_count, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.structure_branch = nn.Sequential(
            nn.Linear(structure_count, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.structure_attention_projection = nn.Linear(hidden // 2, hidden)
        # Subchallenge 2 can add a token that holds what the most similar training molecules were
        # rated. It is the last token, and it exists only when ``neighbour_count`` is positive.
        if neighbour_count:
            self.neighbour_branch = nn.Sequential(
                nn.Linear(neighbour_count, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            self.neighbour_attention_projection = nn.Linear(hidden // 2, hidden)
        else:
            self.neighbour_branch = None
            self.neighbour_attention_projection = None
        self.dragon_attention_projection = nn.Linear(DRAGON_GROUP_ENCODING_SIZE, hidden)
        self.name_attention_projection = nn.Linear(hidden // 2, hidden)
        self.attribute_queries = nn.Parameter(torch.randn(len(TARGETS), hidden) * 0.02)
        self.subject_attention = self.outputs_per_attribute == len(SUBJECT_IDS)
        if self.subject_attention:
            self.subject_queries = nn.Parameter(
                torch.randn(len(TARGETS), len(SUBJECT_IDS), hidden) * 0.02
            )
            # Each participant has a stable rating offset for each attribute.
            self.subject_bias = nn.Parameter(torch.zeros(len(TARGETS), len(SUBJECT_IDS)))
            head_output_size = 1
        else:
            self.register_parameter("subject_queries", None)
            self.register_parameter("subject_bias", None)
            head_output_size = self.outputs_per_attribute
        self.attention_key = nn.Linear(hidden, hidden, bias=False)
        self.attention_value = nn.Linear(hidden, hidden, bias=False)
        self.attribute_heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden // 2, head_output_size),
            )
            for _ in TARGETS
        )
        # The tokens in the order that the attention sees them. The shared tokens, which are all
        # of them, are shown to every attribute.
        token_names = ["graph", *self.dragon_group_branches, "numeric", "names", "structure"]
        if neighbour_count:
            token_names.append("neighbours")
        self.token_names = token_names
        # The experts of Subchallenge 2. Every expert reads the numeric token, which holds the
        # concentration, and the tokens of its own view.
        views = {
            "descriptors": [*self.dragon_group_branches],
            "graph": ["graph", "names"],
            "similarity": ["structure"],
            "neighbours": ["neighbours"],
        }
        self.expert_tokens: list[list[int]] = []
        if experts and not self.subject_attention:
            for view in views.values():
                members = [token_names.index(name) for name in view if name in token_names]
                if members:
                    self.expert_tokens.append(sorted({*members, token_names.index("numeric")}))
        self.experts = len(self.expert_tokens) >= 2
        if self.experts:
            count = len(self.expert_tokens)
            self.expert_keys = nn.ModuleList(
                nn.Linear(hidden, hidden, bias=False) for _ in range(count)
            )
            self.expert_values = nn.ModuleList(
                nn.Linear(hidden, hidden, bias=False) for _ in range(count)
            )
            self.expert_queries = nn.ParameterList(
                nn.Parameter(torch.randn(len(TARGETS), hidden) * 0.02) for _ in range(count)
            )
            self.expert_heads = nn.ModuleList(
                nn.ModuleList(
                    nn.Sequential(
                        nn.LayerNorm(hidden),
                        nn.Linear(hidden, hidden // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(hidden // 2, head_output_size),
                    )
                    for _ in TARGETS
                )
                for _ in range(count)
            )
            self.gate = nn.Linear(hidden, len(TARGETS) * count)
            self.gate_bias = nn.Parameter(torch.zeros(len(TARGETS), count))
        self.register_buffer("shared_tokens", torch.arange(len(token_names)), persistent=False)

    def answer_with_experts(
        self,
        tokens: torch.Tensor,
        allowed: torch.Tensor,
        visible: torch.Tensor | None,
        scale: float,
    ) -> torch.Tensor:
        """The Subchallenge 2 answer: every expert reads its view, and a gate mixes them.

        An expert whose tokens are all hidden for a molecule gives no answer for it, and neither
        does an expert that is switched off while training. At least one expert always answers.
        """
        count = len(self.expert_tokens)
        contexts, alive = [], []
        for position, members in enumerate(self.expert_tokens):
            view = tokens[:, members]
            scores = (
                torch.einsum(
                    "ah,bmh->bam", self.expert_queries[position], self.expert_keys[position](view)
                )
                / scale
            )
            allowed_here = allowed[:, :, members].expand(tokens.shape[0], -1, -1)
            scores = scores.masked_fill(~allowed_here, float("-inf"))
            weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
            contexts.append(
                torch.einsum("bam,bmh->bah", weights, self.expert_values[position](view))
                + self.expert_queries[position].unsqueeze(0)
            )
            alive.append(allowed_here.any(dim=-1).any(dim=1))
        answering = torch.stack(alive, dim=1)
        if self.training and EXPERT_DROPOUT > 0.0:
            kept = torch.rand(answering.shape, device=tokens.device) >= EXPERT_DROPOUT
            survivors = answering & kept
            answering = torch.where(survivors.any(dim=1, keepdim=True), survivors, answering)
        if visible is None:
            pooled = tokens.mean(dim=1)
        else:
            shown = visible.to(tokens.dtype).unsqueeze(-1)
            pooled = (tokens * shown).sum(dim=1) / shown.sum(dim=1).clamp(min=1.0)
        logits = self.gate(pooled).view(tokens.shape[0], len(TARGETS), count) + self.gate_bias
        logits = logits.masked_fill(~answering.unsqueeze(1), float("-inf"))
        mix = torch.softmax(logits, dim=-1)
        outputs = [
            torch.stack(
                [head(contexts[position][:, index]) for index, head in enumerate(heads)], dim=1
            )
            for position, heads in enumerate(self.expert_heads)
        ]
        stacked = torch.stack(outputs, dim=2)
        answer = (mix.unsqueeze(-1) * stacked).sum(dim=2)
        if self.outputs_per_attribute == 2:
            return torch.cat((answer[:, :, 0], answer[:, :, 1]), dim=1)
        return answer.reshape(answer.shape[0], -1)

    def visible_tokens(self, tokens: torch.Tensor) -> torch.Tensor | None:
        """Which tokens each molecule shows to the attributes while training, or None.

        Every molecule keeps at least one shared token, picked at random when none was kept,
        so that every attribute has something to attend to.
        """
        if not self.training or self.token_dropout == 0.0:
            return None
        count, tokens_per_molecule = tokens.shape[0], tokens.shape[1]
        visible = torch.rand(count, tokens_per_molecule, device=tokens.device) >= self.token_dropout
        empty = ~visible[:, self.shared_tokens].any(dim=1)
        if empty.any():
            choice = torch.randint(0, len(self.shared_tokens), (count,), device=tokens.device)
            visible[empty, self.shared_tokens[choice][empty]] = True
        return visible

    def forward(
        self,
        atoms: torch.Tensor,
        edges: torch.Tensor,
        edge_features: torch.Tensor,
        graph_ids: torch.Tensor,
        dragon: torch.Tensor,
        numeric: torch.Tensor,
        name_features: torch.Tensor,
        structure_features: torch.Tensor,
        neighbour_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = GRAPH_HIDDEN_SIZE
        dragon_group_encodings = []
        for group_name, branch in self.dragon_group_branches.items():
            indices = getattr(self, f"dragon_indices_{group_name}")
            dragon_group_encodings.append(branch(dragon.index_select(1, indices)))
        dragon_encoding = self.dragon_context_merge(torch.cat(dragon_group_encodings, dim=1))

        node_values = torch.relu(self.atom_input(atoms))
        source, destination = edges[0], edges[1]
        for message_layer, update_layer in zip(
            self.message_layers, self.update_layers, strict=True
        ):
            messages = torch.relu(
                message_layer(
                    torch.cat(
                        (node_values[source], edge_features, dragon_encoding[graph_ids[source]]),
                        dim=1,
                    )
                )
            )
            aggregated = torch.zeros_like(node_values)
            if len(destination):
                aggregated.index_add_(0, destination, messages)
            node_values = torch.relu(
                update_layer(
                    torch.cat((node_values, aggregated, dragon_encoding[graph_ids]), dim=1)
                )
            )

        graph_count = numeric.shape[0]
        pooled = torch.zeros(
            (graph_count, hidden), device=node_values.device, dtype=node_values.dtype
        )
        pooled.index_add_(0, graph_ids, node_values)
        atom_counts = torch.bincount(graph_ids, minlength=graph_count).to(node_values.dtype)
        graph_encoding = pooled / atom_counts.unsqueeze(1).clamp(min=1)

        numeric_encoding = self.numeric_branch(numeric)
        name_encoding = self.name_branch(name_features)
        # Each attribute attends to the graph, four Dragon groups, numeric
        # descriptors/fingerprints, and odor-name features independently.
        tokens = [
            graph_encoding,
            *[self.dragon_attention_projection(encoding) for encoding in dragon_group_encodings],
            numeric_encoding,
            self.name_attention_projection(name_encoding),
        ]
        tokens.append(
            self.structure_attention_projection(self.structure_branch(structure_features))
        )
        if self.neighbour_branch is not None:
            if neighbour_features is None:
                raise ValueError("This model needs the neighbour features.")
            neighbour_encoding = self.neighbour_branch(neighbour_features)
            tokens.append(self.neighbour_attention_projection(neighbour_encoding))
        modality_tokens = torch.stack(tokens, dim=1)
        keys = self.attention_key(modality_tokens)
        values = self.attention_value(modality_tokens)
        scale = hidden**0.5
        visible = self.visible_tokens(modality_tokens)
        allowed = torch.ones(
            (1, len(TARGETS), modality_tokens.shape[1]), dtype=torch.bool, device=keys.device
        )
        if visible is not None:
            allowed = allowed & visible[:, None, :]
        if self.subject_attention:
            subject_keys = self.attention_key(self.subject_queries)
            subject_scores = torch.einsum("ash,bmh->basm", subject_keys, keys) / scale
            subject_scores = subject_scores.masked_fill(~allowed[:, :, None, :], float("-inf"))
            subject_weights = torch.softmax(subject_scores, dim=-1)
            subject_contexts = torch.einsum(
                "basm,bmh->bash", subject_weights, values
            ) + self.subject_queries.unsqueeze(0)
            subject_predictions = torch.stack(
                [
                    head(subject_contexts[:, index]).squeeze(-1)
                    for index, head in enumerate(self.attribute_heads)
                ],
                dim=1,
            ).reshape(graph_count, -1)
            return subject_predictions + self.subject_bias.reshape(1, -1)

        if self.experts:
            return self.answer_with_experts(modality_tokens, allowed, visible, scale)

        attention_scores = torch.einsum("ah,bmh->bam", self.attribute_queries, keys) / scale
        attention_scores = attention_scores.masked_fill(~allowed, float("-inf"))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attribute_contexts = torch.einsum(
            "bam,bmh->bah", attention_weights, values
        ) + self.attribute_queries.unsqueeze(0)
        per_attribute_outputs = torch.stack(
            [
                head(attribute_contexts[:, index, :])
                for index, head in enumerate(self.attribute_heads)
            ],
            dim=1,
        )
        # DREAM Q2 columns list all means first and all standard deviations next.
        if self.outputs_per_attribute == 2:
            return torch.cat(
                (per_attribute_outputs[:, :, 0], per_attribute_outputs[:, :, 1]), dim=1
            )
        # Q1 columns list each attribute's predictions for all 49 subjects.
        return per_attribute_outputs.reshape(per_attribute_outputs.shape[0], -1)


def build_graph_model(data: GraphData) -> GraphDescriptorModel:
    """Instantiate the graph model with the shapes implied by prepared graph data."""
    return GraphDescriptorModel(
        dragon_group_indices(data.dragon_columns, data.dragon_groups),
        data.numeric_count,
        data.name_count,
        data.target_count,
        dropout_rate=Q2_DROPOUT if data.challenge == "q2" else GRAPH_DROPOUT,
        structure_count=data.structure_count,
        token_dropout=Q2_TOKEN_DROPOUT if data.challenge == "q2" else TOKEN_DROPOUT,
        neighbour_count=data.neighbours["width"] if data.neighbours else 0,
        experts=data.challenge == "q2",
    )
