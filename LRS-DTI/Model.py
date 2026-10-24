
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GINEConv, HeteroConv, global_add_pool
from torch_geometric.utils import softmax


SUBGRAPH_NODE_TYPES = ["drug", "disease", "side_effect", "target"]
TYPE_TO_INDEX = {node_type: idx for idx, node_type in enumerate(SUBGRAPH_NODE_TYPES)}
METAPATH_ROLE_DIM = 8
GLOBAL_RELATION_DIM = 6
STRUCTURAL_DIM = 3
SUBGRAPH_EDGE_TYPES = [
    ("drug", "drug_interacts", "drug"),
    ("drug", "treats", "disease"),
    ("disease", "rev_treats", "drug"),
    ("drug", "causes_se", "side_effect"),
    ("side_effect", "rev_causes_se", "drug"),
    ("target", "target_interacts", "target"),
    ("target", "assoc_disease", "disease"),
    ("disease", "rev_assoc_disease", "target"),
    ("drug", "interacts", "target"),
    ("target", "rev_interacts", "drug"),
]


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class PairSubgraphGATEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, heads=4, num_layers=2):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")

        input_dim = len(SUBGRAPH_NODE_TYPES) + 2 + STRUCTURAL_DIM + METAPATH_ROLE_DIM + GLOBAL_RELATION_DIM
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        out_channels = hidden_dim // heads
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: GATConv(
                            (hidden_dim, hidden_dim),
                            out_channels,
                            heads=heads,
                            concat=True,
                            dropout=dropout,
                            add_self_loops=False,
                        )
                        for edge_type in SUBGRAPH_EDGE_TYPES
                    },
                    aggr="sum",
                )
            )
            self.norms.append(
                nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in SUBGRAPH_NODE_TYPES})
            )

        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.node_gate = nn.Linear(hidden_dim, hidden_dim)
        self.query_gate = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def get_batch(store):
        batch = getattr(store, "batch", None)
        if batch is None:
            return torch.zeros(store.num_nodes, dtype=torch.long, device=store.node_id.device)
        return batch

    @staticmethod
    def build_anchor(store):
        return torch.cat(
            [
                store.is_source.float().unsqueeze(-1),
                store.is_target.float().unsqueeze(-1),
            ],
            dim=-1,
        )

    def build_structural_features(self, store, graph_max_hops):
        batch = self.get_batch(store)
        max_hops = graph_max_hops[batch].unsqueeze(-1).clamp_min(1.0)
        return torch.cat(
            [
                store.min_depth.float().unsqueeze(-1) / max_hops,
                store.max_depth.float().unsqueeze(-1) / max_hops,
                store.dist_to_target.float().unsqueeze(-1) / max_hops,
            ],
            dim=-1,
        )

    def build_input_dict(self, subgraphs):
        x_dict = {}
        for node_type in SUBGRAPH_NODE_TYPES:
            store = subgraphs[node_type]
            num_nodes = int(store.num_nodes)
            features = torch.zeros(
                (
                    num_nodes,
                    len(SUBGRAPH_NODE_TYPES) + 2 + STRUCTURAL_DIM + METAPATH_ROLE_DIM + GLOBAL_RELATION_DIM,
                ),
                dtype=torch.float32,
                device=store.node_id.device,
            )
            if num_nodes > 0:
                features[:, TYPE_TO_INDEX[node_type]] = 1.0
                features[:, len(SUBGRAPH_NODE_TYPES) : len(SUBGRAPH_NODE_TYPES) + 2] = self.build_anchor(store)
                structural_start = len(SUBGRAPH_NODE_TYPES) + 2
                role_start = structural_start + STRUCTURAL_DIM
                relation_start = role_start + METAPATH_ROLE_DIM
                features[:, structural_start:role_start] = self.build_structural_features(
                    store, subgraphs.max_hops.float()
                )
                features[:, role_start:relation_start] = store.metapath_role.float()
                features[:, relation_start:] = store.global_relation_degree.float()
            x_dict[node_type] = self.input_mlp(features)
        return x_dict

    def get_query_states(self, x_dict, subgraphs, num_graphs):
        hidden_dim = next(iter(x_dict.values())).size(-1)
        query_drug = next(iter(x_dict.values())).new_zeros((num_graphs, hidden_dim))
        query_target = next(iter(x_dict.values())).new_zeros((num_graphs, hidden_dim))

        drug_store = subgraphs["drug"]
        drug_batch = self.get_batch(drug_store)
        drug_mask = drug_store.is_source.bool()
        if drug_mask.any():
            query_drug[drug_batch[drug_mask]] = x_dict["drug"][drug_mask]

        target_store = subgraphs["target"]
        target_batch = self.get_batch(target_store)
        target_mask = target_store.is_target.bool()
        if target_mask.any():
            query_target[target_batch[target_mask]] = x_dict["target"][target_mask]

        return query_drug, query_target

    def build_query_repr(self, query_drug, query_target):
        interaction = query_drug * query_target
        diff = torch.abs(query_drug - query_target)
        return self.query_proj(torch.cat([query_drug, query_target, interaction, diff], dim=-1))

    def build_attention_context(self, x_dict, subgraphs, query_repr, num_graphs):
        all_x = []
        all_batch = []
        for node_type in SUBGRAPH_NODE_TYPES:
            x = x_dict[node_type]
            batch = self.get_batch(subgraphs[node_type])
            all_x.append(x)
            all_batch.append(batch)

        node_x = torch.cat(all_x, dim=0)
        node_batch = torch.cat(all_batch, dim=0)
        node_query = query_repr[node_batch]

        attn_logits = self.attn_score(torch.tanh(self.node_gate(node_x) + self.query_gate(node_query))).squeeze(-1)
        attn_weights = softmax(attn_logits, node_batch, num_nodes=num_graphs)
        return global_add_pool(node_x * attn_weights.unsqueeze(-1), node_batch, size=num_graphs)

    def forward(self, subgraphs):
        num_graphs = int(subgraphs.num_graphs)
        x_dict = self.build_input_dict(subgraphs)

        for conv, norm_dict in zip(self.convs, self.norms):
            updated = conv(x_dict, subgraphs.edge_index_dict)
            next_x_dict = {}
            for node_type in SUBGRAPH_NODE_TYPES:
                prev_x = x_dict[node_type]
                cur_x = updated.get(node_type)
                if cur_x is None:
                    next_x_dict[node_type] = prev_x
                else:
                    next_x_dict[node_type] = norm_dict[node_type](prev_x + F.gelu(cur_x))
            x_dict = next_x_dict

        query_drug, query_target = self.get_query_states(x_dict, subgraphs, num_graphs)
        query_repr = self.build_query_repr(query_drug, query_target)
        context_repr = self.build_attention_context(x_dict, subgraphs, query_repr, num_graphs)
        return self.output_proj(torch.cat([query_repr, context_repr], dim=-1))


class DrugEncoder(nn.Module):

    def __init__(self, hidden_dim=128, dropout=0.25, edge_dim=6):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(3))
        self.node_projection = nn.Sequential(
            nn.Linear(62, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        def make_mlp():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.convs = nn.ModuleList(
            [GINEConv(make_mlp(), edge_dim=edge_dim), GINEConv(make_mlp(), edge_dim=edge_dim)]
        )

    def forward(self, drugs):
        hidden = self.node_projection(drugs.x.float())
        layer_states = [hidden]
        edge_attr = drugs.edge_attr.float()
        for conv in self.convs:
            hidden = hidden + conv(hidden, drugs.edge_index, edge_attr)
            layer_states.append(hidden)

        layer_weights = torch.softmax(self.layer_logits, dim=0)
        return sum(weight * state for weight, state in zip(layer_weights, layer_states))

    def get_layer_weights(self):
        return torch.softmax(self.layer_logits.detach(), dim=0)


class TargetEncoder(nn.Module):

    def __init__(self, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.token_projection = nn.Sequential(
            nn.Linear(1280, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.region_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        groups=hidden_dim,
                    ),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in (3, 5, 9)
            ]
        )
        self.scale_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.region_alpha_logit = nn.Parameter(torch.tensor(-1.38629436))

    @staticmethod
    def build_masked_sequence(targets):
        sequence = targets[:, 1:, :].float()
        mask = sequence.abs().sum(dim=-1) > 0
        lengths = mask.long().sum(dim=1)
        valid_rows = lengths > 0
        if valid_rows.any():
            eos_positions = lengths[valid_rows] - 1
            mask[valid_rows, eos_positions] = False
        return sequence, mask

    @staticmethod
    def masked_mean(states, mask):
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(states.dtype)
        return (states * mask.unsqueeze(-1)).sum(dim=1) / count

    def forward(self, targets):
        sequence, mask = self.build_masked_sequence(targets)
        base_states = self.token_projection(sequence) * mask.unsqueeze(-1)
        base_global = self.masked_mean(base_states, mask)

        channel_first = base_states.transpose(1, 2)
        region_states = torch.stack(
            [encoder(channel_first).transpose(1, 2) for encoder in self.region_encoders],
            dim=1,
        )
        scale_weights = torch.softmax(self.scale_selector(base_global), dim=-1)
        selected_regions = (region_states * scale_weights[:, :, None, None]).sum(dim=1)
        region_alpha = torch.sigmoid(self.region_alpha_logit)
        residue_states = (base_states + region_alpha * selected_regions) * mask.unsqueeze(-1)
        return residue_states, mask, scale_weights, region_alpha


class AtomResidueInteraction(nn.Module):

    def __init__(self, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.atom_query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.residue_key = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, atom_states, atom_batch, residue_states, residue_mask):
        atom_queries = self.atom_query(atom_states)
        residue_keys = self.residue_key(residue_states)
        temperature = self.log_temperature.exp().clamp(0.25, 4.0)
        score_scale = math.sqrt(atom_states.size(-1)) * temperature
        joint_attention = atom_states.new_zeros(
            (atom_states.size(0), residue_states.size(1))
        )

        num_graphs = residue_states.size(0)
        for graph_index in range(num_graphs):
            atom_indices = torch.nonzero(atom_batch == graph_index, as_tuple=False).squeeze(-1)
            residue_indices = torch.nonzero(residue_mask[graph_index], as_tuple=False).squeeze(-1)
            if atom_indices.numel() == 0 or residue_indices.numel() == 0:
                continue
            local_logits = (
                atom_queries[atom_indices]
                @ residue_keys[graph_index, residue_indices].transpose(0, 1)
            ) / score_scale
            atom_to_residue = torch.softmax(local_logits, dim=1)
            residue_to_atom = torch.softmax(local_logits, dim=0)
            local_joint = torch.sqrt((atom_to_residue * residue_to_atom).clamp_min(1e-12))
            local_joint = local_joint / local_joint.sum().clamp_min(1e-8)
            joint_attention[atom_indices[:, None], residue_indices[None, :]] = local_joint

        atom_attention = joint_attention.sum(dim=1)
        residue_attention = residue_states.new_zeros(residue_mask.shape)
        residue_attention.index_add_(0, atom_batch, joint_attention)
        return joint_attention, atom_attention, residue_attention, temperature


class NetModel(nn.Module):

    def __init__(self, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.drug_encoder = DrugEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.target_encoder = TargetEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.atom_residue_interaction = AtomResidueInteraction(hidden_dim=hidden_dim, dropout=dropout)
        self.pair_subgraph_gat_encoder = PairSubgraphGATEncoder(hidden_dim=hidden_dim, dropout=dropout)

        self.joint_drug_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.joint_target_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.subgraph_residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.residual_alpha_logit = nn.Parameter(torch.tensor(-1.38629436))
        self.pair_predictor = nn.Linear(hidden_dim, 2)
        self.subgraph_predictor = nn.Linear(hidden_dim, 2)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.apply(init_weights)

    @staticmethod
    def pair_features(left, right):
        return torch.cat([left * right, torch.abs(left - right)], dim=-1)

    def forward(self, drugs, targets, pairs, subgraphs):
        batch_drugs = Batch.from_data_list(drugs[pairs[:, 0]])
        batch_targets = targets[pairs[:, 1]]

        atom_states = self.drug_encoder(batch_drugs)
        residue_states, residue_mask, scale_weights, scale_alpha = self.target_encoder(batch_targets)
        joint_attention, atom_attention, residue_attention, temperature = self.atom_residue_interaction(
            atom_states,
            batch_drugs.batch,
            residue_states,
            residue_mask,
        )

        joint_drug = global_add_pool(
            atom_states * atom_attention.unsqueeze(-1),
            batch_drugs.batch,
            size=batch_targets.size(0),
        )
        joint_target = (residue_states * residue_attention.unsqueeze(-1)).sum(dim=1)
        drug_repr = self.joint_drug_projection(joint_drug)
        target_repr = self.joint_target_projection(joint_target)
        pair_repr = self.pair_encoder(self.pair_features(drug_repr, target_repr))

        subgraph_repr = self.pair_subgraph_gat_encoder(subgraphs)
        pair_logits = self.pair_predictor(pair_repr)
        subgraph_logits = self.subgraph_predictor(subgraph_repr)
        fusion_input = torch.cat([pair_repr, subgraph_repr], dim=-1)
        residual_delta = self.subgraph_residual(fusion_input)
        residual_gate = self.residual_gate(fusion_input)
        residual_alpha = torch.sigmoid(self.residual_alpha_logit)
        main_logits = self.scorer(pair_repr + residual_alpha * residual_gate * residual_delta)

        return {
            "main_logits": main_logits,
            "pair_logits": pair_logits,
            "subgraph_logits": subgraph_logits,
            "residual_gate": residual_gate,
            "residual_alpha": residual_alpha,
            "target_scale_weights": scale_weights,
            "target_scale_alpha": scale_alpha,
            "atom_residue_attention": joint_attention,
            "drug_atom_attention": atom_attention,
            "target_residue_attention": residue_attention,
            "drug_node_batch": batch_drugs.batch,
            "target_residue_mask": residue_mask,
            "interaction_temperature": temperature,
        }