
import argparse
import csv
import json
import os
import random
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader

from Dataset import DrugDataset, FixedRatioBatchSampler, TargetDataset


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from Model import NetModel, SUBGRAPH_EDGE_TYPES, SUBGRAPH_NODE_TYPES

warnings.filterwarnings("ignore")


NODE_TYPES = ["drug", "disease", "side_effect", "target"]
TYPE_TO_INDEX = {name: idx for idx, name in enumerate(NODE_TYPES)}


def configure_runtime(torch_num_threads: Optional[int] = None) -> None:
    if torch_num_threads is not None and torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(torch_num_threads, 4)))
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_binary_matrix(path: str) -> np.ndarray:
    mat = np.loadtxt(path, dtype=np.int8)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix at {path}, got shape={mat.shape}")
    return np.ascontiguousarray(mat > 0)


def load_similarity_matrix(path: str) -> np.ndarray:
    mat = np.loadtxt(path, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square similarity matrix at {path}, got shape={mat.shape}")
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(mat.min())
    max_val = float(mat.max())
    if max_val > min_val:
        mat = (mat - min_val) / (max_val - min_val)
    else:
        mat = np.zeros_like(mat, dtype=np.float32)
    np.fill_diagonal(mat, 1.0)
    return np.clip(mat, 0.0, 1.0).astype(np.float32, copy=False)


def get_split_dir(data_dir: str, n_folds: int, setting: str) -> str:
    mapping = {
        "normal": os.path.join(data_dir, f"{n_folds}_folds"),
        "drug_cold": os.path.join(data_dir, f"{n_folds}_folds_drug_cold"),
        "protein_cold": os.path.join(data_dir, f"{n_folds}_folds_protein_cold"),
    }
    if setting not in mapping:
        raise ValueError(f"Unsupported setting: {setting}")
    return mapping[setting]


def fold_to_pairs(fold_dict: Dict[str, List[List[int]]], fold_id: int) -> np.ndarray:
    rows, cols = fold_dict[f"fold_{fold_id}"]
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)], axis=1)


def concat_pairs_excluding_fold(
    fold_dict: Dict[str, List[List[int]]], n_folds: int, holdout_fold: int
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for fid in range(n_folds):
        if fid == holdout_fold:
            continue
        part = fold_to_pairs(fold_dict, fid)
        if len(part) > 0:
            parts.append(part)
    if len(parts) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(parts, axis=0)


def load_fold_data(
    data_dir: str, n_folds: int, setting: str, neg_times: int, fold_id: int
) -> Dict[str, torch.Tensor]:
    split_dir = get_split_dir(data_dir, n_folds, setting)
    pos_folds = load_json(os.path.join(split_dir, "pos_folds.json"))
    neg_path = os.path.join(split_dir, f"neg_folds_times_{neg_times}.json")
    if not os.path.exists(neg_path):
        raise FileNotFoundError(f"Cannot find negative fold file: {neg_path}")
    neg_folds = load_json(neg_path)

    train_pos = concat_pairs_excluding_fold(pos_folds, n_folds, holdout_fold=fold_id)
    valid_pos = fold_to_pairs(pos_folds, fold_id)
    train_neg = concat_pairs_excluding_fold(neg_folds, n_folds, holdout_fold=fold_id)
    valid_neg = fold_to_pairs(neg_folds, fold_id)

    return {
        "train_pos": torch.as_tensor(train_pos, dtype=torch.long),
        "train_neg": torch.as_tensor(train_neg, dtype=torch.long),
        "valid_pos": torch.as_tensor(valid_pos, dtype=torch.long),
        "valid_neg": torch.as_tensor(valid_neg, dtype=torch.long),
    }


def target_collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    max_len = 512
    truncated_data = [temp[: max_len + 2] for temp in batch]
    return pad_sequence(truncated_data, batch_first=True, padding_value=0)


def load_entity_features(data_type: str, device: torch.device):
    drug_data = DrugDataset(data_type=data_type)
    drug_loader = PyGDataLoader(drug_data, batch_size=len(drug_data), shuffle=False)
    drugs = next(iter(drug_loader)).to(device)

    target_data = TargetDataset(data_type=data_type)
    target_loader = DataLoader(
        target_data,
        batch_size=len(target_data),
        shuffle=False,
        collate_fn=target_collate_fn,
    )
    targets = next(iter(target_loader)).to(device)
    return drugs, targets


def calculate_metrics(labels: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)
    labels_np = labels.detach().cpu().numpy()
    metrics = {
        "Accuracy": accuracy_score(labels_np, preds),
        "Precision": precision_score(labels_np, preds, zero_division=0),
        "Recall": recall_score(labels_np, preds, zero_division=0),
        "F1": f1_score(labels_np, preds, zero_division=0),
        "AUC": roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0.0,
        "AUPR": average_precision_score(labels_np, probs) if len(np.unique(labels_np)) > 1 else 0.0,
    }
    return {k: float(v) for k, v in metrics.items()}


def infer_dataset_tag(dataset_name: str) -> str:
    lower = dataset_name.lower()
    if "luo" in lower:
        return "luo"
    if "li" in lower:
        return "li"
    cleaned = "".join(ch for ch in lower if ch.isalnum() or ch in ("_", "-"))
    return cleaned or "dataset"


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class NodeSetGlobalFeatureBuilder:

    def __init__(self, data_dir: str, train_pos_pairs: torch.Tensor, use_train_dti_edges: bool = True):
        self.use_train_dti_edges = bool(use_train_dti_edges)
        self.drug_disease = load_binary_matrix(os.path.join(data_dir, "mat_drug_disease.txt"))
        self.drug_se = load_binary_matrix(os.path.join(data_dir, "mat_drug_se.txt"))
        self.target_disease = load_binary_matrix(os.path.join(data_dir, "mat_protein_disease.txt"))
        self.drug_sim = load_similarity_matrix(os.path.join(data_dir, "Similarity_Matrix_Drugs.txt"))
        self.target_sim = load_similarity_matrix(os.path.join(data_dir, "Similarity_Matrix_Proteins.txt"))

        self.num_drugs = int(self.drug_sim.shape[0])
        self.num_diseases = int(self.drug_disease.shape[1])
        self.num_side_effects = int(self.drug_se.shape[1])
        self.num_targets = int(self.target_sim.shape[0])
        self.offsets = {
            "drug": 0,
            "disease": self.num_drugs,
            "side_effect": self.num_drugs + self.num_diseases,
            "target": self.num_drugs + self.num_diseases + self.num_side_effects,
        }

        self.train_dti = np.zeros((self.num_drugs, self.num_targets), dtype=bool)
        if self.use_train_dti_edges and train_pos_pairs.numel() > 0:
            pairs_np = train_pos_pairs.cpu().numpy()
            self.train_dti[pairs_np[:, 0], pairs_np[:, 1]] = True

        self.drug_disease_deg = self.drug_disease.sum(axis=1).astype(np.float32)
        self.drug_se_deg = self.drug_se.sum(axis=1).astype(np.float32)
        self.target_disease_deg = self.target_disease.sum(axis=1).astype(np.float32)
        self.disease_drug_deg = self.drug_disease.sum(axis=0).astype(np.float32)
        self.disease_target_deg = self.target_disease.sum(axis=0).astype(np.float32)
        self.se_drug_deg = self.drug_se.sum(axis=0).astype(np.float32)
        self.drug_train_deg = self.train_dti.sum(axis=1).astype(np.float32)
        self.target_train_deg = self.train_dti.sum(axis=0).astype(np.float32)
        self.drug_sim_mean = self._mean_topk_similarity(self.drug_sim, topk=10)
        self.target_sim_mean = self._mean_topk_similarity(self.target_sim, topk=10)

        disease_total_deg = self.disease_drug_deg + self.disease_target_deg
        self.disease_specificity = self._specificity(disease_total_deg)
        self.se_specificity = self._specificity(self.se_drug_deg)

        self.max_values = {
            "drug_disease": self._safe_max(self.drug_disease_deg, self.disease_drug_deg),
            "drug_se": self._safe_max(self.drug_se_deg, self.se_drug_deg),
            "target_disease": self._safe_max(self.target_disease_deg, self.disease_target_deg),
            "train_dti": self._safe_max(self.drug_train_deg, self.target_train_deg),
        }
        self.global_relation_degree = self._build_global_relation_degree()

    @staticmethod
    def _safe_max(*arrays: np.ndarray) -> float:
        value = max(float(arr.max()) if arr.size else 0.0 for arr in arrays)
        return max(value, 1.0)

    @staticmethod
    def _degree_norm(value: float, max_value: float) -> float:
        return float(np.log1p(value) / np.log1p(max_value))

    @staticmethod
    def _specificity(degree: np.ndarray) -> np.ndarray:
        values = 1.0 / np.log1p(degree.astype(np.float32) + 1.0)
        denom = max(float(values.max()) if values.size else 0.0, 1e-8)
        return (values / denom).astype(np.float32, copy=False)

    @staticmethod
    def _mean_topk_similarity(sim: np.ndarray, topk: int) -> np.ndarray:
        n = int(sim.shape[0])
        if n <= 1 or topk <= 0:
            return np.zeros(n, dtype=np.float32)
        k = min(topk, n - 1)
        result = np.zeros(n, dtype=np.float32)
        for row_id in range(n):
            row = sim[row_id].copy()
            row[row_id] = -np.inf
            idx = np.argpartition(row, -k)[-k:]
            finite = row[idx][np.isfinite(row[idx])]
            result[row_id] = float(finite.mean()) if finite.size else 0.0
        return result

    def _build_global_relation_degree(self) -> np.ndarray:
        total_nodes = self.num_drugs + self.num_diseases + self.num_side_effects + self.num_targets
        counts = np.zeros((total_nodes, 6), dtype=np.float32)

        drug_start = self.offsets["drug"]
        disease_start = self.offsets["disease"]
        se_start = self.offsets["side_effect"]
        target_start = self.offsets["target"]

        counts[drug_start:disease_start, 0] = np.minimum((self.drug_sim > 0).sum(axis=1) - 1, 10)
        counts[drug_start:disease_start, 1] = self.drug_disease_deg
        counts[drug_start:disease_start, 2] = self.drug_se_deg
        counts[drug_start:disease_start, 5] = self.drug_train_deg

        counts[disease_start:se_start, 1] = self.disease_drug_deg
        counts[disease_start:se_start, 4] = self.disease_target_deg

        counts[se_start:target_start, 2] = self.se_drug_deg

        counts[target_start:, 3] = np.minimum((self.target_sim > 0).sum(axis=1) - 1, 10)
        counts[target_start:, 4] = self.target_disease_deg
        counts[target_start:, 5] = self.target_train_deg

        scales = np.ones((len(NODE_TYPES), 6), dtype=np.float32)
        slices = {
            "drug": slice(drug_start, disease_start),
            "disease": slice(disease_start, se_start),
            "side_effect": slice(se_start, target_start),
            "target": slice(target_start, total_nodes),
        }
        for type_name, type_slice in slices.items():
            type_idx = TYPE_TO_INDEX[type_name]
            values = counts[type_slice]
            for rel_idx in range(6):
                scales[type_idx, rel_idx] = max(float(np.percentile(values[:, rel_idx], 99.0)), 1.0)

        degree = np.zeros_like(counts, dtype=np.float32)
        for type_name, type_slice in slices.items():
            type_idx = TYPE_TO_INDEX[type_name]
            degree[type_slice] = np.log1p(counts[type_slice]) / np.log1p(scales[type_idx]).reshape(1, -1)
        return np.clip(degree, 0.0, 1.0).astype(np.float32, copy=False)
    def global_id(self, node_type: str, local_id: int) -> int:
        return int(self.offsets[node_type] + int(local_id))

    def decode_gid(self, gid: int) -> Tuple[str, int]:
        gid = int(gid)
        if gid < self.offsets["disease"]:
            return "drug", gid
        if gid < self.offsets["side_effect"]:
            return "disease", gid - self.offsets["disease"]
        if gid < self.offsets["target"]:
            return "side_effect", gid - self.offsets["side_effect"]
        return "target", gid - self.offsets["target"]

    def ensure_query_nodes(self, nodes: torch.Tensor, drug_id: int, target_id: int) -> torch.Tensor:
        nodes = torch.unique(nodes.to(torch.long), sorted=True)
        required_nodes = [
            self.global_id("drug", drug_id),
            self.global_id("target", target_id),
        ]
        node_set = set(int(value) for value in nodes.tolist())
        missing = [gid for gid in required_nodes if gid not in node_set]
        if missing:
            raise ValueError(
                "Subgraph node-set missing query anchor nodes; "
                f"drug_id={drug_id}, target_id={target_id}, missing_global_ids={missing}"
            )
        return nodes

    def _empty_edge_index(self) -> torch.Tensor:
        return torch.empty((2, 0), dtype=torch.long)

    def _sim_edges(self, local_ids: List[int], sim: np.ndarray, topk: int = 5) -> torch.Tensor:
        if len(local_ids) <= 1:
            return self._empty_edge_index()
        id_to_pos = {local_id: pos for pos, local_id in enumerate(local_ids)}
        local_array = np.asarray(local_ids, dtype=np.int64)
        src_list: List[int] = []
        dst_list: List[int] = []
        for src_local in local_ids:
            candidates = local_array[local_array != src_local]
            if candidates.size == 0:
                continue
            scores = sim[src_local, candidates]
            keep = np.flatnonzero(scores > 0)
            if keep.size == 0:
                continue
            keep = keep[np.argsort(scores[keep])[::-1]][:topk]
            for pos in keep.tolist():
                dst_local = int(candidates[pos])
                src_list.append(id_to_pos[src_local])
                dst_list.append(id_to_pos[dst_local])
        if not src_list:
            return self._empty_edge_index()
        return torch.tensor([src_list, dst_list], dtype=torch.long)

    def _binary_edges(
        self,
        src_ids: List[int],
        dst_ids: List[int],
        matrix: np.ndarray,
        exclude_pair: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if not src_ids or not dst_ids:
            return self._empty_edge_index()
        src_list: List[int] = []
        dst_list: List[int] = []
        for src_pos, src_id in enumerate(src_ids):
            row = matrix[src_id]
            for dst_pos, dst_id in enumerate(dst_ids):
                if exclude_pair is not None and src_id == exclude_pair[0] and dst_id == exclude_pair[1]:
                    continue
                if row[dst_id]:
                    src_list.append(src_pos)
                    dst_list.append(dst_pos)
        if not src_list:
            return self._empty_edge_index()
        return torch.tensor([src_list, dst_list], dtype=torch.long)

    def _metapath_role(self, node_type: str, local_id: int, drug_id: int, target_id: int) -> np.ndarray:
        role = np.zeros(8, dtype=np.float32)
        if node_type == "drug":
            role[1] = float(local_id != drug_id and self.drug_sim[local_id, drug_id] > 0)
            role[3] = float(local_id != drug_id and self.train_dti[local_id, target_id])
            role[6] = role[1]
            if local_id == drug_id:
                role[0] = 1.0
        elif node_type == "target":
            role[2] = float(local_id != target_id and self.target_sim[local_id, target_id] > 0)
            role[5] = float(local_id != target_id and self.train_dti[drug_id, local_id])
            role[7] = role[2]
            if local_id == target_id:
                role[0] = 1.0
        elif node_type == "disease":
            has_drug = bool(self.drug_disease[drug_id, local_id])
            has_target = bool(self.target_disease[target_id, local_id])
            role[0] = float(has_drug and has_target)
            role[3] = float(has_drug)
            role[5] = float(has_target)
        elif node_type == "side_effect":
            role[4] = float(self.drug_se[drug_id, local_id])
        return role

    def _node_store(self, data: HeteroData, node_type: str, gids: List[int], drug_id: int, target_id: int) -> List[int]:
        local_ids = [self.decode_gid(gid)[1] for gid in gids]
        store = data[node_type]
        store.num_nodes = len(gids)
        store.node_id = torch.tensor(gids, dtype=torch.long)
        store.x = torch.zeros((len(gids), 1), dtype=torch.float32)
        store.is_source = torch.tensor(
            [node_type == "drug" and local_id == drug_id for local_id in local_ids],
            dtype=torch.bool,
        )
        store.is_target = torch.tensor(
            [node_type == "target" and local_id == target_id for local_id in local_ids],
            dtype=torch.bool,
        )
        store.min_depth = torch.zeros(len(gids), dtype=torch.long)
        store.max_depth = torch.ones(len(gids), dtype=torch.long)
        store.dist_to_target = torch.zeros(len(gids), dtype=torch.long)
        store.metapath_role = torch.tensor(
            [self._metapath_role(node_type, local_id, drug_id, target_id) for local_id in local_ids],
            dtype=torch.float32,
        )

        relation_degree = self.global_relation_degree[gids].copy() if gids else np.empty((0, 6), dtype=np.float32)
        if bool(self.train_dti[drug_id, target_id]) and relation_degree.size:
            for row_idx, local_id in enumerate(local_ids):
                if (node_type == "drug" and local_id == drug_id) or (node_type == "target" and local_id == target_id):
                    raw_degree = self.drug_train_deg[drug_id] if node_type == "drug" else self.target_train_deg[target_id]
                    relation_degree[row_idx, 5] = self._degree_norm(max(float(raw_degree) - 1.0, 0.0), self.max_values["train_dti"])
        store.global_relation_degree = torch.tensor(relation_degree, dtype=torch.float32)
        return local_ids

    def _assign_shortest_path_positions(self, data: HeteroData, type_gids: Dict[str, List[int]], drug_id: int, target_id: int) -> None:
        gid_to_node: Dict[int, Tuple[str, int]] = {}
        adjacency: Dict[int, List[int]] = {}
        for node_type, gids in type_gids.items():
            for pos, gid_value in enumerate(gids):
                gid_int = int(gid_value)
                gid_to_node[gid_int] = (node_type, pos)
                adjacency[gid_int] = []

        for edge_type in SUBGRAPH_EDGE_TYPES:
            src_type, _, dst_type = edge_type
            edge_index = data[edge_type].edge_index
            if edge_index.numel() == 0:
                continue
            src_gids = type_gids[src_type]
            dst_gids = type_gids[dst_type]
            for src_pos, dst_pos in edge_index.t().tolist():
                src_gid = int(src_gids[src_pos])
                dst_gid = int(dst_gids[dst_pos])
                adjacency[src_gid].append(dst_gid)
                adjacency[dst_gid].append(src_gid)

        def bfs(start_gid: int) -> Dict[int, int]:
            distances = {start_gid: 0}
            queue = deque([start_gid])
            while queue:
                cur = queue.popleft()
                for nxt in adjacency.get(cur, []):
                    if nxt in distances:
                        continue
                    distances[nxt] = distances[cur] + 1
                    queue.append(nxt)
            return distances

        drug_gid = self.global_id("drug", drug_id)
        target_gid = self.global_id("target", target_id)
        drug_dist = bfs(drug_gid)
        target_dist = bfs(target_gid)
        missing_distance = 4
        max_distance = missing_distance
        for dist_map in (drug_dist, target_dist):
            if dist_map:
                max_distance = max(max_distance, max(dist_map.values()))
        max_distance = max(1, min(int(max_distance), missing_distance))
        data.max_hops = torch.tensor([max_distance], dtype=torch.long)

        for node_type, gids in type_gids.items():
            min_depth = []
            max_depth = []
            dist_to_target = []
            for gid_value in gids:
                gid_int = int(gid_value)
                d_drug = min(int(drug_dist.get(gid_int, missing_distance)), missing_distance)
                d_target = min(int(target_dist.get(gid_int, missing_distance)), missing_distance)
                min_depth.append(min(d_drug, d_target))
                max_depth.append(max(d_drug, d_target))
                dist_to_target.append(d_target)
            data[node_type].min_depth = torch.tensor(min_depth, dtype=torch.long)
            data[node_type].max_depth = torch.tensor(max_depth, dtype=torch.long)
            data[node_type].dist_to_target = torch.tensor(dist_to_target, dtype=torch.long)

    def build_subgraph(self, nodes: torch.Tensor, drug_id: int, target_id: int) -> HeteroData:
        nodes = self.ensure_query_nodes(nodes, drug_id, target_id)
        type_gids: Dict[str, List[int]] = {node_type: [] for node_type in NODE_TYPES}
        for gid_value in nodes.tolist():
            node_type, _ = self.decode_gid(int(gid_value))
            type_gids[node_type].append(int(gid_value))

        data = HeteroData()
        data.max_hops = torch.tensor([4], dtype=torch.long)
        local_ids = {
            node_type: self._node_store(data, node_type, type_gids[node_type], drug_id, target_id)
            for node_type in NODE_TYPES
        }

        dd = self._sim_edges(local_ids["drug"], self.drug_sim)
        td = self._binary_edges(local_ids["drug"], local_ids["disease"], self.drug_disease)
        ds = self._binary_edges(local_ids["drug"], local_ids["side_effect"], self.drug_se)
        tt = self._sim_edges(local_ids["target"], self.target_sim)
        pt = self._binary_edges(local_ids["target"], local_ids["disease"], self.target_disease)
        dti = self._binary_edges(
            local_ids["drug"],
            local_ids["target"],
            self.train_dti,
            exclude_pair=(drug_id, target_id),
        )

        data[("drug", "drug_interacts", "drug")].edge_index = dd
        data[("drug", "treats", "disease")].edge_index = td
        data[("disease", "rev_treats", "drug")].edge_index = td.flip(0)
        data[("drug", "causes_se", "side_effect")].edge_index = ds
        data[("side_effect", "rev_causes_se", "drug")].edge_index = ds.flip(0)
        data[("target", "target_interacts", "target")].edge_index = tt
        data[("target", "assoc_disease", "disease")].edge_index = pt
        data[("disease", "rev_assoc_disease", "target")].edge_index = pt.flip(0)
        data[("drug", "interacts", "target")].edge_index = dti
        data[("target", "rev_interacts", "drug")].edge_index = dti.flip(0)

        for edge_type in SUBGRAPH_EDGE_TYPES:
            if "edge_index" not in data[edge_type]:
                data[edge_type].edge_index = self._empty_edge_index()
        self._assign_shortest_path_positions(data, type_gids, drug_id, target_id)
        return data


def load_node_set_manifest(manifest_path: str, subgraph_dir: str) -> Dict[Tuple[int, int, int], Dict[str, str]]:
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mapping: Dict[Tuple[int, int, int], Dict[str, str]] = {}
    shard_dir = os.path.join(subgraph_dir, "shards")
    for row in rows:
        key = (int(row["drug_id"]), int(row["target_id"]), int(row["label"]))
        mapping[key] = {
            "sample_key": row["sample_key"],
            "shard_path": os.path.join(shard_dir, row["shard_path"]),
        }
    return mapping


class NodeSetNodeSetDataset(Dataset):
    def __init__(
        self,
        pairs: torch.Tensor,
        labels: torch.Tensor,
        graphs: List[HeteroData],
    ):
        if len(pairs) != len(labels) or len(pairs) != len(graphs):
            raise ValueError(
                f"Mismatched dataset lengths: pairs={len(pairs)}, "
                f"labels={len(labels)}, graphs={len(graphs)}"
            )
        self.pairs = pairs
        self.labels = labels
        self.graphs = graphs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx], self.graphs[idx]


class NodeSetNodeSetCollator:
    def __call__(self, batch):
        batch_pairs = torch.stack([item[0] for item in batch], dim=0)
        batch_labels = torch.stack([item[1] for item in batch], dim=0)
        graph_list = [item[2] for item in batch]
        return batch_pairs, batch_labels, Batch.from_data_list(graph_list)


def prebuild_node_set_subgraphs(
    pairs: torch.Tensor,
    sample_refs: List[Dict[str, str]],
    feature_builder: NodeSetGlobalFeatureBuilder,
    cache_shards: bool,
    split_name: str,
) -> List[HeteroData]:
    total = len(sample_refs)
    shard_cache: Dict[str, Dict[str, Any]] = {}
    graphs: List[HeteroData] = []

    for idx, (pair, sample_ref) in enumerate(zip(pairs.tolist(), sample_refs), start=1):
        shard_path = sample_ref["shard_path"]
        if cache_shards and shard_path in shard_cache:
            shard = shard_cache[shard_path]
        else:
            shard = safe_torch_load(shard_path)
            if cache_shards:
                shard_cache[shard_path] = shard

        sample = shard[sample_ref["sample_key"]]
        graphs.append(
            feature_builder.build_subgraph(
                sample["nodes"].to(torch.long),
                drug_id=int(pair[0]),
                target_id=int(pair[1]),
            )
        )

    shard_cache.clear()
    return graphs

@dataclass
class TrainConfig:
    data_dir: str
    data_type: str
    dataset_tag: str
    subgraph_dir: str
    setting: str
    n_folds: int = 5
    neg_times: int = 1
    seed: int = 42
    lr: float = 2e-4
    weight_decay: float = 3e-7
    batch_size: int = 64
    epochs: int = 800
    patience: int = 40
    num_workers: int = 0
    device: str = "cuda:0"
    hidden_dim: int = 128
    dropout: float = 0.2
    torch_num_threads: Optional[int] = 2
    pair_aux_weight: float = 0.0
    subgraph_aux_weight: float = 0.2
    cache_shards: bool = True


class FoldTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = self.resolve_device(cfg.device)
        self.drugs, self.targets = load_entity_features(cfg.data_type, self.device)

    @staticmethod
    def resolve_device(device_name: str = "cuda:0") -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def build_model(self):
        return NetModel(
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)

    def make_loader_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "num_workers": self.cfg.num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if self.cfg.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs

    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, criterion: nn.Module):
        main_loss = criterion(outputs["main_logits"], labels)
        pair_loss = criterion(outputs["pair_logits"], labels)
        subgraph_loss = criterion(outputs["subgraph_logits"], labels)
        total_loss = (
            main_loss
            + self.cfg.pair_aux_weight * pair_loss
            + self.cfg.subgraph_aux_weight * subgraph_loss
        )
        return total_loss, main_loss, pair_loss, subgraph_loss

    def build_loader(
        self,
        pairs: torch.Tensor,
        labels: torch.Tensor,
        graphs: List[HeteroData],
        train: bool,
    ) -> DataLoader:
        dataset = NodeSetNodeSetDataset(pairs, labels, graphs)
        collator = NodeSetNodeSetCollator()
        loader_kwargs = self.make_loader_kwargs()

        if train:
            return DataLoader(
                dataset,
                batch_sampler=FixedRatioBatchSampler(labels=labels, batch_size=self.cfg.batch_size),
                collate_fn=collator,
                **loader_kwargs,
            )

        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
            drop_last=False,
            **loader_kwargs,
        )

    def run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        is_train = optimizer is not None
        model.train(is_train)

        all_logits = []
        all_labels = []
        all_scale_weights = []
        all_scale_alpha = []
        all_residual_gate = []
        all_residual_alpha = []
        total_loss = 0.0
        total_main_loss = 0.0
        total_pair_loss = 0.0
        total_subgraph_loss = 0.0
        total_steps = 0

        for batch_pairs, batch_labels, batch_subgraphs in loader:
            non_blocking = self.device.type == "cuda"
            batch_pairs = batch_pairs.to(self.device, non_blocking=non_blocking)
            batch_labels = batch_labels.to(self.device, non_blocking=non_blocking)
            batch_subgraphs = batch_subgraphs.to(self.device, non_blocking=non_blocking)

            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                outputs = model(
                    self.drugs,
                    self.targets,
                    batch_pairs,
                    batch_subgraphs,
                )
                loss, main_loss, pair_loss, subgraph_loss = self.compute_loss(
                    outputs, batch_labels, criterion
                )
                if is_train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_pair_loss += pair_loss.item()
            total_subgraph_loss += subgraph_loss.item()
            total_steps += 1
            all_logits.append(outputs["main_logits"].detach().float())
            all_labels.append(batch_labels.detach())
            all_scale_weights.append(outputs["target_scale_weights"].detach().float())
            all_scale_alpha.append(outputs["target_scale_alpha"].detach().float().view(1))
            all_residual_gate.append(outputs["residual_gate"].detach().float())
            all_residual_alpha.append(outputs["residual_alpha"].detach().float().view(1))

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = calculate_metrics(labels, logits)
        metrics["MainLoss"] = total_main_loss / max(total_steps, 1)
        metrics["PairLoss"] = total_pair_loss / max(total_steps, 1)
        metrics["SubgraphLoss"] = total_subgraph_loss / max(total_steps, 1)
        scale_weights = torch.cat(all_scale_weights, dim=0)
        metrics["ScaleK3"] = float(scale_weights[:, 0].mean())
        metrics["ScaleK5"] = float(scale_weights[:, 1].mean())
        metrics["ScaleK9"] = float(scale_weights[:, 2].mean())
        metrics["ScaleEntropy"] = float(
            -(scale_weights * scale_weights.clamp_min(1e-8).log()).sum(dim=1).mean()
        )
        metrics["ScaleMaxWeight"] = float(scale_weights.max(dim=1).values.mean())
        metrics["ScaleResidualAlpha"] = float(torch.cat(all_scale_alpha).mean())
        residual_gate = torch.cat(all_residual_gate, dim=0)
        residual_alpha = torch.cat(all_residual_alpha).mean()
        metrics["ResidualGateMean"] = float(residual_gate.mean())
        metrics["ResidualGateStd"] = float(residual_gate.std(unbiased=False))
        metrics["ResidualAlpha"] = float(residual_alpha)
        metrics["ResidualEffectiveMean"] = float(residual_gate.mean() * residual_alpha)
        return total_loss / max(total_steps, 1), metrics

    def resolve_sample_refs(
        self,
        pairs: torch.Tensor,
        labels: torch.Tensor,
        manifest_map: Dict[Tuple[int, int, int], Dict[str, str]],
    ) -> List[Dict[str, str]]:
        refs = []
        missing = []
        for pair, label in zip(pairs.tolist(), labels.tolist()):
            key = (int(pair[0]), int(pair[1]), int(label))
            ref = manifest_map.get(key)
            if ref is None:
                missing.append(key)
            else:
                refs.append(ref)
        if missing:
            raise KeyError(f"Cannot find {len(missing)} subgraph node-set samples; first={missing[0]}")
        return refs

    def use_train_dti_edges(self) -> bool:
        return self.cfg.setting == "normal"

    def checkpoint_path(self, fold_id: int) -> str:
        filename = (
            f"{self.cfg.dataset_tag}_{self.cfg.setting}_1to{self.cfg.neg_times}_"
            f"fold{fold_id + 1}_best.pt"
        )
        return os.path.join(CURRENT_DIR, "weights", filename)

    def save_checkpoint(self, path: str, model: nn.Module, fold_id: int) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save(state_dict, path)

    def train_one_fold(self, fold_id: int) -> Dict[str, float]:
        fold_data = load_fold_data(
            data_dir=self.cfg.data_dir,
            n_folds=self.cfg.n_folds,
            setting=self.cfg.setting,
            neg_times=self.cfg.neg_times,
            fold_id=fold_id,
        )
        train_pairs = torch.cat([fold_data["train_pos"], fold_data["train_neg"]], dim=0)
        train_labels = torch.cat(
            [
                torch.ones(len(fold_data["train_pos"]), dtype=torch.long),
                torch.zeros(len(fold_data["train_neg"]), dtype=torch.long),
            ],
            dim=0,
        )
        valid_pairs = torch.cat([fold_data["valid_pos"], fold_data["valid_neg"]], dim=0)
        valid_labels = torch.cat(
            [
                torch.ones(len(fold_data["valid_pos"]), dtype=torch.long),
                torch.zeros(len(fold_data["valid_neg"]), dtype=torch.long),
            ],
            dim=0,
        )
        manifest_path = os.path.join(self.cfg.subgraph_dir, f"fold_{fold_id}_pairs.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing subgraph manifest: {manifest_path}")
        manifest_map = load_node_set_manifest(manifest_path, self.cfg.subgraph_dir)
        train_refs = self.resolve_sample_refs(train_pairs, train_labels, manifest_map)
        valid_refs = self.resolve_sample_refs(valid_pairs, valid_labels, manifest_map)

        use_train_dti_edges = self.use_train_dti_edges()
        feature_builder = NodeSetGlobalFeatureBuilder(
            self.cfg.data_dir,
            fold_data["train_pos"],
            use_train_dti_edges=use_train_dti_edges,
        )
        train_graphs = prebuild_node_set_subgraphs(
            train_pairs,
            train_refs,
            feature_builder,
            cache_shards=self.cfg.cache_shards,
            split_name=f"fold{fold_id}_train",
        )
        valid_graphs = prebuild_node_set_subgraphs(
            valid_pairs,
            valid_refs,
            feature_builder,
            cache_shards=self.cfg.cache_shards,
            split_name=f"fold{fold_id}_valid",
        )
        train_loader = self.build_loader(train_pairs, train_labels, train_graphs, train=True)
        valid_loader = self.build_loader(valid_pairs, valid_labels, valid_graphs, train=False)

        model = self.build_model()

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        best_metrics: Optional[Dict[str, float]] = None
        best_score = -1.0
        best_epoch = 0
        patience_count = 0
        checkpoint_path = self.checkpoint_path(fold_id)

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, _ = self.run_epoch(model, train_loader, criterion, optimizer)
            valid_loss, valid_metrics = self.run_epoch(model, valid_loader, criterion, None)
            monitor_score = valid_metrics["AUC"]
            print(
                f"[Fold {fold_id + 1}] Epoch {epoch:03d} | "
                f"Train Loss={train_loss:.5f} | Valid Loss={valid_loss:.5f}",
                flush=True,
            )
            if monitor_score > best_score:
                best_score = monitor_score
                best_metrics = valid_metrics
                best_epoch = epoch
                patience_count = 0
                self.save_checkpoint(checkpoint_path, model, fold_id)
            else:
                patience_count += 1
            if patience_count >= self.cfg.patience:
                print(f"[Fold {fold_id + 1}] Early stopping at epoch {epoch}.", flush=True)
                break

        if best_metrics is None:
            best_metrics = {
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "AUC": 0.0,
                "AUPR": 0.0,
                "MainLoss": 0.0,
                "PairLoss": 0.0,
                "SubgraphLoss": 0.0,
                "ResidualGateMean": 0.0,
                "ResidualGateStd": 0.0,
                "ResidualAlpha": 0.0,
                "ResidualEffectiveMean": 0.0,
            }
        return best_metrics

    def run(self):
        metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC", "AUPR"]
        fold_results = []
        for fold_id in range(self.cfg.n_folds):
            metrics = self.train_one_fold(fold_id)
            fold_results.append(metrics)
            values = " ".join(f"{name}={metrics[name]:.4f}" for name in metric_names)
            print(f"[Fold {fold_id + 1} Result] {values}", flush=True)

        averages = {
            name: float(np.mean([metrics[name] for metrics in fold_results]))
            for name in metric_names
        }
        summary = " ".join(f"{name}={averages[name]:.4f}" for name in metric_names)
        print(f"[Overall Average] {summary}", flush=True)
        return fold_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the drug-target interaction model")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="luo's dataset",
        choices=["li's dataset", "luo's dataset"],
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="normal",
        choices=["normal", "drug_cold", "protein_cold"],
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default="1:1",
        choices=["1:1", "1:3", "1:5"],
        help="positive:negative sampling ratio",
    )
    args = parser.parse_args()
    if args.setting != "normal" and args.ratio != "1:1":
        parser.error("cold-start settings currently support only --ratio 1:1")
    return args


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    data_dir = os.path.join(PROJECT_ROOT, "data", args.dataset_name)
    dataset_tag = infer_dataset_tag(args.dataset_name)
    neg_times = int(args.ratio.split(":", 1)[1])
    subgraph_dir = os.path.join(
        PROJECT_ROOT,
        "subgraph_extration",
        "outputs",
        dataset_tag,
        f"nodes_{args.setting}_1to{neg_times}",
    )
    return data_dir, args.dataset_name, dataset_tag, subgraph_dir


if __name__ == "__main__":
    args = parse_args()
    configure_runtime(torch_num_threads=2)
    set_seed(42)
    data_dir, data_type, dataset_tag, subgraph_dir = resolve_paths(args)
    cfg = TrainConfig(
        data_dir=data_dir,
        data_type=data_type,
        dataset_tag=dataset_tag,
        subgraph_dir=subgraph_dir,
        setting=args.setting,
        neg_times=int(args.ratio.split(":", 1)[1]),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    print(
        f"[Run] dataset={dataset_tag} setting={args.setting} ratio={args.ratio}",
        flush=True,
    )
    print(f"[Run] data_dir={data_dir}", flush=True)
    print(f"[Run] subgraph_dir={subgraph_dir}", flush=True)
    print(
        f"[Config] device={cfg.device} epochs={cfg.epochs} batch_size={cfg.batch_size} "
        f"patience={cfg.patience} lr={cfg.lr:.8g}",
        flush=True,
    )
    FoldTrainer(cfg).run()








