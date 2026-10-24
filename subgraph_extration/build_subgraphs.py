import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch


# Subgraph node-set builder
# --------------------
# This builder creates global node-id sets for each drug-target pair.

PairArray = np.ndarray
NODE_TYPES = ["drug", "disease", "side_effect", "target"]

METAPATHS = [
    "N1_drug_disease_target",
    "N2_drug_simdrug_disease_target",
    "N3_drug_disease_simtarget_target",
    "N4_drug_simdrug_disease_simtarget_target",
    "N5_drug_side_effect_drug_disease_target",
    "N6_drug_ddi_drug_disease_target",
    "N7_drug_disease_ppi_target_target",
    "N8_drug_side_effect_drug_train_dti_target",
    "N9_drug_disease_drug_train_dti_target",
    "N10_drug_train_dti_target_disease_target",
    "N11_drug_simdrug_disease_drug_train_dti_target",
    "N12_drug_train_dti_target_disease_simtarget_target",
    "N13_drug_simdrug_side_effect_drug_train_dti_target",
    "N14_drug_simdrug_train_dti_target",
    "N15_drug_train_dti_simtarget_target",
    "N16_drug_ddi_drug_train_dti_target",
    "N17_drug_train_dti_ppi_target_target",
]
METAPATH_TO_CODE = {name: idx for idx, name in enumerate(METAPATHS)}


@dataclass(frozen=True)
class PathRecord:
    template_idx: int
    phenotype_type: str
    phenotype_gid: int
    node_gids: Tuple[int, ...]


@dataclass
class SubgraphIndex:
    type_offsets: Dict[str, int]
    type_sizes: Dict[str, int]
    disease_specificity: np.ndarray
    side_effect_specificity: np.ndarray
    drug_sim_neighbors: List[np.ndarray]
    target_sim_neighbors: List[np.ndarray]
    drug_ddi_neighbors: List[np.ndarray]
    target_ppi_neighbors: List[np.ndarray]
    drug_diseases: List[np.ndarray]
    disease_drugs: List[np.ndarray]
    drug_side_effects: List[np.ndarray]
    side_effect_drugs: List[np.ndarray]
    target_diseases: List[np.ndarray]
    disease_targets: List[np.ndarray]
    drug_train_targets: List[np.ndarray]
    target_train_drugs: List[np.ndarray]
    use_dti_metapaths: bool


_WORKER_INDEX: Optional[SubgraphIndex] = None
_WORKER_DISEASE_TOPK = 20
_WORKER_SIDE_EFFECT_TOPK = 10
_WORKER_METAPATH_BONUS = 0.1
_WORKER_FOLD_ID = -1
_WORKER_TOTAL = 0


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def remove_file_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def load_binary_matrix(path: str) -> np.ndarray:
    mat = np.loadtxt(path, dtype=np.int32)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix at {path}, got shape={mat.shape}")
    return np.ascontiguousarray((mat > 0).astype(np.int8, copy=False))


def load_numeric_matrix_python(path: str, dtype=np.float32) -> np.ndarray:
    """Read a numeric text matrix without np.loadtxt.

    Some server builds can segfault inside np.loadtxt on Li's larger similarity
    files. A simple two-pass Python parser is slower, but it keeps the process
    in Python space and avoids that native parser crash.
    """
    n_rows = 0
    n_cols = None
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if n_cols is None:
                n_cols = len(parts)
                if n_cols == 0:
                    raise ValueError(f"No numeric columns found in {path}")
            elif len(parts) != n_cols:
                raise ValueError(
                    f"Inconsistent column count in {path}: line={line_no}, "
                    f"expected={n_cols}, got={len(parts)}"
                )
            n_rows += 1

    if n_cols is None or n_rows == 0:
        raise ValueError(f"Empty numeric matrix: {path}")

    mat = np.empty((n_rows, n_cols), dtype=dtype)
    row_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            values = [float(x) for x in line.replace(",", " ").split()]
            mat[row_idx, :] = values
            row_idx += 1
    return mat


def load_similarity_matrix(path: str) -> np.ndarray:
    mat = load_numeric_matrix_python(path, dtype=np.float32)
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


def matrix_rows_to_lists(mat: np.ndarray) -> List[np.ndarray]:
    return [np.flatnonzero(mat[row] > 0).astype(np.int64, copy=False) for row in range(mat.shape[0])]


def matrix_cols_to_lists(mat: np.ndarray) -> List[np.ndarray]:
    return [np.flatnonzero(mat[:, col] > 0).astype(np.int64, copy=False) for col in range(mat.shape[1])]


def build_topk_lists(sim: np.ndarray, topk: int) -> List[np.ndarray]:
    n = int(sim.shape[0])
    if topk <= 0 or n <= 1:
        return [np.empty((0,), dtype=np.int64) for _ in range(n)]
    k = min(int(topk), n - 1)
    result: List[np.ndarray] = []
    for src in range(n):
        row = sim[src].copy()
        row[src] = -np.inf
        cand = np.argpartition(row, -k)[-k:]
        cand = cand[np.argsort(row[cand])[::-1]]
        cand = cand[np.isfinite(row[cand])]
        result.append(cand.astype(np.int64, copy=False))
    return result


def build_topk_lists_from_similarity_file(path: str, topk: int) -> List[np.ndarray]:

    if topk <= 0:
        n_rows = 0
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                if raw_line.strip():
                    n_rows += 1
        return [np.empty((0,), dtype=np.int64) for _ in range(n_rows)]

    result: List[np.ndarray] = []
    expected_cols: Optional[int] = None
    with open(path, "r", encoding="utf-8") as f:
        for row_idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if expected_cols is None:
                expected_cols = len(parts)
            elif len(parts) != expected_cols:
                raise ValueError(
                    f"Inconsistent column count in {path}: row={row_idx}, "
                    f"expected={expected_cols}, got={len(parts)}"
                )

            scored = []
            for col_idx, value_text in enumerate(parts):
                if col_idx == row_idx:
                    continue
                try:
                    score = float(value_text)
                except ValueError:
                    score = 0.0
                if np.isfinite(score) and score > 0.0:
                    scored.append((score, col_idx))
            scored.sort(key=lambda item: item[0], reverse=True)
            result.append(np.asarray([idx for _, idx in scored[: int(topk)]], dtype=np.int64))

    if expected_cols is None:
        raise ValueError(f"Empty similarity matrix: {path}")
    if len(result) != expected_cols:
        raise ValueError(f"Expected square similarity matrix at {path}, got shape=({len(result)}, {expected_cols})")
    return result


def specificity_from_degree(degree: np.ndarray) -> np.ndarray:
    values = 1.0 / np.log1p(degree.astype(np.float32) + 1.0)
    return (values / max(float(values.max()) if values.size else 0.0, 1e-8)).astype(np.float32, copy=False)


def get_split_dir(data_dir: str, n_folds: int, setting: str) -> str:
    mapping = {
        "normal": os.path.join(data_dir, f"{n_folds}_folds"),
        "drug_cold": os.path.join(data_dir, f"{n_folds}_folds_drug_cold"),
        "protein_cold": os.path.join(data_dir, f"{n_folds}_folds_protein_cold"),
    }
    if setting not in mapping:
        raise ValueError(f"Unsupported setting: {setting}")
    return mapping[setting]


def fold_to_pairs(fold_dict: Dict[str, List[List[int]]], fold_id: int) -> PairArray:
    rows, cols = fold_dict[f"fold_{fold_id}"]
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)], axis=1)


def concat_pairs_excluding_fold(fold_dict: Dict[str, List[List[int]]], n_folds: int, holdout_fold: int) -> PairArray:
    parts: List[PairArray] = []
    for fid in range(n_folds):
        if fid == holdout_fold:
            continue
        part = fold_to_pairs(fold_dict, fid)
        if len(part) > 0:
            parts.append(part)
    if not parts:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(parts, axis=0)


def load_fold_pairs(data_dir: str, n_folds: int, setting: str, neg_times: int, fold_id: int) -> Dict[str, PairArray]:
    split_dir = get_split_dir(data_dir, n_folds, setting)
    pos_folds = load_json(os.path.join(split_dir, "pos_folds.json"))
    neg_folds = load_json(os.path.join(split_dir, f"neg_folds_times_{neg_times}.json"))
    return {
        "train_pos": concat_pairs_excluding_fold(pos_folds, n_folds, holdout_fold=fold_id),
        "valid_pos": fold_to_pairs(pos_folds, fold_id),
        "train_neg": concat_pairs_excluding_fold(neg_folds, n_folds, holdout_fold=fold_id),
        "valid_neg": fold_to_pairs(neg_folds, fold_id),
    }


def iter_pairs_with_meta(
    fold_pairs: Dict[str, PairArray], limit_per_split: Optional[int]
) -> Iterable[Tuple[str, int, int, int]]:
    for split_name in ["train_pos", "valid_pos", "train_neg", "valid_neg"]:
        pairs = fold_pairs[split_name]
        if limit_per_split is not None:
            pairs = pairs[:limit_per_split]
        label = 1 if "pos" in split_name else 0
        for drug_id, target_id in pairs.tolist():
            yield split_name, int(drug_id), int(target_id), label


def build_index(
    data_dir: str,
    train_pos_pairs: PairArray,
    drug_sim_topk: int,
    target_sim_topk: int,
    use_dti_metapaths: bool = False,
) -> SubgraphIndex:
    t0 = time.time()
    print(f"  build_index: loading matrices from {data_dir}", flush=True)

    def load_logged(label: str, filename: str, loader):
        path = os.path.join(data_dir, filename)
        size_mb = os.path.getsize(path) / (1024.0 * 1024.0) if os.path.exists(path) else -1.0
        print(f"  build_index: loading {label} from {filename} ({size_mb:.1f} MB)", flush=True)
        t_file = time.time()
        value = loader(path)
        print(f"  build_index: loaded {label} shape={value.shape} time={time.time() - t_file:.1f}s", flush=True)
        return value

    drug_disease = load_logged("drug_disease", "mat_drug_disease.txt", load_binary_matrix)
    drug_side_effect = load_logged("drug_side_effect", "mat_drug_se.txt", load_binary_matrix)
    target_disease = load_logged("target_disease", "mat_protein_disease.txt", load_binary_matrix)
    drug_ddi = load_logged("drug_ddi", "mat_drug_drug.txt", load_binary_matrix)
    target_ppi = load_logged("target_ppi", "mat_protein_protein.txt", load_binary_matrix)

    drug_sim_path = os.path.join(data_dir, "Similarity_Matrix_Drugs.txt")
    target_sim_path = os.path.join(data_dir, "Similarity_Matrix_Proteins.txt")
    print(f"  build_index: loading drug_similarity_topk from Similarity_Matrix_Drugs.txt ({os.path.getsize(drug_sim_path) / (1024.0 * 1024.0):.1f} MB)", flush=True)
    t_file = time.time()
    drug_sim_neighbors = build_topk_lists_from_similarity_file(drug_sim_path, drug_sim_topk)
    print(f"  build_index: loaded drug_similarity_topk rows={len(drug_sim_neighbors)} time={time.time() - t_file:.1f}s", flush=True)
    print(f"  build_index: loading target_similarity_topk from Similarity_Matrix_Proteins.txt ({os.path.getsize(target_sim_path) / (1024.0 * 1024.0):.1f} MB)", flush=True)
    t_file = time.time()
    target_sim_neighbors = build_topk_lists_from_similarity_file(target_sim_path, target_sim_topk)
    print(f"  build_index: loaded target_similarity_topk rows={len(target_sim_neighbors)} time={time.time() - t_file:.1f}s", flush=True)

    type_sizes = {
        "drug": int(drug_disease.shape[0]),
        "disease": int(drug_disease.shape[1]),
        "side_effect": int(drug_side_effect.shape[1]),
        "target": int(target_disease.shape[0]),
    }
    type_offsets: Dict[str, int] = {}
    total = 0
    for type_name in NODE_TYPES:
        type_offsets[type_name] = total
        total += type_sizes[type_name]

    train_mat = np.zeros((type_sizes["drug"], type_sizes["target"]), dtype=np.int8)
    if use_dti_metapaths and train_pos_pairs.size > 0:
        train_mat[train_pos_pairs[:, 0], train_pos_pairs[:, 1]] = 1

    disease_degree = drug_disease.sum(axis=0).astype(np.float32) + target_disease.sum(axis=0).astype(np.float32)
    side_effect_degree = drug_side_effect.sum(axis=0).astype(np.float32)
    index = SubgraphIndex(
        type_offsets=type_offsets,
        type_sizes=type_sizes,
        disease_specificity=specificity_from_degree(disease_degree),
        side_effect_specificity=specificity_from_degree(side_effect_degree),
        drug_sim_neighbors=drug_sim_neighbors,
        target_sim_neighbors=target_sim_neighbors,
        drug_ddi_neighbors=matrix_rows_to_lists(drug_ddi),
        target_ppi_neighbors=matrix_rows_to_lists(target_ppi),
        drug_diseases=matrix_rows_to_lists(drug_disease),
        disease_drugs=matrix_cols_to_lists(drug_disease),
        drug_side_effects=matrix_rows_to_lists(drug_side_effect),
        side_effect_drugs=matrix_cols_to_lists(drug_side_effect),
        target_diseases=matrix_rows_to_lists(target_disease),
        disease_targets=matrix_cols_to_lists(target_disease),
        drug_train_targets=matrix_rows_to_lists(train_mat),
        target_train_drugs=matrix_cols_to_lists(train_mat),
        use_dti_metapaths=bool(use_dti_metapaths),
    )
    print(
        f"  build_index: use_dti_metapaths={bool(use_dti_metapaths)} "
        f"time={time.time() - t0:.1f}s",
        flush=True,
    )
    return index


def gid(index: SubgraphIndex, type_name: str, local_id: int) -> int:
    return int(index.type_offsets[type_name] + int(local_id))


def local_id(index: SubgraphIndex, node_gid: int, type_name: str) -> int:
    return int(node_gid - index.type_offsets[type_name])


def is_type_gid(index: SubgraphIndex, node_gid: int, type_name: str) -> bool:
    start = int(index.type_offsets[type_name])
    end = start + int(index.type_sizes[type_name])
    return start <= int(node_gid) < end



def add_record(records: List[PathRecord], template_name: str, phenotype_type: str, phenotype_gid: int, node_gids: Tuple[int, ...]) -> None:
    records.append(PathRecord(METAPATH_TO_CODE[template_name], phenotype_type, int(phenotype_gid), tuple(int(x) for x in node_gids)))


def collect_metapaths(drug_id: int, target_id: int, index: SubgraphIndex) -> List[PathRecord]:
    """Step 1: get local metapath hits for one pair.

    N1~N7 are non-DTI rules used by all settings.  N8~N17 are enabled only
    when index.use_dti_metapaths is true, which should be used for normal
    splits only.  The query pair itself is excluded from train-DTI evidence.
    """

    records: List[PathRecord] = []
    d_gid = gid(index, "drug", drug_id)
    t_gid = gid(index, "target", target_id)
    sim_drugs = [int(value) for value in index.drug_sim_neighbors[drug_id].tolist() if int(value) != int(drug_id)]
    sim_targets = [int(value) for value in index.target_sim_neighbors[target_id].tolist() if int(value) != int(target_id)]

    # N1: drug - disease - target
    for disease_id in np.intersect1d(index.drug_diseases[drug_id], index.target_diseases[target_id]).tolist():
        x = gid(index, "disease", disease_id)
        add_record(records, "N1_drug_disease_target", "disease", x, (d_gid, x, t_gid))

    # N2: drug - similar_drug - disease - target
    for sim_drug in sim_drugs:
        sd = gid(index, "drug", sim_drug)
        shared = np.intersect1d(index.drug_diseases[sim_drug], index.target_diseases[target_id])
        for disease_id in shared.tolist():
            x = gid(index, "disease", disease_id)
            add_record(records, "N2_drug_simdrug_disease_target", "disease", x, (d_gid, sd, x, t_gid))

    # N3: drug - disease - similar_target - target
    for sim_target in sim_targets:
        st = gid(index, "target", sim_target)
        shared = np.intersect1d(index.drug_diseases[drug_id], index.target_diseases[sim_target])
        for disease_id in shared.tolist():
            x = gid(index, "disease", disease_id)
            add_record(records, "N3_drug_disease_simtarget_target", "disease", x, (d_gid, x, st, t_gid))

    # N4: drug - similar_drug - disease - similar_target - target
    for sim_drug in sim_drugs:
        sd = gid(index, "drug", sim_drug)
        sim_drug_diseases = index.drug_diseases[sim_drug]
        for sim_target in sim_targets:
            st = gid(index, "target", sim_target)
            shared = np.intersect1d(sim_drug_diseases, index.target_diseases[sim_target])
            for disease_id in shared.tolist():
                x = gid(index, "disease", disease_id)
                add_record(records, "N4_drug_simdrug_disease_simtarget_target", "disease", x, (d_gid, sd, x, st, t_gid))

    # N5: drug - side_effect - drug - disease - target.
    # To avoid replacing the old DTI filter with an uncontrolled high-degree
    # side-effect expansion, bridge drugs are restricted to the current drug's
    # top-k similarity neighbors that share the side effect.
    # build_node_set() further requires the disease in an N5 path to be selected
    # by disease_topk, so side_effect_topk cannot re-expand uncontrolled diseases.
    sim_drug_set = set(sim_drugs)
    for se_id in index.drug_side_effects[drug_id].tolist():
        se_gid = gid(index, "side_effect", se_id)
        bridge_drugs = [int(value) for value in index.side_effect_drugs[int(se_id)].tolist() if int(value) in sim_drug_set]
        for bridge_drug in bridge_drugs:
            bd = gid(index, "drug", bridge_drug)
            shared = np.intersect1d(index.drug_diseases[bridge_drug], index.target_diseases[target_id])
            for disease_id in shared.tolist():
                x = gid(index, "disease", disease_id)
                add_record(records, "N5_drug_side_effect_drug_disease_target", "side_effect", se_gid, (d_gid, se_gid, bd, x, t_gid))

    # N6: drug - DDI_drug - disease - target
    for ddi_drug in index.drug_ddi_neighbors[drug_id].tolist():
        ddi_drug = int(ddi_drug)
        if ddi_drug == int(drug_id):
            continue
        dd = gid(index, "drug", ddi_drug)
        shared = np.intersect1d(index.drug_diseases[ddi_drug], index.target_diseases[target_id])
        for disease_id in shared.tolist():
            x = gid(index, "disease", disease_id)
            add_record(records, "N6_drug_ddi_drug_disease_target", "disease", x, (d_gid, dd, x, t_gid))

    # N7: drug - disease - PPI_target - target
    for ppi_target in index.target_ppi_neighbors[target_id].tolist():
        ppi_target = int(ppi_target)
        if ppi_target == int(target_id):
            continue
        pt = gid(index, "target", ppi_target)
        shared = np.intersect1d(index.drug_diseases[drug_id], index.target_diseases[ppi_target])
        for disease_id in shared.tolist():
            x = gid(index, "disease", disease_id)
            add_record(records, "N7_drug_disease_ppi_target_target", "disease", x, (d_gid, x, pt, t_gid))

    if index.use_dti_metapaths:
        collect_normal_dti_metapaths(
            drug_id=drug_id,
            target_id=target_id,
            index=index,
            records=records,
            d_gid=d_gid,
            t_gid=t_gid,
            sim_drugs=sim_drugs,
            sim_targets=sim_targets,
        )

    return records


def collect_normal_dti_metapaths(
    drug_id: int,
    target_id: int,
    index: SubgraphIndex,
    records: List[PathRecord],
    d_gid: int,
    t_gid: int,
    sim_drugs: List[int],
    sim_targets: List[int],
) -> None:
    """Add normal-only fold-local train-DTI metapaths.

    These rules add fold-local train-DTI metapaths for normal splits. The current
    query DTI edge is excluded from both drug-side and target-side train-DTI
    neighbor sets.
    """

    target_train_drugs = {
        int(value)
        for value in index.target_train_drugs[target_id].tolist()
        if int(value) != int(drug_id)
    }
    drug_train_targets = {
        int(value)
        for value in index.drug_train_targets[drug_id].tolist()
        if int(value) != int(target_id)
    }
    query_sim_targets = set(sim_targets)

    # N8: drug - side_effect - drug - train_DTI - target
    for se_id in index.drug_side_effects[drug_id].tolist():
        se_gid = gid(index, "side_effect", se_id)
        for via_drug in index.side_effect_drugs[int(se_id)].tolist():
            via_drug = int(via_drug)
            if via_drug in target_train_drugs:
                add_record(
                    records,
                    "N8_drug_side_effect_drug_train_dti_target",
                    "side_effect",
                    se_gid,
                    (d_gid, se_gid, gid(index, "drug", via_drug), t_gid),
                )

    # N9: drug - disease - drug - train_DTI - target
    for disease_id in index.drug_diseases[drug_id].tolist():
        x = gid(index, "disease", disease_id)
        for via_drug in index.disease_drugs[int(disease_id)].tolist():
            via_drug = int(via_drug)
            if via_drug in target_train_drugs:
                add_record(
                    records,
                    "N9_drug_disease_drug_train_dti_target",
                    "disease",
                    x,
                    (d_gid, x, gid(index, "drug", via_drug), t_gid),
                )

    # N10: drug - train_DTI - target - disease - target
    for via_target in drug_train_targets:
        vt = gid(index, "target", via_target)
        shared = np.intersect1d(index.target_diseases[via_target], index.target_diseases[target_id])
        for disease_id in shared.tolist():
            x = gid(index, "disease", disease_id)
            add_record(
                records,
                "N10_drug_train_dti_target_disease_target",
                "disease",
                x,
                (d_gid, vt, x, t_gid),
            )

    # N11: drug - sim_drug - disease - drug - train_DTI - target
    for sim_drug in sim_drugs:
        sd = gid(index, "drug", sim_drug)
        for disease_id in index.drug_diseases[sim_drug].tolist():
            x = gid(index, "disease", disease_id)
            for via_drug in index.disease_drugs[int(disease_id)].tolist():
                via_drug = int(via_drug)
                if via_drug in target_train_drugs:
                    add_record(
                        records,
                        "N11_drug_simdrug_disease_drug_train_dti_target",
                        "disease",
                        x,
                        (d_gid, sd, x, gid(index, "drug", via_drug), t_gid),
                    )

    # N12: drug - train_DTI - target - disease - sim_target - target
    for via_target in drug_train_targets:
        vt = gid(index, "target", via_target)
        for disease_id in index.target_diseases[via_target].tolist():
            x = gid(index, "disease", disease_id)
            for sim_target in index.disease_targets[int(disease_id)].tolist():
                sim_target = int(sim_target)
                if sim_target in query_sim_targets:
                    add_record(
                        records,
                        "N12_drug_train_dti_target_disease_simtarget_target",
                        "disease",
                        x,
                        (d_gid, vt, x, gid(index, "target", sim_target), t_gid),
                    )

    # N13: drug - sim_drug - side_effect - drug - train_DTI - target
    for sim_drug in sim_drugs:
        sd = gid(index, "drug", sim_drug)
        for se_id in index.drug_side_effects[sim_drug].tolist():
            se_gid = gid(index, "side_effect", se_id)
            for via_drug in index.side_effect_drugs[int(se_id)].tolist():
                via_drug = int(via_drug)
                if via_drug in target_train_drugs:
                    add_record(
                        records,
                        "N13_drug_simdrug_side_effect_drug_train_dti_target",
                        "side_effect",
                        se_gid,
                        (d_gid, sd, se_gid, gid(index, "drug", via_drug), t_gid),
                    )

    # N14: drug - sim_drug - train_DTI - target
    for sim_drug in sim_drugs:
        if sim_drug in target_train_drugs:
            add_record(
                records,
                "N14_drug_simdrug_train_dti_target",
                "direct",
                gid(index, "drug", sim_drug),
                (d_gid, gid(index, "drug", sim_drug), t_gid),
            )

    # N15: drug - train_DTI - sim_target - target
    for sim_target in sim_targets:
        if sim_target in drug_train_targets:
            add_record(
                records,
                "N15_drug_train_dti_simtarget_target",
                "direct",
                gid(index, "target", sim_target),
                (d_gid, gid(index, "target", sim_target), t_gid),
            )

    # N16: drug - DDI_drug - train_DTI - target
    for ddi_drug in index.drug_ddi_neighbors[drug_id].tolist():
        ddi_drug = int(ddi_drug)
        if ddi_drug in target_train_drugs:
            add_record(
                records,
                "N16_drug_ddi_drug_train_dti_target",
                "direct",
                gid(index, "drug", ddi_drug),
                (d_gid, gid(index, "drug", ddi_drug), t_gid),
            )

    # N17: drug - train_DTI - PPI_target - target
    for ppi_target in index.target_ppi_neighbors[target_id].tolist():
        ppi_target = int(ppi_target)
        if ppi_target in drug_train_targets:
            add_record(
                records,
                "N17_drug_train_dti_ppi_target_target",
                "direct",
                gid(index, "target", ppi_target),
                (d_gid, gid(index, "target", ppi_target), t_gid),
            )


def phenotype_masks(records: List[PathRecord]) -> Tuple[Dict[int, int], Dict[int, int]]:
    disease_masks: Dict[int, int] = defaultdict(int)
    side_effect_masks: Dict[int, int] = defaultdict(int)
    for record in records:
        if record.phenotype_type == "disease":
            disease_masks[record.phenotype_gid] |= 1 << record.template_idx
        elif record.phenotype_type == "side_effect":
            side_effect_masks[record.phenotype_gid] |= 1 << record.template_idx
    return disease_masks, side_effect_masks


def select_phenotypes(index: SubgraphIndex, disease_masks: Dict[int, int], side_effect_masks: Dict[int, int], disease_topk: int, side_effect_topk: int, metapath_bonus: float) -> Set[int]:
    """Step 2: filter disease and side-effect nodes."""

    disease_rows = []
    for node_gid, mask in disease_masks.items():
        specificity = float(index.disease_specificity[local_id(index, node_gid, "disease")])
        score = specificity + float(metapath_bonus) * int(bin(mask).count("1"))
        disease_rows.append((score, specificity, -node_gid, node_gid))
    disease_rows.sort(reverse=True)

    se_rows = []
    for node_gid, mask in side_effect_masks.items():
        specificity = float(index.side_effect_specificity[local_id(index, node_gid, "side_effect")])
        score = specificity + float(metapath_bonus) * int(bin(mask).count("1"))
        se_rows.append((score, specificity, -node_gid, node_gid))
    se_rows.sort(reverse=True)

    selected = {row[-1] for row in disease_rows[: max(int(disease_topk), 0)]}
    selected.update(row[-1] for row in se_rows[: max(int(side_effect_topk), 0)])
    return selected


def build_node_set(drug_id: int, target_id: int, index: SubgraphIndex, disease_topk: int, side_effect_topk: int, metapath_bonus: float) -> Tuple[List[int], int, int]:
    """Step 3: keep all nodes on metapaths involving retained phenotypes."""

    records = collect_metapaths(drug_id, target_id, index)
    disease_masks, side_effect_masks = phenotype_masks(records)
    selected_phenotypes = select_phenotypes(index, disease_masks, side_effect_masks, disease_topk, side_effect_topk, metapath_bonus)
    selected_diseases = {
        node_gid for node_gid in selected_phenotypes if is_type_gid(index, node_gid, "disease")
    }

    nodes = {gid(index, "drug", drug_id), gid(index, "target", target_id)}
    retained_paths = 0
    for record in records:
        if record.phenotype_type != "direct" and record.phenotype_gid not in selected_phenotypes:
            continue
        if record.template_idx == METAPATH_TO_CODE["N5_drug_side_effect_drug_disease_target"]:
            path_diseases = [
                node_gid for node_gid in record.node_gids if is_type_gid(index, node_gid, "disease")
            ]
            if not any(node_gid in selected_diseases for node_gid in path_diseases):
                continue
        nodes.update(record.node_gids)
        retained_paths += 1
    return sorted(nodes), len(records), retained_paths


def init_worker(index: SubgraphIndex, disease_topk: int, side_effect_topk: int, metapath_bonus: float, fold_id: int, total: int) -> None:
    global _WORKER_INDEX, _WORKER_DISEASE_TOPK, _WORKER_SIDE_EFFECT_TOPK, _WORKER_METAPATH_BONUS, _WORKER_FOLD_ID, _WORKER_TOTAL
    _WORKER_INDEX = index
    _WORKER_DISEASE_TOPK = int(disease_topk)
    _WORKER_SIDE_EFFECT_TOPK = int(side_effect_topk)
    _WORKER_METAPATH_BONUS = float(metapath_bonus)
    _WORKER_FOLD_ID = int(fold_id)
    _WORKER_TOTAL = int(total)


def process_entry_worker(task: Tuple[int, str, int, int, int]) -> Dict[str, object]:
    if _WORKER_INDEX is None:
        raise RuntimeError("Worker index is not initialized.")
    idx, split_name, drug_id, target_id, label = task
    print(
        f"fold={_WORKER_FOLD_ID} pair={idx}/{_WORKER_TOTAL} start split={split_name} "
        f"drug={drug_id} target={target_id} label={label}",
        flush=True,
    )
    nodes, total_paths, retained_paths = build_node_set(
        drug_id,
        target_id,
        _WORKER_INDEX,
        _WORKER_DISEASE_TOPK,
        _WORKER_SIDE_EFFECT_TOPK,
        _WORKER_METAPATH_BONUS,
    )
    return {
        "idx": idx,
        "split_name": split_name,
        "drug_id": drug_id,
        "target_id": target_id,
        "label": label,
        "nodes": nodes,
        "total_paths": total_paths,
        "retained_paths": retained_paths,
    }


def sample_key(drug_id: int, target_id: int) -> str:
    return f"d{drug_id}-t{target_id}"


def shard_filename(fold_id: int, shard_id: int) -> str:
    return f"fold{fold_id}_shard{shard_id:05d}.pt"


def save_shard(path: str, shard: Dict[str, Dict[str, object]]) -> None:
    tmp_path = f"{path}.tmp"
    remove_file_if_exists(tmp_path)
    torch.save(
        shard,
        tmp_path,
        _use_new_zipfile_serialization=False,
    )
    os.replace(tmp_path, path)


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold_id", "split", "drug_id", "target_id", "label", "sample_key", "shard_path", "num_nodes"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build drug-target pair subgraphs")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="li's dataset",
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


def main() -> None:
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", args.dataset_name)
    dataset_tag = os.path.basename(args.dataset_name).replace("'s dataset", "").replace(" ", "_").lower()
    ratio_count = int(args.ratio.split(":", 1)[1])
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        dataset_tag,
        f"nodes_{args.setting}_1to{ratio_count}",
    )
    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Subgraph output already exists; move it or remove it before rebuilding: {output_dir}"
        )

    use_dti_metapaths = args.setting == "normal"
    print(
        f"[Config] dataset={dataset_tag} setting={args.setting} ratio={args.ratio} "
        f"use_dti_metapaths={use_dti_metapaths} output_dir={output_dir}",
        flush=True,
    )
    shard_dir = os.path.join(output_dir, "shards")
    make_dir(output_dir)
    make_dir(shard_dir)

    for fold_id in range(5):
        fold_pairs = load_fold_pairs(data_dir, 5, args.setting, ratio_count, fold_id)
        entries = list(iter_pairs_with_meta(fold_pairs, None))
        index = build_index(
            data_dir=data_dir,
            train_pos_pairs=fold_pairs["train_pos"],
            drug_sim_topk=10,
            target_sim_topk=10,
            use_dti_metapaths=use_dti_metapaths,
        )

        rows = []
        shard_size = 1000
        current_shard_id = -1
        current_shard: Dict[str, Dict[str, object]] = {}

        def flush_shard() -> None:
            nonlocal current_shard
            if not current_shard:
                return
            shard_file = shard_filename(fold_id, current_shard_id)
            save_shard(os.path.join(shard_dir, shard_file), current_shard)
            current_shard = {}

        tasks = [
            (idx, split_name, drug_id, target_id, label)
            for idx, (split_name, drug_id, target_id, label) in enumerate(entries, start=1)
        ]
        worker_count = 16
        init_worker(index, 20, 10, 0.1, fold_id, len(entries))
        if worker_count == 1:
            results = [process_entry_worker(task) for task in tasks]
        else:
            results_by_idx: Dict[int, Dict[str, object]] = {}
            pool = Pool(
                processes=worker_count,
                initializer=init_worker,
                initargs=(index, 20, 10, 0.1, fold_id, len(entries)),
            )
            try:
                for result in pool.imap_unordered(process_entry_worker, tasks, chunksize=1):
                    results_by_idx[int(result["idx"])] = result
                    print(f"fold={fold_id} completed={len(results_by_idx)}/{len(entries)}", flush=True)
                pool.close()
            except BaseException:
                pool.terminate()
                raise
            finally:
                pool.join()
            results = [results_by_idx[idx] for idx in sorted(results_by_idx)]

        for result in results:
            idx = int(result["idx"])
            split_name = str(result["split_name"])
            drug_id = int(result["drug_id"])
            target_id = int(result["target_id"])
            label = int(result["label"])
            nodes = result["nodes"]
            total_paths = int(result["total_paths"])
            retained_paths = int(result["retained_paths"])
            shard_id = (idx - 1) // shard_size
            if shard_id != current_shard_id:
                flush_shard()
                current_shard_id = shard_id
            shard_file = shard_filename(fold_id, shard_id)
            key = sample_key(drug_id, target_id)
            current_shard[key] = {
                "drug_id": drug_id,
                "target_id": target_id,
                "label": label,
                "nodes": torch.tensor(nodes, dtype=torch.long),
            }
            rows.append({
                "fold_id": fold_id,
                "split": split_name,
                "drug_id": drug_id,
                "target_id": target_id,
                "label": label,
                "sample_key": key,
                "shard_path": shard_file,
                "num_nodes": len(nodes),
            })
            print(
                f"fold={fold_id} pair={idx}/{len(entries)} split={split_name} "
                f"drug={drug_id} target={target_id} label={label} nodes={len(nodes)} "
                f"paths={total_paths}->{retained_paths} shard={shard_file}",
                flush=True,
            )

        flush_shard()
        write_csv(os.path.join(output_dir, f"fold_{fold_id}_pairs.csv"), rows)


if __name__ == "__main__":
    main()


















