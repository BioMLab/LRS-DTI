import os

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from torch.utils.data import BatchSampler, Dataset
from torch_geometric.data import Data
from tqdm import tqdm

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ATOM_SYMBOLS = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Na",
    "Mg", "Al", "K", "Ca", "Fe", "Zn", "Cu", "Mn", "Co", "Pt", "Gd",
    "As", "Li", "Ag", "H",
]
ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
TOTAL_H_COUNTS = [0, 1, 2, 3, 4]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
ATOM_FEATURE_DIM = 62


class DrugDataset(Dataset):
    def __init__(self, data_type):
        self.data_path = data_type
        self.base_path = os.path.join(PROJECT_ROOT, "data", data_type)
        self.ids = [temp.strip() for temp in open(os.path.join(self.base_path, "drug.txt")).readlines()]

        self.structures = {}
        for temp_id in self.ids:
            sdf_path = os.path.join(self.base_path, "drug", f"{temp_id}.sdf")
            if not os.path.exists(sdf_path):
                raise FileNotFoundError(f"Missing raw SDF file: {sdf_path}")

            self.structures[temp_id] = self.read_sdf(sdf_path)

    def read_sdf(self, sdf_path):
        mol = Chem.RemoveHs(Chem.SDMolSupplier(sdf_path)[0])
        fp = torch.tensor(MACCSkeys.GenMACCSKeys(mol)).to(torch.float)

        src, dst, edge_features = [], [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feature = self.get_bond_features(bond)

            src += [i, j]
            dst += [j, i]
            edge_features += [bond_feature, bond_feature]

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float).reshape(-1, 6)

        atom_features = []
        for i, atom in enumerate(mol.GetAtoms()):
            features = self.one_hot_with_unknown(atom.GetSymbol(), ATOM_SYMBOLS)
            features += self.one_hot_with_unknown(atom.GetDegree(), ATOM_DEGREES)
            features += self.one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGES)
            features += self.one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS)
            features += self.one_hot_with_unknown(atom.GetTotalNumHs(), TOTAL_H_COUNTS)
            features += self.one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS)
            features += [
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                float(atom.GetMass()) / 200.0,
            ]
            atom_features.append(features)

        atom_feature = torch.tensor(atom_features, dtype=torch.float)
        if atom_feature.size(1) != ATOM_FEATURE_DIM:
            raise RuntimeError(
                f"Expected {ATOM_FEATURE_DIM} atom features, got {atom_feature.size(1)}"
            )
        return Data(x=atom_feature, edge_index=edge_index, edge_attr=edge_attr, fp=fp)

    @staticmethod
    def one_hot_with_unknown(value, allowable_set):
        return [float(value == item) for item in allowable_set] + [
            float(value not in allowable_set)
        ]

    def get_bond_features(self, bond):
        bond_type_map = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
        }
        b_type = bond_type_map.get(bond.GetBondType(), -1)
        return [
            b_type,
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo()),
            int(bond.GetBondDir()),
            int(bond.GetIsAromatic()),
        ]

    def __len__(self):
        return len(self.ids)

    def getIds(self, idx):
        return self.ids[idx]

    def __getitem__(self, idx):
        id_name = self.ids[idx]
        return self.structures[id_name]


class TargetDataset(Dataset):
    def __init__(self, data_type, max_len=2048):
        self.data_type = data_type
        self.data_path = os.path.join(PROJECT_ROOT, "data", data_type, "protein.txt")
        self.sequence_path = os.path.join(PROJECT_ROOT, "data", data_type, "target_sequence.txt")
        self.max_len = max_len
        self.ids = [temp.strip() for temp in open(self.data_path).readlines()]

        if not os.path.exists(self.sequence_path):
            raise FileNotFoundError(f"Missing target sequence file: {self.sequence_path}")

        self.seqs = []
        current_id = None
        sequence_parts = []
        with open(self.sequence_path, encoding="utf-8") as sequence_file:
            for line in sequence_file:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id is not None:
                        self.seqs.append((current_id, "".join(sequence_parts)))
                    current_id = line[1:].split()[0]
                    sequence_parts = []
                elif current_id is None:
                    raise ValueError(
                        f"Sequence data appears before a FASTA header in {self.sequence_path}"
                    )
                else:
                    sequence_parts.append(line)
        if current_id is not None:
            self.seqs.append((current_id, "".join(sequence_parts)))

        seq_by_id = dict(self.seqs)
        missing_ids = sorted(set(self.ids) - set(seq_by_id))
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(
                f"Missing sequences for {len(missing_ids)} targets in {self.sequence_path}: {preview}"
            )
        self.seqs = [(item_id, seq_by_id[item_id]) for item_id in dict.fromkeys(self.ids)]

        self.preprocess()

        self.esm_feature = {}
        esm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", data_type, "esm_33_full")
        unique_ids = list(dict.fromkeys(self.ids))
        expected_cache = {f"{item_id}.pth" for item_id in unique_ids}
        missing_cache = sorted(expected_cache - set(os.listdir(esm_path)))
        print(
            f"[Stage] Loading ESM embeddings from disk: unique_targets={len(unique_ids)}",
            flush=True,
        )
        if missing_cache:
            preview = ", ".join(missing_cache[:5])
            raise FileNotFoundError(
                f"Missing ESM embeddings in {esm_path}: {len(missing_cache)} files, {preview}"
            )
        for item_id in tqdm(unique_ids, desc="Loading ESM embeddings"):
            temp_path = os.path.join(esm_path, f"{item_id}.pth")
            self.esm_feature[item_id] = torch.load(temp_path)
        print("[Stage] ESM embeddings loaded", flush=True)

    def preprocess(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        layer_id = 33
        esm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", self.data_type, f"esm_{layer_id}_full")
        if not os.path.exists(esm_dir):
            os.makedirs(esm_dir, exist_ok=True)

        expected_cache = {f"{item_id}.pth" for item_id, _ in self.seqs}
        missing_cache = expected_cache - set(os.listdir(esm_dir))
        if missing_cache:
            try:
                from esm.pretrained import esm2_t33_650M_UR50D
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "Generating ESM-2 embeddings requires the Meta fair-esm package. "
                    "Install it in the active environment with: "
                    "python -m pip uninstall -y esm && python -m pip install fair-esm"
                ) from exc
            print("Missing ESM embeddings, generating them now.", flush=True)
            model, alphabet = esm2_t33_650M_UR50D()
            model = model.to(device)
            model.eval()

            batch_converter = alphabet.get_batch_converter()
            with torch.no_grad():
                for i in tqdm(range(len(self.seqs)), desc="Generating ESM embeddings"):
                    save_path = os.path.join(esm_dir, f"{self.seqs[i][0]}.pth")
                    if os.path.exists(save_path):
                        continue

                    seq = (self.seqs[i][0], self.seqs[i][1])
                    _, _, tokens = batch_converter([seq])
                    tokens = tokens.to(device)
                    outputs = model(tokens, repr_layers=[layer_id], return_contacts=False)
                    temp = outputs["representations"][layer_id].squeeze().detach().cpu()
                    torch.save(temp, save_path)

    def __len__(self):
        return len(self.ids)

    def getIds(self, idx):
        return self.ids[idx]

    def __getitem__(self, idx):
        item_id = self.ids[idx]
        return self.esm_feature[item_id]


class FixedRatioBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size):
        super().__init__(None, batch_size, False)

        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.batch_size = int(batch_size)
        self.pos_indices = torch.where(self.labels == 1)[0]
        self.neg_indices = torch.where(self.labels == 0)[0]

        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for fixed-ratio sampling.")
        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError("FixedRatioBatchSampler requires both positive and negative samples.")

        pos_ratio = len(self.pos_indices) / len(self.labels)
        self.pos_per_batch = int(round(self.batch_size * pos_ratio))
        self.pos_per_batch = min(max(self.pos_per_batch, 1), self.batch_size - 1)
        self.neg_per_batch = self.batch_size - self.pos_per_batch
        self.batch_count = min(
            len(self.pos_indices) // self.pos_per_batch,
            len(self.neg_indices) // self.neg_per_batch,
        )

        if self.batch_count == 0:
            raise ValueError("Not enough samples to build one fixed-ratio batch.")

    def __iter__(self):
        pos_indices = self.pos_indices[torch.randperm(len(self.pos_indices))]
        neg_indices = self.neg_indices[torch.randperm(len(self.neg_indices))]

        for i in range(self.batch_count):
            pos_start = i * self.pos_per_batch
            neg_start = i * self.neg_per_batch
            batch_pos = pos_indices[pos_start : pos_start + self.pos_per_batch]
            batch_neg = neg_indices[neg_start : neg_start + self.neg_per_batch]
            batch = torch.cat([batch_pos, batch_neg], dim=0)
            batch = batch[torch.randperm(len(batch))]
            yield batch.tolist()

    def __len__(self):
        return self.batch_count
