import subprocess
import os

import torch


def getGpu():
    if not torch.cuda.is_available():
        return "cpu"
    visible_count = torch.cuda.device_count()
    if visible_count <= 0:
        return "cpu"

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )

        lines = result.stdout.strip().split("\n")
        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        visible_physical_ids = None
        if visible_env and visible_env not in {"-1", "none", "None"}:
            tokens = [token.strip() for token in visible_env.split(",") if token.strip()]
            if tokens and all(token.isdigit() for token in tokens):
                visible_physical_ids = [int(token) for token in tokens]

        gpus = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            free_mem, util = map(int, line.split(","))
            if visible_physical_ids is not None and i not in visible_physical_ids:
                continue
            gpus.append((i, free_mem, util))

        candidates = [(i, free, util) for i, free, util in gpus if free > 10000]
        if candidates:
            best = sorted(candidates, key=lambda x: -x[1])[0]
            if visible_physical_ids is not None:
                local_idx = visible_physical_ids.index(best[0])
                if local_idx < visible_count:
                    return f"cuda:{local_idx}"
            elif best[0] < visible_count:
                return f"cuda:{best[0]}"
            return "cuda:0"
        return "cuda:0"
    except Exception:
        return "cuda:0"


if __name__ == "__main__":
    print(getGpu())
