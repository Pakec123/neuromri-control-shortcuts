"""Dispatch an immutable P1 CSV row without shell interpolation."""
import argparse
from pathlib import Path
import pandas as pd
import yaml
from .common import config_cli
from .p1_finetune import train


def main():
    import os
    p=argparse.ArgumentParser()
    p.add_argument("--runlist",required=True)
    p.add_argument("--row",type=int,required=True)
    a=p.parse_args()
    runs=pd.read_csv(a.runlist)
    if a.row >= len(runs):
        print("No row in last partial group:",a.row)
        return
    row=runs.iloc[a.row].to_dict()
    c=yaml.safe_load(Path("configs/p1.yaml").read_text())
    c.update({k:v for k,v in row.items() if k in c})
    for k in ["seed","fold","batch_size","epochs"]:
        c[k]=int(c[k])
    c["data_root"]=os.environ["NEUROMRI_DATA"]
    c["weights_dir"]=os.environ.get("NEUROMRI_WEIGHTS","weights/files")
    c["output"]=f"results/p1/{row['run_id']}"
    if (Path(c["output"])/"metrics.csv").exists():
        import json
        from .common import config_hash
        prior=json.loads((Path(c["output"])/"provenance.json").read_text())
        if prior["config_sha256"] != config_hash(c):
            raise RuntimeError("Completed row has a different config; use a fresh result directory")
        print("Already complete:",c["output"])
        return
    train(c)


if __name__=="__main__":
    main()
