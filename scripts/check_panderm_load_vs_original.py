#!/usr/bin/env python3
"""Run PanDerm's ORIGINAL run_class_finetuning.py (unmodified) up to checkpoint loading, intercept
furnace utils.load_state_dict, save the loaded encoder state, stop. Then compare with our backbones.build_panderm.

    python scripts/check_panderm_load_vs_original.py --out logs/panderm_load_vs_original.json
"""
import argparse, json, os, runpy, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="/workspace/PanDerm/classification")
ap.add_argument("--ckpt", default="/workspace/checkpoints/panderm_ll_data6_checkpoint-499.pth")
ap.add_argument("--csv", default="/workspace/data/partition/panderm_profile.csv")
ap.add_argument("--root", default="/workspace/data/images/")
ap.add_argument("--out", required=True)
a = ap.parse_args()
proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.abspath(a.out)


class Stop(Exception):
    pass


captured = {}
os.chdir(a.repo)
sys.path.insert(0, a.repo)
os.environ["WANDB_MODE"] = "disabled"
import furnace.utils as fu  # noqa: E402
_orig = fu.load_state_dict


def hook(model, state_dict, prefix="", **kw):
    _orig(model, state_dict, prefix=prefix, **kw)
    captured["sd"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    raise Stop


fu.load_state_dict = hook
# same CLI as logs/original_cmd.sh (model/loading flags identical)
sys.argv = ["run_class_finetuning.py", "--model", "PanDerm_Large_FT", "--pretrained_checkpoint", a.ckpt,
            "--nb_classes", "2", "--batch_size", "128", "--lr", "5e-4", "--epochs", "1", "--layer_decay", "0.65",
            "--drop_path", "0.2", "--weight_decay", "0.05", "--sin_pos_emb", "--no_auto_resume", "--no_save_ckpt",
            "--imagenet_default_mean_and_std", "--num_workers", "2", "--seed", "0", "--exp_name", "loadcheck",
            "--wandb_name", "loadcheck", "--output_dir", "/tmp/panderm_loadcheck", "--csv_path", a.csv,
            "--root_path", a.root]
try:
    runpy.run_path("run_class_finetuning.py", run_name="__main__")
except Stop:
    pass
orig = captured["sd"]
os.chdir(proj)
sys.path.insert(0, os.path.join(proj, "scripts"))
from backbones import build_panderm  # noqa: E402
ours = build_panderm(os.path.dirname(a.repo), a.ckpt).state_dict()
ours = {k: v.detach().cpu() for k, v in ours.items()}
common = sorted(set(orig) & set(ours))
diffs = {k: float((orig[k].float() - ours[k].float()).abs().max()) for k in common if orig[k].shape == ours[k].shape}
shape_mismatch = [k for k in common if orig[k].shape != ours[k].shape]
rpb = [k for k in orig if k.endswith("relative_position_bias_table")]
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
ck = ck.get("model", ck)
res = {
    "checkpoint_keys": len(ck),
    "checkpoint_keys_with_relative_position_or_table": [k for k in ck if "relative_position" in k or "table" in k],
    "original_rel_pos_tables": len(rpb),
    "original_rel_pos_tables_all_zero": all(float(orig[k].abs().max()) == 0.0 for k in rpb),
    "ours_rel_pos_tables_all_zero": all(float(ours[k].abs().max()) == 0.0 for k in rpb if k in ours),
    "only_in_original": sorted(set(orig) - set(ours)),
    "only_in_ours": sorted(set(ours) - set(orig)),
    "shape_mismatch": shape_mismatch,
    "n_compared": len(diffs),
    "max_abs_diff_all_compared": max(diffs.values()),
    "nonzero_diff_keys": {k: v for k, v in diffs.items() if v != 0.0},
}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
