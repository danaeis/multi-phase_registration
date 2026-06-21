"""
baselines/run_voxelmorph.py  —  B4: VoxelMorph deformable registration.
PyTorch backend (avoids TF/Keras 3 incompatibility).

Two modes:
  --mode train   Train from scratch on your dataset first (~1 GPU-day), then infer.
  --mode infer   Load a pretrained checkpoint and run inference (fast).

Two loss options (train separate checkpoints to compare):
  --loss nmi     NMI (normalised mutual information) — phase-invariant, RECOMMENDED
                 for multi-contrast CT. This is the proposed/custom loss.
  --loss ncc     NCC (normalised cross-correlation) — original VoxelMorph default.
                 Invalid across contrast phases but included for ablation comparison.

Requirements (already satisfied in your ncct_env):
    pip install git+https://github.com/adalca/neurite.git
    pip install git+https://github.com/voxelmorph/voxelmorph.git
    torch >= 2.0  (you have 2.6+cu124)

Usage:
    # Train with NMI (proposed, phase-invariant):
    python run_voxelmorph.py --mode train --loss nmi --input aligned --gpu 0 --epochs 100

    # Train with NCC (original VXM baseline for ablation):
    python run_voxelmorph.py --mode train --loss ncc --input aligned --gpu 0 --epochs 100

    # Inference with a saved checkpoint:
    python run_voxelmorph.py --mode infer --checkpoint /path/to/vxm_nmi_100.pt --input both

    # Already have warped outputs? Just evaluate:
    python evaluate_all.py --conditions B4_vxm__aligned B4_vxm__baseline
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import time

import numpy as np
import SimpleITK as sitk
import torch

try:
    from profiling import profile_training, report_model_stats, PROFILING_DIR as _PDIR
    _PROFILING = True
except ImportError:
    _PROFILING = False

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
try:
    from ..registration import _common as K
except ImportError:
    import _common as K

import compare_config as C

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import evaluate_pipeline.evaluate_registration as E

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALGO_TAG   = "B4_vxm"
INSHAPE    = (192, 192, 192)   # resize all inputs to this before VXM
VOL_SHAPE  = INSHAPE
NMI_BINS   = 32
LAMBDA_REG = 0.5               # regularisation weight (diffusion / bend energy)
NB_FEATS   = [[16, 32, 32, 32], [32, 32, 32, 32, 16, 16]]  # encoder/decoder


# ---------------------------------------------------------------------------
# Backend bootstrap — force PyTorch to avoid TF/Keras 3 breakage
# ---------------------------------------------------------------------------

def _load_vxm_pytorch():
    """
    Import voxelmorph with the PyTorch backend.

    Strategy 1: import voxelmorph.torch subpackage directly — avoids the
    TF-priority race in the top-level __init__ and the sibling-except bug
    (an ImportError raised inside `except AttributeError` is NOT caught by a
    later `except ImportError` at the same try-level; it propagates to the
    caller).
    Strategy 2: clear module cache, set VXM_BACKEND, reimport top-level.
    """
    import types, sys
    os.environ["VXM_BACKEND"] = "pytorch"

    # Strategy 1: direct subpackage import
    try:
        import voxelmorph.torch as _vxm_pt
        vxm = types.SimpleNamespace(networks=_vxm_pt.networks,
                                    losses=_vxm_pt.losses)
        _ = vxm.networks.VxmDense  # confirm PyTorch classes present
        return vxm
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    # Strategy 2: flush cached modules so VXM_BACKEND env var takes effect
    for k in list(sys.modules.keys()):
        if k == "voxelmorph" or k.startswith("voxelmorph."):
            del sys.modules[k]
    try:
        import voxelmorph as vxm
        _ = vxm.networks.VxmDense
        return vxm
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    raise ImportError(
        "voxelmorph has no PyTorch backend. Reinstall from GitHub:\n"
        "  pip uninstall voxelmorph neurite -y\n"
        "  pip install git+https://github.com/adalca/neurite.git\n"
        "  pip install git+https://github.com/voxelmorph/voxelmorph.git\n"
        "Docs: https://github.com/voxelmorph/voxelmorph"
    )


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _resize_vol(vol_np: np.ndarray, target_shape) -> np.ndarray:
    """Trilinear resize to target_shape (Z,Y,X) via SimpleITK."""
    img = sitk.GetImageFromArray(vol_np.astype(np.float32))
    out = sitk.Resample(
        img,
        size=[int(s) for s in reversed(target_shape)],   # sitk wants (X,Y,Z)
        transform=sitk.Transform(),
        interpolator=sitk.sitkLinear,
        outputOrigin=img.GetOrigin(),
        outputSpacing=[
            img.GetSpacing()[i] * img.GetSize()[i] / target_shape[2 - i]
            for i in range(3)
        ],
        outputDirection=img.GetDirection(),
        defaultPixelValue=-1024.0,
    )
    return sitk.GetArrayFromImage(out)


def _normalise(vol: np.ndarray) -> np.ndarray:
    """Clip HU to body range and normalise to [0, 1]."""
    v = np.clip(vol, -1024, 1024).astype(np.float32)
    return (v + 1024) / 2048.0


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """(Z,Y,X) numpy -> (1,1,Z,Y,X) float32 tensor on device."""
    return torch.from_numpy(arr[None, None]).to(device)


# ---------------------------------------------------------------------------
# Loss builder
# ---------------------------------------------------------------------------

def _build_losses(vxm, loss_type: str):
    """
    Return (sim_loss_fn, reg_loss_fn) for the chosen loss_type.

    nmi  — NMI(bins=32): phase-invariant, correct for multi-contrast CT.
    ncc  — NCC(win=9):   original VoxelMorph default (valid only for same-contrast).
    """
    loss_type = loss_type.lower()
    if loss_type == "nmi":
        # FIX 4: was NMI(image_sigma=0.05, nb_bins=NMI_BINS) — `image_sigma`
        # is not a valid VoxelMorph NMI constructor parameter → TypeError at runtime.
        sim_loss = vxm.losses.NMI(nb_bins=NMI_BINS).loss
    elif loss_type == "ncc":
        sim_loss = vxm.losses.NCC(win=[9, 9, 9]).loss
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'. Choose 'nmi' or 'ncc'.")

    reg_loss = vxm.losses.Grad("l2", loss_mult=LAMBDA_REG).loss
    return sim_loss, reg_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    input_key: str = "aligned",
    gpu: int = 0,
    epochs: int = 100,
    steps_per_epoch: int = 100,
    checkpoint_dir: str = None,
    loss_type: str = "nmi",
):
    """
    Train VoxelMorph (PyTorch) on your dataset.

    Checkpoints are saved as:
        {checkpoint_dir}/vxm_{loss_type}_{epoch:03d}.pt
    """
    vxm = _load_vxm_pytorch()

    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if checkpoint_dir is None:
        checkpoint_dir = str(C.RESULTS_DIR / f"B4_vxm_checkpoint_{loss_type}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    import pandas as pd

    labels_df = pd.read_csv(str(C.LABELS_CSV))

    # Filter to train-split studies only (avoids evaluating on training data).
    # Run `python generate_split.py` once to create the split file.
    try:
        from generate_split import load_split
        train_studies = load_split("train")
        labels_df = labels_df[labels_df["study_id"].isin(train_studies)].reset_index(drop=True)
        print(f"Using {len(train_studies)} train-split studies (split file found)", flush=True)
    except FileNotFoundError:
        print("WARNING: no split file found — training on ALL studies. "
              "Run `python generate_split.py` first.", flush=True)

    src = C.INPUTS[input_key]
    items = list(K.iter_pairs(labels_df, src.dir, src.vol_postfix, src.seg_postfix))
    print(f"Training on {len(items)} pairs from '{input_key}' input", flush=True)
    print(f"Loss: {loss_type.upper()}  |  lambda_reg={LAMBDA_REG}", flush=True)

    model = vxm.networks.VxmDense(
        inshape=VOL_SHAPE,
        nb_unet_features=NB_FEATS,
        int_steps=7,
        int_downsize=2,
    ).to(device)

    # Report model stats (params + FLOPs) before training starts
    if _PROFILING:
        model_stats = report_model_stats(model, VOL_SHAPE, device)
    else:
        model_stats = {}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    sim_loss_fn, reg_loss_fn = _build_losses(vxm, loss_type)

    rng = np.random.default_rng(42)

    _train_ctx = (
        profile_training(ALGO_TAG, loss_type, len(items), epochs, steps_per_epoch,
                         _PDIR, gpu=gpu)
        if _PROFILING else __import__("contextlib").nullcontext()
    )

    with _train_ctx as train_rec:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_sim = 0.0
            epoch_reg = 0.0

            for step in range(steps_per_epoch):
                item = items[rng.integers(len(items))]

                f_np = _normalise(_resize_vol(
                    sitk.GetArrayFromImage(sitk.ReadImage(item["fixed_vol"])), VOL_SHAPE))
                m_np = _normalise(_resize_vol(
                    sitk.GetArrayFromImage(sitk.ReadImage(item["moving_vol"])), VOL_SHAPE))

                fixed_t  = _to_tensor(f_np, device)
                moving_t = _to_tensor(m_np, device)

                optimizer.zero_grad()
                moved_t, flow_t = model(moving_t, fixed_t, registration=False)

                loss_sim = sim_loss_fn(fixed_t, moved_t)
                loss_reg = reg_loss_fn(None, flow_t)
                loss = loss_sim + loss_reg

                loss.backward()
                optimizer.step()

                epoch_sim += loss_sim.item()
                epoch_reg += loss_reg.item()

            avg_sim = epoch_sim / steps_per_epoch
            avg_reg = epoch_reg / steps_per_epoch
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"sim({loss_type})={avg_sim:.4f}  reg={avg_reg:.4f}  "
                f"total={avg_sim + avg_reg:.4f}",
                flush=True,
            )

            ckpt_path = os.path.join(checkpoint_dir, f"vxm_{loss_type}_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "loss_type": loss_type,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "inshape": VOL_SHAPE,
                    "nb_unet_features": NB_FEATS,
                    "model_stats": model_stats,
                },
                ckpt_path,
            )

        if _PROFILING and train_rec is not None:
            train_rec.update(model_stats)

    final_ckpt = os.path.join(checkpoint_dir, f"vxm_{loss_type}_{epochs:03d}.pt")
    print(f"\nTraining done. Final checkpoint: {final_ckpt}", flush=True)
    print(f"Run inference with:\n"
          f"  python run_voxelmorph.py --mode infer --checkpoint {final_ckpt} "
          f"--loss {loss_type} --input both", flush=True)
    return final_ckpt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _infer_one(
    vxm,
    model: torch.nn.Module,
    device: torch.device,
    fixed: sitk.Image,
    moving: sitk.Image,
    moving_seg: sitk.Image,
    item: dict,
) -> K.RegResult:
    """Run inference for one pair; DVF is upsampled back to the original grid."""
    f_np = sitk.GetArrayFromImage(fixed).astype(np.float32)
    m_np = sitk.GetArrayFromImage(moving).astype(np.float32)
    orig_shape = np.array(f_np.shape)   # (Z, Y, X) numpy order

    f_r = _normalise(_resize_vol(f_np, VOL_SHAPE))
    m_r = _normalise(_resize_vol(m_np, VOL_SHAPE))

    fixed_t  = _to_tensor(f_r, device)
    moving_t = _to_tensor(m_r, device)

    model.eval()
    with torch.no_grad():
        # Warm-up pass so CUDA launch latency doesn't inflate the first measurement.
        if device.type == "cuda":
            _ = model(moving_t, fixed_t, registration=True)
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        moved_t, flow_t = model(moving_t, fixed_t, registration=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_sec = time.perf_counter() - t0
    print(f"    vxm inference: {infer_sec*1000:.1f} ms", flush=True)

    # ── DVF component order ──────────────────────────────────────────────────
    # FIX 2+3: the original code claimed "PyTorch VXM outputs (x,y,z)" — wrong.
    #
    # VoxelMorph PyTorch SpatialTransformer builds its identity grid from
    # torch.meshgrid([z_coords, y_coords, x_coords]), so flow channel c is
    # the displacement along axis c of the (Z,Y,X) volume:
    #   c=0 → z-displacement,  c=1 → y-displacement,  c=2 → x-displacement
    # After moveaxis: (Z,Y,X,3) array, component order = (z,y,x).
    #
    # The original code then did [..,::-1] calling it "zyx" and passed it to
    # voxel_dvf_to_mm (expects zyx), while the UN-flipped (actually zyx) array
    # was used for the sitk warp (needs xyz). The two arrays were exactly swapped.
    # ─────────────────────────────────────────────────────────────────────────
    flow_np = flow_t[0].cpu().numpy()          # (3, Z, Y, X), comp order: (z,y,x)
    dvf_r   = np.moveaxis(flow_np, 0, -1)      # (Z, Y, X, 3), comp order: (z,y,x)

    # Upsample from INSHAPE grid to orig_shape grid and scale displacement values.
    # scale[c] = orig_dim_c / inshape_dim_c, matching the (Z,Y,X) axis numbering.
    scale = (orig_shape / np.array(VOL_SHAPE)).astype(np.float32)
    dvf_orig_vox_zyx = np.zeros((*orig_shape, 3), np.float32)
    for c in range(3):
        ch = _resize_vol(dvf_r[..., c], tuple(orig_shape.tolist()))
        dvf_orig_vox_zyx[..., c] = ch * scale[c]
    # dvf_orig_vox_zyx: (Z,Y,X,3), component order (z,y,x), units = orig voxels

    # ── Saved DVF: (z,y,x) components, mm ───────────────────────────────────
    sp_zyx = E.get_spacing_zyx(fixed)           # (z-sp, y-sp, x-sp)
    dvf_mm = K.voxel_dvf_to_mm(dvf_orig_vox_zyx, sp_zyx)
    # dvf_mm: (Z,Y,X,3), (z,y,x) order, mm  ← correct input for neg_jacobian_pct

    # ── sitk DisplacementFieldTransform: (x,y,z) components, mm ────────────
    # sitk vector images store components in (x,y,z) order: flip the last axis.
    sp_xyz = fixed.GetSpacing()                 # (x-sp, y-sp, z-sp) from sitk
    dvf_sitk_arr = dvf_orig_vox_zyx[..., ::-1].copy()   # (Z,Y,X,3), comp: (x,y,z)
    for c in range(3):
        dvf_sitk_arr[..., c] *= sp_xyz[c]      # voxels→mm: x-sp*x-disp, y-sp*y-disp, z-sp*z-disp
    dvf_sitk_img = sitk.GetImageFromArray(np.ascontiguousarray(dvf_sitk_arr), isVector=True)
    dvf_sitk_img.CopyInformation(fixed)
    interp_tx = sitk.DisplacementFieldTransform(dvf_sitk_img)

    warped_vol = K.warp_volume(moving, fixed, interp_tx)
    warped_seg = K.warp_label(moving_seg, fixed, interp_tx)

    return K.RegResult(
        warped_vol=warped_vol,
        warped_seg=warped_seg,
        dvf_zyx3_mm=dvf_mm,
    )


def _load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a .pt checkpoint saved by this script's train()."""
    vxm = _load_vxm_pytorch()
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)

    ckpt = torch.load(checkpoint_path, map_location=device)
    inshape      = ckpt.get("inshape", VOL_SHAPE)
    nb_unet_feat = ckpt.get("nb_unet_features", NB_FEATS)

    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=nb_unet_feat,
        int_steps=7,
        int_downsize=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loss_type = ckpt.get("loss_type", "unknown")
    print(f"Model loaded (loss={loss_type})", flush=True)

    if _PROFILING:
        report_model_stats(model, tuple(inshape), device)

    return vxm, model


def make_infer_fn(checkpoint_path: str, gpu: int = 0):
    """Returns a register_fn(fixed, moving, moving_seg, item) -> RegResult."""
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
    vxm, model = _load_checkpoint(checkpoint_path, device)

    def register_vxm(fixed, moving, moving_seg, item):
        return _infer_one(vxm, model, device, fixed, moving, moving_seg, item)

    return register_vxm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VoxelMorph B4 runner (PyTorch backend)")
    p.add_argument("--mode",       choices=["train", "infer"], default="infer")
    p.add_argument("--loss",       choices=["nmi", "ncc"],     default="nmi",
                   help="Similarity loss: 'nmi' (phase-invariant, proposed) "
                        "or 'ncc' (original VXM, ablation baseline).")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt checkpoint (required for --mode infer).")
    p.add_argument("--input",      choices=["aligned", "baseline", "both"], default="both")
    p.add_argument("--studies",    nargs="*", default=None)
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--no-skip",    action="store_true")
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--steps",      type=int, default=100)
    args = p.parse_args()

    # ── Train ──────────────────────────────────────────────────────────────
    if args.mode == "train":
        ckpt = train(
            input_key=args.input if args.input != "both" else "aligned",
            gpu=args.gpu,
            epochs=args.epochs,
            steps_per_epoch=args.steps,
            loss_type=args.loss,
        )
        sys.exit(0)

    # ── Infer ──────────────────────────────────────────────────────────────
    if args.checkpoint is None:
        print("ERROR: --checkpoint is required for --mode infer")
        sys.exit(1)

    import pandas as pd

    labels_df   = pd.read_csv(args.labels_csv)
    register_fn = make_infer_fn(args.checkpoint, gpu=args.gpu)
    inputs      = ["aligned", "baseline"] if args.input == "both" else [args.input]
    algo        = C.BASELINE_ALGOS[ALGO_TAG]

    for ikey in inputs:
        if ikey == "baseline" and not algo.run_on_baseline:
            continue
        K.run_baseline(
            ALGO_TAG, ikey, register_fn, labels_df,
            studies=args.studies, skip_existing=not args.no_skip,
        )