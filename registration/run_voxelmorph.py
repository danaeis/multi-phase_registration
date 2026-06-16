"""
baselines/run_voxelmorph.py  —  B4: VoxelMorph deformable registration.

Two modes:
  --mode infer   Load a pretrained checkpoint and run inference (fast, default).
  --mode train   Train from scratch on your dataset first (~1 GPU-day), then infer.

Requirements:
    pip install voxelmorph tensorflow  (or torch if using PyTorch backend)
    A GPU is strongly recommended for training; inference runs on CPU too.

Loss during training: NMI (normalised mutual information) — NOT MSE/NCC on raw
HU, which is invalid across contrast phases.

Usage:
    # Training (run once, saves checkpoint):
    python -m baselines.run_voxelmorph --mode train --input aligned --gpu 0

    # Inference with a pretrained checkpoint:
    python -m baselines.run_voxelmorph --mode infer --checkpoint /path/to/model.h5 --input both

    # If you already have warped outputs from a prior run and just need eval:
    python evaluate_all.py --conditions B4_vxm__aligned B4_vxm__baseline
"""
from __future__ import annotations
import argparse, os, sys, tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
try:
    from . import _common as K
except ImportError:
    import _common as K

import compare_config as C
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import evaluate_pipeline.evaluate_registration as E

ALGO_TAG   = "B4_vxm"
INSHAPE    = (192, 192, 192)   # resize all inputs to this before VXM
VOL_SHAPE  = INSHAPE
NMI_BINS   = 32
LAMBDA_REG = 0.5               # regularisation weight (bend energy)


def _check_vxm():
    try:
        import voxelmorph as vxm
        return vxm
    except ImportError:
        raise ImportError(
            "voxelmorph not installed.\n"
            "Install: pip install voxelmorph tensorflow\n"
            "Docs: https://github.com/voxelmorph/voxelmorph"
        )


def _resize_vol(vol_np: np.ndarray, target_shape) -> np.ndarray:
    """Trilinear resize to target_shape (Z,Y,X) via SimpleITK."""
    img = sitk.GetImageFromArray(vol_np.astype(np.float32))
    out = sitk.Resample(
        img,
        size=[int(s) for s in reversed(target_shape)],
        transform=sitk.Transform(),
        interpolator=sitk.sitkLinear,
        outputOrigin=img.GetOrigin(),
        outputSpacing=[img.GetSpacing()[i] * img.GetSize()[i] / target_shape[2-i]
                       for i in range(3)],
        outputDirection=img.GetDirection(),
        defaultPixelValue=-1024.0,
    )
    return sitk.GetArrayFromImage(out)


def _normalise(vol: np.ndarray) -> np.ndarray:
    """Normalise HU to [0,1] for VoxelMorph input (clip first to body range)."""
    v = np.clip(vol, -1024, 1024).astype(np.float32)
    return (v + 1024) / 2048.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(input_key: str = "aligned", gpu: int = 0,
          epochs: int = 100, steps_per_epoch: int = 100,
          checkpoint_dir: str = None):
    """
    Train VoxelMorph on your dataset with NMI loss.
    Checkpoint saved to results/B4_vxm_checkpoint/ by default.
    """
    vxm = _check_vxm()
    import tensorflow as tf

    if gpu >= 0:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[gpu], 'GPU')

    if checkpoint_dir is None:
        checkpoint_dir = str(C.RESULTS_DIR / "B4_vxm_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    import pandas as pd
    labels_df = pd.read_csv(str(C.LABELS_CSV))
    src = C.INPUTS[input_key]
    items = list(K.iter_pairs(labels_df, src.dir, src.vol_postfix, src.seg_postfix))
    print(f"Training on {len(items)} pairs from {input_key} input", flush=True)

    # Build model
    nb_feats = [16, 32, 32, 32, 32, 16]
    model = vxm.networks.VxmDense(
        inshape=VOL_SHAPE,
        nb_unet_features=nb_feats,
        int_steps=7,
    )

    # NMI loss — valid across contrast phases
    losses = [vxm.losses.NMI(image_sigma=0.05, nb_bins=NMI_BINS).loss,
              vxm.losses.Grad('l2').loss]
    loss_weights = [1.0, LAMBDA_REG]
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss=losses, loss_weights=loss_weights)

    def data_gen():
        rng = np.random.default_rng(42)
        zero_phi = np.zeros([1, *VOL_SHAPE, 3], dtype=np.float32)
        while True:
            item = items[rng.integers(len(items))]
            f_np = _normalise(_resize_vol(
                sitk.GetArrayFromImage(sitk.ReadImage(item["fixed_vol"])), VOL_SHAPE))
            m_np = _normalise(_resize_vol(
                sitk.GetArrayFromImage(sitk.ReadImage(item["moving_vol"])), VOL_SHAPE))
            yield ([m_np[None,...,None], f_np[None,...,None]],
                   [f_np[None,...,None], zero_phi])

    cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "vxm_{epoch:03d}.h5"),
        save_best_only=False, save_freq='epoch')
    model.fit(data_gen(), epochs=epochs, steps_per_epoch=steps_per_epoch,
              callbacks=[cb])
    best = os.path.join(checkpoint_dir, f"vxm_{epochs:03d}.h5")
    print(f"Training done. Final checkpoint: {best}", flush=True)
    return best


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _infer_one(vxm, model, fixed: sitk.Image, moving: sitk.Image,
               moving_seg: sitk.Image, item: dict) -> K.RegResult:
    """Run inference for one pair; warp back to original fixed grid."""
    f_np = sitk.GetArrayFromImage(fixed).astype(np.float32)
    m_np = sitk.GetArrayFromImage(moving).astype(np.float32)
    orig_shape = np.array(f_np.shape)   # (Z,Y,X)

    f_r = _normalise(_resize_vol(f_np, VOL_SHAPE))
    m_r = _normalise(_resize_vol(m_np, VOL_SHAPE))

    pred = model.predict([m_r[None,...,None], f_r[None,...,None]], verbose=0)
    warped_r = pred[0][0,...,0]          # (Z,Y,X) warped moving in INSHAPE space
    dvf_r    = pred[1][0]                # (Z,Y,X,3) displacement in voxels (INSHAPE)

    # Scale DVF voxels back to original grid voxels
    scale = (orig_shape / np.array(VOL_SHAPE)).astype(np.float32)
    dvf_orig_vox = np.zeros(list(orig_shape) + [3], np.float32)
    for c in range(3):
        ch = _resize_vol(dvf_r[..., c], tuple(orig_shape.tolist()))
        dvf_orig_vox[..., c] = ch * scale[c]
    # component order: VXM outputs (z,y,x) — no reorder needed
    sp = E.get_spacing_zyx(fixed)
    dvf_mm = K.voxel_dvf_to_mm(dvf_orig_vox, sp)

    # Build a SimpleITK DisplacementFieldTransform from the DVF
    # VXM field (z,y,x) → SimpleITK expects (x,y,z) component order
    dvf_xyz = np.ascontiguousarray(dvf_orig_vox[..., ::-1])
    sp_xyz  = fixed.GetSpacing()
    # convert voxels to mm for sitk
    for c in range(3):
        dvf_xyz[..., c] *= sp_xyz[c]
    dvf_sitk = sitk.GetImageFromArray(dvf_xyz, isVector=True)
    dvf_sitk.CopyInformation(fixed)
    interp_tx = sitk.DisplacementFieldTransform(dvf_sitk)

    warped_vol = K.warp_volume(moving, fixed, interp_tx)
    warped_seg = K.warp_label(moving_seg, fixed, interp_tx)

    return K.RegResult(warped_vol=warped_vol, warped_seg=warped_seg,
                       dvf_zyx3_mm=dvf_mm)


def make_infer_fn(checkpoint_path: str):
    """Returns a register_fn that uses the loaded VXM model."""
    vxm = _check_vxm()
    print(f"Loading VoxelMorph checkpoint: {checkpoint_path}", flush=True)
    model = vxm.networks.VxmDense.load(checkpoint_path)
    print("Model loaded.", flush=True)

    def register_vxm(fixed, moving, moving_seg, item):
        return _infer_one(vxm, model, fixed, moving, moving_seg, item)
    return register_vxm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VoxelMorph B4 runner")
    p.add_argument("--mode", choices=["train", "infer"], default="infer")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .h5 checkpoint (required for infer).")
    p.add_argument("--input", choices=["aligned", "baseline", "both"], default="both")
    p.add_argument("--studies", nargs="*", default=None)
    p.add_argument("--labels_csv", default=str(C.LABELS_CSV))
    p.add_argument("--no-skip", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()

    if args.mode == "train":
        ckpt = train(input_key=args.input if args.input != "both" else "aligned",
                     gpu=args.gpu, epochs=args.epochs, steps_per_epoch=args.steps)
        print(f"Use --mode infer --checkpoint {ckpt} to run evaluation.")
        sys.exit(0)

    if args.checkpoint is None:
        print("ERROR: --checkpoint is required for --mode infer")
        sys.exit(1)

    import pandas as pd
    labels_df = pd.read_csv(args.labels_csv)
    register_fn = make_infer_fn(args.checkpoint)
    inputs = ["aligned", "baseline"] if args.input == "both" else [args.input]
    algo = C.BASELINE_ALGOS[ALGO_TAG]
    for ikey in inputs:
        if ikey == "baseline" and not algo.run_on_baseline:
            continue
        K.run_baseline(ALGO_TAG, ikey, register_fn, labels_df,
                       studies=args.studies, skip_existing=not args.no_skip)
