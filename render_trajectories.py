#!/usr/bin/env python3
"""
render_trajectories.py
======================

Batch visualisation of the annotated trajectories written by annotate_xyz.py.

  video   One MP4 per trajectory, rendered headless with the free OVITO Python
          module (pip install ovito -- MIT licence, no Pro key). The camera is a
          fixed-zoom, top-down orthographic view that stays put while the
          filament moves, so displacement is visible on screen, and jumps to
          re-centre on the filament when (a) any bead comes within MARGIN of
          the frame edge, or (b) every EVERY video frames, if set.

  still   One PNG per trajectory (matplotlib only, no OVITO): the head-bead path
          coloured by time, with chain snapshots overlaid.

Bonds come from the bead ids (bead i -- bead i+1), exactly as in
topology_N<n>.data, so no topology file and no "Load trajectory" step are
needed, and the per-frame attributes in the comment line still reach the
text overlay.

USAGE
-----
    python render_trajectories.py video  annotated/*.xyz --out videos --jobs 4
    python render_trajectories.py video  run.xyz --half-height 60 --every 200
    python render_trajectories.py video  run.xyz --stride 5 --fps 30
    python render_trajectories.py still  annotated/*.xyz --out stills

Requires: numpy; video also needs ovito >= 3.10 and ffmpeg on PATH; still
needs matplotlib. Headless Linux needs a Vulkan driver (GPU vendor driver, or
`apt install mesa-vulkan-drivers` for CPU rendering).
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np


# --------------------------------------------------------------------------
# XYZ reading (shared by both modes; no OVITO needed)
# --------------------------------------------------------------------------
def parse_comment(line: str) -> dict[str, str]:
    """key=value pairs of an extended-XYZ comment line (values may be quoted)."""
    out = {}
    for tok in shlex.split(line):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def column_of(properties: str, name: str) -> int | None:
    """First column index of property `name` in a Properties= string."""
    f = properties.split(":")
    col = 0
    for i in range(0, len(f) - 2, 3):
        if f[i] == name:
            return col
        col += int(f[i + 2])
    return None


def read_xyz(path: str):
    """Positions (F, N, 2), the bead ids in chain order, and frame-0 metadata.

    Beads are sorted by their id column, so index 0 is id 1. Assumes a fixed
    bead count and storage order across frames, which annotate_xyz.py
    guarantees.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[0].split()[0])
    stride = n + 2
    n_frames = len(lines) // stride
    if n_frames * stride != len(lines) and lines[n_frames * stride :] != [""]:
        # tolerate a trailing blank line only
        extra = [ln for ln in lines[n_frames * stride :] if ln.strip()]
        if extra:
            raise ValueError(f"{path}: truncated last frame")
    meta = parse_comment(lines[1])
    props = meta.get("Properties", "species:S:1:pos:R:3")
    xc = column_of(props, "pos")
    ic = column_of(props, "id")
    if xc is None:
        raise ValueError(f"{path}: no pos column in Properties={props}")

    atom_lines = [
        ln for f in range(n_frames) for ln in lines[f * stride + 2 : (f + 1) * stride]
    ]
    usecols = (xc, xc + 1) if ic is None else (xc, xc + 1, ic)
    arr = np.loadtxt(atom_lines, usecols=usecols, dtype=float)
    arr = arr.reshape(n_frames, n, len(usecols))
    ids = arr[0, :, 2].astype(int) if ic is not None else np.arange(1, n + 1)
    order = np.argsort(ids)
    return arr[:, order, :2], ids[order], meta


# --------------------------------------------------------------------------
# camera schedule (pure numpy -- testable without rendering)
# --------------------------------------------------------------------------
def reference_points(pos: np.ndarray, mode: str, head_index: int) -> np.ndarray:
    if mode == "com":
        return pos.mean(axis=1)
    if mode == "bbox":
        return 0.5 * (pos.min(axis=1) + pos.max(axis=1))
    if mode == "head":
        return pos[:, head_index, :]
    raise ValueError(mode)


def camera_schedule(pos, ref, half_w, half_h, margin, every, bead_radius):
    """Camera centre per video frame, and the frames on which it jumped.

    The camera holds still; it re-centres on `ref` when any bead's edge enters
    the outer `margin` fraction of the window, or every `every` frames (0 =
    never). Decided on the frames that will actually be rendered.
    """
    inner = np.array([half_w, half_h]) * (1.0 - margin)
    centers = np.empty((len(pos), 2))
    jumps = []
    c = ref[0].copy()
    for k in range(len(pos)):
        lo = pos[k].min(axis=0) - bead_radius
        hi = pos[k].max(axis=0) + bead_radius
        escaping = np.any(lo < c - inner) or np.any(hi > c + inner)
        periodic = every > 0 and k > 0 and k % every == 0
        if k == 0 or escaping or periodic:
            c = ref[k].copy()
            if k:
                jumps.append(k)
        centers[k] = c
    return centers, jumps


# --------------------------------------------------------------------------
# video
# --------------------------------------------------------------------------
def render_video(xyz: str, args) -> str:
    from ovito.io import import_file
    from ovito.modifiers import (
        ClearSelectionModifier,
        ExpressionSelectionModifier,
        GenerateTrajectoryLinesModifier,
    )
    from ovito.qt_compat import QtCore
    from ovito.vis import OpenGLRenderer, TachyonRenderer, TextLabelOverlay, Viewport

    t0 = time.time()
    stem = os.path.splitext(os.path.basename(xyz))[0]
    out_mp4 = os.path.join(args.out, stem + ".mp4")
    if os.path.exists(out_mp4) and not args.overwrite:
        return f"skip  {stem} (exists)"

    pos, ids, meta = read_xyz(xyz)
    n = pos.shape[1]
    head_index = int(np.nonzero(ids == args.head_id)[0][0])
    frames = np.arange(0, len(pos), args.stride)
    p = pos[frames]

    width, height = args.size
    # Measured: for an orthographic view, Viewport.fov is the half-extent
    # along the SHORTER image side, in simulation length units.
    bond = np.median(np.linalg.norm(np.diff(pos[0], axis=0), axis=1))
    half_short = args.half_height or 1.25 * (n - 1) * bond
    aspect = max(width, height) / min(width, height)
    half_w, half_h = (
        (half_short * aspect, half_short)
        if width >= height
        else (half_short, half_short * aspect)
    )
    extent = (p.max(axis=1) - p.min(axis=1)).max(axis=0) + 2 * args.radius
    if np.any(extent > 2 * np.array([half_w, half_h]) * (1 - args.margin)):
        print(
            f"  warning {stem}: chain extent {extent.round(1)} exceeds the inner "
            f"window; it will re-centre every frame. Increase --half-height.",
            file=sys.stderr,
        )
    ref = reference_points(p, args.center_on, head_index)
    centers, jumps = camera_schedule(
        p, ref, half_w, half_h, args.margin, args.every, args.radius
    )

    # ---- pipeline ----------------------------------------------------------
    pipeline = import_file(xyz)
    head_id, radius, bond_w = args.head_id, args.radius, args.bond_width

    def style(frame, data):
        """Backbone bonds by id, uniform radius, head bead coloured."""
        pid = np.asarray(data.particles.identifiers)
        order = np.argsort(pid)
        pairs = np.column_stack([order[:-1], order[1:]])
        bonds = data.particles_.create_bonds(vis_params={"width": bond_w})
        bonds.create_property("Topology", data=pairs)
        data.particles_.create_property("Radius", data=np.full(len(pid), radius))
        col = np.tile([0.30, 0.35, 0.85], (len(pid), 1))
        col[pid == head_id] = [0.90, 0.25, 0.20]
        data.particles_.create_property("Color", data=col)

    pipeline.modifiers.append(style)
    if args.trail:
        pipeline.modifiers.append(
            ExpressionSelectionModifier(expression=f"ParticleIdentifier == {head_id}")
        )
        tl = GenerateTrajectoryLinesModifier(only_selected=True)
        tl.vis.width = args.trail_width
        tl.vis.color = (0.55, 0.55, 0.55)
        tl.vis.upto_current_time = True  # trail grows with the video
        pipeline.modifiers.append(tl)
        pipeline.modifiers.append(ClearSelectionModifier())  # else beads render red
    data0 = pipeline.compute(0)
    if data0.cell is not None:
        data0.cell.vis.enabled = False
    pipeline.add_to_scene()

    vp = Viewport(type=Viewport.Type.Top, fov=half_short)
    label = TextLabelOverlay(
        source_pipeline=pipeline,
        text=args.label,
        format_string="%.4g",
        font_family=args.font,
        font_size=0.04,
        text_color=(0, 0, 0),
        offset_x=0.01,
        offset_y=-0.01,
    )
    info = TextLabelOverlay(
        font_family=args.font,
        font_size=0.03,
        text_color=(0.35, 0.35, 0.35),
        alignment=QtCore.Qt.AlignmentFlag.AlignLeft
        | QtCore.Qt.AlignmentFlag.AlignBottom,
        offset_x=0.01,
        offset_y=0.01,
    )
    vp.overlays.append(label)
    vp.overlays.append(info)
    renderer = TachyonRenderer() if args.renderer == "tachyon" else OpenGLRenderer()

    # ---- frames -> PNG -> MP4 ----------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=stem + "_", dir=args.tmp)
    jump_set = set(jumps)
    last_jump = -10**9
    try:
        for k, f in enumerate(frames):
            cx, cy = centers[k]
            vp.camera_pos = (cx, cy, 50.0)
            if k in jump_set:
                last_jump = k
            flag = "  re-centred" if k - last_jump < args.flag_frames else ""
            info.text = (
                f"view {2 * half_w:.0f} x {2 * half_h:.0f} (centre "
                f"{cx:.1f}, {cy:.1f}){flag}"
            )
            vp.render_image(
                size=(width, height),
                frame=int(f),
                filename=os.path.join(tmp, f"{k:06d}.png"),
                background=(1, 1, 1),
                renderer=renderer,
            )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(args.fps),
                "-i", os.path.join(tmp, "%06d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
                out_mp4,
            ],
            check=True,
        )
    finally:
        if not args.keep_frames:
            shutil.rmtree(tmp, ignore_errors=True)
    return (
        f"video {stem}: {len(frames)} frames, {len(jumps)} re-centres, "
        f"{time.time() - t0:.0f} s -> {out_mp4}"
    )


# --------------------------------------------------------------------------
# still
# --------------------------------------------------------------------------
def render_still(xyz: str, args) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    stem = os.path.splitext(os.path.basename(xyz))[0]
    out_png = os.path.join(args.out, stem + ".png")
    if os.path.exists(out_png) and not args.overwrite:
        return f"skip  {stem} (exists)"
    pos, ids, meta = read_xyz(xyz)
    head = pos[:, int(np.nonzero(ids == args.head_id)[0][0]), :]
    com = pos.mean(axis=1)
    F = len(pos)

    fig, ax = plt.subplots(figsize=(7, 7))
    path = head if args.path == "head" else com
    seg = np.stack([path[:-1], path[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", linewidths=1.2)
    lc.set_array(np.arange(F - 1))
    ax.add_collection(lc)
    snaps = np.linspace(0, F - 1, args.snapshots).astype(int)
    cmap = plt.get_cmap("viridis")
    for s in snaps:
        c = cmap(s / max(F - 1, 1))
        ax.plot(pos[s, :, 0], pos[s, :, 1], "-", color=c, lw=2.5, alpha=0.8)
        ax.plot(*head[s], "o", color="k", ms=4)  # head bead marks the direction
    ax.set_aspect("equal")
    ax.autoscale()
    cb = fig.colorbar(lc, ax=ax, shrink=0.8)
    cb.set_label("frame")
    title = stem
    if "activity" in meta:
        title = (
            f"F = {float(meta['activity']):g},  K = {float(meta['bending']):g},  "
            f"theta = {float(meta['ChiralityAngle']):g} deg\n{stem}"
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return f"still {stem} -> {out_png}"


# --------------------------------------------------------------------------
def _work(job):
    fn, xyz, args = job
    try:
        return fn(xyz, args)
    except Exception as e:  # one bad file must not kill the batch
        return f"FAIL  {os.path.basename(xyz)}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("files", nargs="+", help="annotated trajectory_*.xyz files")
    common.add_argument("--out", default="renders")
    common.add_argument("--head-id", type=int, default=1,
                        help="bead id of the leading (head) bead (default 1)")
    common.add_argument("--jobs", type=int, default=1, help="files in parallel")
    common.add_argument("--overwrite", action="store_true")

    v = sub.add_parser("video", parents=[common])
    v.add_argument("--half-height", type=float, default=None,
                   help="half-extent of the shorter image side, in sigma "
                        "(default 1.25 x contour length)")
    v.add_argument("--margin", type=float, default=0.10,
                   help="re-centre when a bead enters this edge fraction")
    v.add_argument("--every", type=int, default=0,
                   help="also re-centre every N video frames (0 = off)")
    v.add_argument("--center-on", choices=["com", "bbox", "head"], default="com")
    v.add_argument("--stride", type=int, default=1, help="render every k-th frame")
    v.add_argument("--fps", type=int, default=30)
    v.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    v.add_argument("--renderer", choices=["opengl", "tachyon"], default="opengl")
    v.add_argument("--radius", type=float, default=0.5)
    v.add_argument("--bond-width", type=float, default=0.6)
    v.add_argument("--trail", action=argparse.BooleanOptionalAction, default=True,
                   help="draw the head bead's past path (default on)")
    v.add_argument("--trail-width", type=float, default=0.25)
    v.add_argument("--label", default="F = [activity]   K = [bending]   "
                   "theta = [ChiralityAngle] deg   frame [SourceFrame]")
    v.add_argument("--font", default="Helvetica" if sys.platform == "darwin"
                   else "DejaVu Sans",
                   help="explicit font; OVITO's default fallback font clipped labels")
    v.add_argument("--flag-frames", type=int, default=15,
                   help="show 're-centred' for this many frames after a jump")
    v.add_argument("--crf", type=int, default=20, help="x264 quality (lower=better)")
    v.add_argument("--tmp", default=None, help="scratch dir for PNG frames")
    v.add_argument("--keep-frames", action="store_true")

    s = sub.add_parser("still", parents=[common])
    s.add_argument("--path", choices=["head", "com"], default="head")
    s.add_argument("--snapshots", type=int, default=6)
    s.add_argument("--dpi", type=int, default=200)

    args = ap.parse_args()
    if args.mode == "video" and shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH")
    fn = render_video if args.mode == "video" else render_still
    jobs = [(fn, f, args) for f in args.files]
    if args.jobs > 1:
        with multiprocessing.get_context("spawn").Pool(args.jobs) as pool:
            for msg in pool.imap_unordered(_work, jobs):
                print(msg, flush=True)
    else:
        for j in jobs:
            print(_work(j), flush=True)


if __name__ == "__main__":
    main()
