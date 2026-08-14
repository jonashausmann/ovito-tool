#!/usr/bin/env python3
"""
annotate_xyz.py
===============

Batch-annotate every simulation run under a data root so that OVITO can display
per-frame analysis values as live text overlays.

HOW IT WORKS
------------
OVITO reads the "comment line" (2nd line of every frame) of an *extended XYZ*
file as a set of key=value pairs, and exposes every key as a **global
attribute**. The built-in "Text label" viewport layer can then reference those
attributes with placeholders like [Rg], and OVITO re-evaluates them at every
frame. This works in free OVITO Basic -- no Pro license, no Python script
modifier required.

Each frame's comment line gets

  * every numeric column of the run's ``csv_files/*_measurements.csv``, and
  * the three constants read from the run's ``input_values.txt``:
    ``activity``, ``bending`` and ``ChiralityAngle`` (converted to DEGREES).

    120
    Lattice="..." Properties=species:S:1:pos:R:3 Frame=57 Rg=4.7231 ...
      activity=5.000000 bending=0.500000 ChiralityAngle=45.000000
    A  1.234  5.678  0.000
    ...

DIRECTORY LAYOUT
----------------
    <root>/                       e.g. .../new-python-polymer-analysis/data
      F05_K0.5_theta45.0/         parameter directory
        RUN_0001/                 run directory
          config0.xyz             input trajectory
          input_values.txt        constants (key:value per line)
          csv_files/
            ..._measurements.csv  per-frame analysis values

Directories without a ``config0.xyz`` (e.g. ``Averaged_Runs``) are skipped.

Every annotated trajectory is written flat into a single OUTPUT_DIR -- set it
on the OUTPUT_DIR line in the CONFIG block below (e.g. a path on an external
disk). The run number is appended so the five runs of one parameter set do not
collide:

    <OUTPUT_DIR>/trajectory_F5.0_K0.5_T45.0_Run1.xyz
    <OUTPUT_DIR>/trajectory_F5.0_K0.5_T45.0_Run2.xyz
    ...

DRAWING THE CHAIN AS A LINE
---------------------------
XYZ carries no bonds, and a distance cutoff cannot recover them: bonded
neighbours reach ~1.03 while non-adjacent monomers close to ~0.88, so
"Create bonds" would stitch the chain to itself wherever it folds. Instead a
``topology_N<n>.data`` LAMMPS file is written alongside the trajectories, one
per distinct chain length, holding N beads and the N-1 backbone bonds. Load
that file in OVITO, then add "Load trajectory" pointed at a trajectory_*.xyz;
the ``id`` column written into each atom line matches beads to the topology by
identity rather than by storage order.

HIGHLIGHTING INDIVIDUAL MONOMERS
--------------------------------
Any CSV column naming a monomer (``max_curv_monomer``, ``min_curv_monomer``,
...) is already a per-frame global attribute, so OVITO can colour that bead and
its neighbours with an "Expression selection" + "Assign color" pair -- the
selection follows the monomer as it moves from frame to frame, and unticking
the two modifiers switches the highlight off. Nothing is baked into the .xyz
files, so a new highlight costs no rewriting.

The groups are declared in the HIGHLIGHTS list in the CONFIG block; the script
checks them against the data and writes the exact expressions and colours to
``<OUTPUT_DIR>/ovito_setup.txt``. Add an entry, rerun with --guide-only, and
the instructions refresh in seconds.

USAGE
-----
    python annotate_xyz.py                     # every run under DATA_ROOT
    python annotate_xyz.py --dry-run           # list what would be written
    python annotate_xyz.py --only 'F05_*'      # subset by path pattern
    python annotate_xyz.py --limit 20          # first 20 pending runs
    python annotate_xyz.py --out-dir /Volumes/big/annotated   # override OUTPUT_DIR
    python annotate_xyz.py --jobs 8 --overwrite

The annotated copies are slightly larger than the inputs, so the run refuses to
start unless the target volume has room for the estimate (override: --force).

Requires: pandas, numpy.
"""

from __future__ import annotations

import argparse
import errno
import fnmatch
import functools
import glob
import math
import multiprocessing
import os
import re
import shutil
import sys
from typing import NamedTuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG -- edit these, or override with command-line flags
# --------------------------------------------------------------------------
DATA_ROOT = "./data"

# >>> Every annotated .xyz lands here, in one flat directory. CHANGE THIS LINE
# >>> to point at your external disk (mount it first). Also: --out-dir PATH.
OUTPUT_DIR = "./output"

XYZ_NAME = "config0.xyz"  # input trajectory, inside each run dir
INPUT_VALUES_NAME = "input_values.txt"  # constants, inside each run dir
CSV_GLOBS = (  # per-frame analysis CSV, first match wins
    "csv_files/*_measurements.csv",
    "csv-files/*_measurements.csv",
)

# Constants lifted from input_values.txt into every frame's comment line.
# ChiralityAngle is stored in radians there and written out in degrees.
CONST_PARAMS = ("activity", "bending", "ChiralityAngle")


class Highlight(NamedTuple):
    """A group of monomers to colour differently in OVITO, defined per frame.

    ``anchor`` names a CSV column holding a monomer index; the group is that
    bead and the ``span - 1`` beads after it. Because the anchor is already
    written into every frame's comment line as a global attribute, OVITO can
    resolve the group itself with an expression -- so adding a highlight needs
    two modifiers in the GUI and NO regeneration of the trajectories.
    """

    name: str  # label, used in the generated guide
    anchor: str  # CSV column holding the monomer index
    span: int  # beads in the group, counting from the anchor
    color: tuple  # R, G, B in 0..1
    note: str  # what it means, for the guide


# Add an entry here to define another highlight; anything in the CSV that names
# a monomer works. Nothing needs rewriting afterwards -- rerun with --guide-only
# to refresh the instructions.
#
# A curvature angle is measured at bead m from the beads m, m+1 and m+2 (verified
# against the geometry: the angle reconstructed at those three beads reproduces
# the CSV's max_curv/min_curv exactly), hence span=3.
HIGHLIGHTS = (
    Highlight(
        "Max curvature",
        "max_curv_monomer",
        3,
        (1.00, 0.20, 0.20),
        "the sharpest bend in the chain",
    ),
    Highlight(
        "Min curvature",
        "min_curv_monomer",
        3,
        (0.10, 0.85, 0.30),
        # Green, not blue: OVITO's default colour for particle type 2 is already
        # a blue-violet, and a blue highlight is hard to pick out against it.
        "the straightest joint in the chain",
    ),
)

# Which CSV columns to embed. None = every numeric column except the frame column.
COLUMNS = None
# e.g. ["Rg", "head_theta", "end_to_end"]

# Name of the frame-index column in the CSV. None = auto-detect.
FRAME_COLUMN = None  # e.g. "frames"

# Annotated output is bigger than the input (longer comment lines). Used only
# for the pre-flight free-space estimate.
SIZE_GROWTH = 1.30
# --------------------------------------------------------------------------


FRAME_COLUMN_CANDIDATES = (
    "frame",
    "frames",
    "frame_index",
    "frame_number",
    "step",
    "timestep",
    "time_step",
    "t",
    "index",
)


class RunError(Exception):
    """A single run could not be annotated; the batch carries on without it."""


class DiskFull(Exception):
    """The volume filled up mid-write; the whole batch has to stop."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sanitize(name: str) -> str:
    """Turn a CSV header into a legal extended-XYZ key / OVITO attribute name.

    OVITO placeholders are [Name]; keys must not contain whitespace, '=' or ']'.
    """
    clean = re.sub(r"[^0-9A-Za-z_.]", "_", str(name).strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "value"
    if clean[0].isdigit():
        clean = "v" + clean
    return clean


def is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def fmt_param(value: float, decimals: int = 4) -> str:
    """Render a parameter for the output filename: 5.0, 0.5, 45.0, 300.0.

    ChiralityAngle round-trips through radians, so 45 deg comes back as
    44.99998823...; rounding first keeps the filenames readable.
    """
    s = f"{round(float(value), decimals):.{decimals}f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def detect_frame_column(df: pd.DataFrame, explicit: str | None) -> str | None:
    if explicit is not None:
        if explicit not in df.columns:
            raise RunError(
                f"frame column '{explicit}' not found in CSV. "
                f"Available: {list(df.columns)}"
            )
        return explicit
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in FRAME_COLUMN_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    return None  # fall back to positional row order


def iter_frames(path: str):
    """Stream an XYZ file frame by frame.

    Yields (natoms, comment_line, [atom_line, ...]). Memory-safe for long runs.
    """
    with open(path, "r") as fh:
        while True:
            header = fh.readline()
            if not header:
                return
            if not header.strip():
                continue  # tolerate blank lines between frames
            try:
                natoms = int(header.split()[0])
            except (ValueError, IndexError):
                raise RunError(f"expected an atom count, got: {header!r}")
            comment = fh.readline().rstrip("\n").rstrip("\r")
            atoms = []
            for _ in range(natoms):
                line = fh.readline()
                if not line:
                    raise RunError("file ended mid-frame -- truncated XYZ?")
                atoms.append(line.rstrip("\n").rstrip("\r"))
            yield natoms, comment, atoms


def decide_layout(tokens: list[str], override: str):
    """Decide (has_species, n_coords) from one representative atom line.

    The naive test 'is the first token non-numeric?' fails on files whose
    species column is a numeric type ID, e.g. "2 -0.105 0.139 0.0". So the
    token COUNT is the primary signal, matching the XYZ convention of
    <species> <x> <y> <z>.
    """
    n = len(tokens)
    if override != "auto":
        return {
            "species3": (True, 3),
            "species2": (True, 2),
            "plain3": (False, 3),
            "plain2": (False, 2),
        }[override]

    if n >= 4:
        # Standard XYZ: first column is the species/type, numeric or not.
        return True, 3
    if n == 3:
        # Ambiguous: "type x y" vs "x y z". Fall back to the numeric test.
        return (False, 3) if is_float(tokens[0]) else (True, 2)
    if n == 2:
        return False, 2
    raise RunError(f"cannot interpret atom line with {n} column(s): {' '.join(tokens)}")


def scan_geometry(path: str, layout: str = "auto"):
    """First pass: global bounding box and the atom-line column layout."""
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    n_frames = 0
    has_species = None
    n_coords = None
    n_tokens = None
    sample = None

    for _, _, atoms in iter_frames(path):
        n_frames += 1
        for line in atoms:
            tok = line.split()
            if not tok:
                continue
            if has_species is None:
                has_species, n_coords = decide_layout(tok, layout)
                n_tokens = len(tok)
                sample = tok
            start = 1 if has_species else 0
            xyz = np.zeros(3)
            for k in range(n_coords):
                xyz[k] = float(tok[start + k])
            lo = np.minimum(lo, xyz)
            hi = np.maximum(hi, xyz)

    if n_frames == 0:
        raise RunError("no frames found in the XYZ file.")
    n_extra = n_tokens - (1 if has_species else 0) - n_coords
    return n_frames, has_species, n_coords, n_extra, sample, lo, hi


def build_lattice(lo, hi, pad_frac=0.05) -> str:
    """Fixed, trajectory-spanning cell so the camera does not jump between frames."""
    span = hi - lo
    span = np.where(span > 0, span, 1.0)
    pad = pad_frac * span
    L = span + 2 * pad
    org = lo - pad
    return (
        f'Lattice="{L[0]:.6f} 0.0 0.0 0.0 {L[1]:.6f} 0.0 '
        f'0.0 0.0 {max(L[2], 1e-6):.6f}" '
        f'Origin="{org[0]:.6f} {org[1]:.6f} {org[2]:.6f}"'
    )


def format_value(val, precision: int) -> str | None:
    """Render a CSV cell as an extended-XYZ value. Returns None to skip."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return "T" if val else "F"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            # Emit a placeholder rather than omitting the key: a missing attribute
            # would make OVITO print the literal "[Rg]" on that frame.
            return "nan"
        return (
            f"{f:.{precision}g}"
            if abs(f) and (abs(f) < 1e-4 or abs(f) >= 1e6)
            else f"{f:.{precision}f}"
        )
    s = str(val).strip()
    if not s:
        return None
    if any(ch in s for ch in ' \t"='):
        s = '"' + s.replace('"', "'") + '"'
    return s


# --------------------------------------------------------------------------
# run discovery
# --------------------------------------------------------------------------
def discover_runs(root: str) -> list[str]:
    """Every <root>/<parameter dir>/<run dir> that holds a trajectory."""
    runs = []
    for param in sorted(os.listdir(root)):
        param_dir = os.path.join(root, param)
        if not os.path.isdir(param_dir):
            continue
        for run in sorted(os.listdir(param_dir)):
            run_dir = os.path.join(param_dir, run)
            if os.path.isdir(run_dir) and os.path.isfile(
                os.path.join(run_dir, XYZ_NAME)
            ):
                runs.append(run_dir)
    return runs


def read_input_values(path: str) -> dict[str, str]:
    """Parse the 'key:value' lines of input_values.txt."""
    values = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, val = line.split(":", 1)
            values[key.strip()] = val.strip()
    return values


def run_parameters(run_dir: str) -> dict[str, float]:
    """The constants for one run. ChiralityAngle is converted to degrees."""
    path = os.path.join(run_dir, INPUT_VALUES_NAME)
    if not os.path.isfile(path):
        raise RunError(f"no {INPUT_VALUES_NAME}")
    raw = read_input_values(path)

    params = {}
    for key in CONST_PARAMS:
        if key not in raw:
            raise RunError(f"'{key}' missing from {INPUT_VALUES_NAME}")
        try:
            params[key] = float(raw[key])
        except ValueError:
            raise RunError(
                f"'{key}' is not a number in {INPUT_VALUES_NAME}: {raw[key]!r}"
            )

    params["ChiralityAngle"] = params["ChiralityAngle"] * 180.0 / np.pi
    return params


def find_csv(run_dir: str) -> str:
    matches = []
    for pattern in CSV_GLOBS:
        matches.extend(sorted(glob.glob(os.path.join(run_dir, pattern))))
    if not matches:
        raise RunError("no *_measurements.csv")
    if len(matches) > 1:
        names = ", ".join(os.path.basename(m) for m in matches)
        raise RunError(f"{len(matches)} measurement CSVs, cannot choose: {names}")
    return matches[0]


def atom_count(xyz_path: str) -> int:
    """Monomer count = the atom count on the first frame's header line."""
    with open(xyz_path) as fh:
        for line in fh:
            if line.strip():
                try:
                    return int(line.split()[0])
                except (ValueError, IndexError):
                    raise RunError(f"expected an atom count, got: {line!r}")
    raise RunError("empty XYZ file")


def first_species(xyz_path: str) -> int:
    """The numeric particle type on the first atom line, or 1 if it is not one.

    The topology has to declare the same type the trajectory carries, otherwise
    OVITO warns that "numeric particle type ID 2 ... does not exist" and falls
    back to the default colour and radius for every bead.
    """
    for natoms, _, atoms in iter_frames(xyz_path):
        for line in atoms:
            tok = line.split()
            if len(tok) >= 4 and re.fullmatch(r"\d+", tok[0]):
                return int(tok[0])
            return 1
    return 1


def write_topology(
    n_atoms: str | int, out_dir: str, box: float = 500.0, atom_type: int = 1
) -> str:
    """A LAMMPS data file holding the chain connectivity: N beads, N-1 bonds.

    XYZ cannot carry bonds, and a distance cutoff cannot recover them here --
    bonds stretch to ~1.03 while non-adjacent monomers approach ~0.88, so
    Create Bonds would cross-link the chain wherever it folds onto itself.
    Load this file in OVITO and stream the coordinates onto it with the
    'Load trajectory' modifier. One file serves every run of equal length.

    Coordinates below are placeholders (a straight line) -- the trajectory
    overwrites them, as does its Lattice, so the huge box here only exists to
    keep the placeholders unwrapped and never reaches the viewport.
    """
    n = int(n_atoms)
    t = max(1, int(atom_type))
    path = os.path.join(out_dir, f"topology_N{n}.data")
    with open(path, "w") as fh:
        fh.write(
            f"LAMMPS data file -- linear polymer topology, {n} monomers, "
            f"{n - 1} bonds (generated by annotate_xyz.py)\n\n"
        )
        fh.write(f"{n} atoms\n{n - 1} bonds\n\n")
        # Declaring t types (not 1) so the type the trajectory carries exists.
        fh.write(f"{t} atom types\n1 bond types\n\n")
        for axis in "xyz":
            fh.write(f"{-box:.1f} {box:.1f} {axis}lo {axis}hi\n")
        fh.write("\nAtoms # bond\n\n")
        for i in range(1, n + 1):
            # atom-ID molecule-ID atom-type x y z
            fh.write(f"{i} 1 {t} {float(i - 1):.1f} 0.0 0.0\n")
        fh.write("\nBonds\n\n")
        for i in range(1, n):
            # bond-ID bond-type atom-1 atom-2
            fh.write(f"{i} 1 {i} {i + 1}\n")
    return path


def highlight_expression(h: Highlight) -> str:
    """The OVITO 'Expression selection' text that picks out this group.

    The anchor resolves to the frame's global attribute, so the selection
    follows the moving monomer without any per-particle data in the file.
    """
    key = sanitize(h.anchor)
    if h.span == 1:
        return f"ParticleIdentifier == {key}"
    return f"ParticleIdentifier >= {key} && ParticleIdentifier <= {key} + {h.span - 1}"


def write_highlight_guide(out_dir: str, chain_lengths, highlights=HIGHLIGHTS) -> str:
    """Write ovito_setup.txt: the click-by-click recipe, next to the data.

    It lives beside the trajectories so it travels with them onto the external
    disk, where this script will not be.
    """
    path = os.path.join(out_dir, "ovito_setup.txt")
    # Without a known chain length, leave the placeholder rather than guess a
    # filename the reader would go looking for and not find.
    n = sorted(chain_lengths)[0] if chain_lengths else "<n>"
    with open(path, "w") as fh:
        fh.write(
            "OVITO setup for these trajectories\n"
            "==================================\n"
            "Generated by annotate_xyz.py. Everything below works in OVITO Basic;\n"
            "no Python script modifier, no Pro licence.\n\n"
            "1. DRAW THE POLYMER AS A CONNECTED LINE\n"
            "---------------------------------------\n"
            f"  a. File > Load File ...    topology_N{n}.data\n"
            "     A straight row of beads in a huge box appears. Both are\n"
            "     placeholders that the next step replaces.\n"
            "  b. Add modifier > Load trajectory ...   trajectory_F*.xyz\n"
            "     Beads are matched to the topology by their id column, so the\n"
            "     bonds follow the chain however it folds.\n"
            "  c. Bonds visual element > set the display width (default 0.4).\n\n"
            "  The bonds come from the topology file, not from a distance cutoff:\n"
            "  bonded neighbours reach ~1.03 while non-adjacent monomers close to\n"
            "  ~0.88, so 'Create bonds' would stitch the chain to itself wherever\n"
            "  it folds.\n\n"
            "2. SHOW THE PER-FRAME NUMBERS\n"
            "-----------------------------\n"
            "  Viewport layers > Text label. Reference any attribute in [brackets]:\n"
            "    F = [activity]<br>K = [bending]<br>theta = [ChiralityAngle] deg\n"
            "    <br>Rg = [Rg]<br>Frame [Frame]\n\n"
            "3. HIGHLIGHT MONOMERS\n"
            "---------------------\n"
            "  For each highlight below, add TWO modifiers, in this order:\n"
            "    Add modifier > Expression selection   -> paste the expression\n"
            "    Add modifier > Assign color           -> pick the colour\n"
            "  Then, once, at the TOP of the pipeline list (i.e. added last):\n"
            "    Add modifier > Clear selection\n"
            "  so the last group is not left tinted red by OVITO's selection\n"
            "  highlighting in the interactive viewport.\n\n"
            "  The bonds between highlighted beads change colour automatically:\n"
            "  the Bonds visual element defaults to 'Use particle colors'.\n\n"
            "  TO TURN A HIGHLIGHT OFF: untick the checkbox next to its two\n"
            "  modifiers in the pipeline. Tick it again to bring it back.\n\n"
        )
        for h in highlights:
            r, g, b = h.color
            fh.write(
                f"  {h.name.upper()}  --  {h.note}\n"
                f"    anchor attribute : {sanitize(h.anchor)} "
                f"({h.span} beads: m, m+1, m+2)\n"
                f"    expression       : {highlight_expression(h)}\n"
                f"    colour           : R {r:.2f}  G {g:.2f}  B {b:.2f}"
                f"   (0-255: {round(r * 255)}, {round(g * 255)}, {round(b * 255)})\n\n"
            )
        fh.write(
            "4. ADDING YOUR OWN HIGHLIGHT LATER\n"
            "----------------------------------\n"
            "  Every numeric CSV column is already a global attribute in every\n"
            "  frame, so any column naming a monomer can drive a highlight with\n"
            "  no rewriting of the .xyz files. Just add the two modifiers with\n"
            "    ParticleIdentifier >= <column> && ParticleIdentifier <= <column> + <span-1>\n"
            "  To have annotate_xyz.py document it here too, add a Highlight(...)\n"
            "  entry to the HIGHLIGHTS list near the top of the script and rerun\n"
            "  with --guide-only.\n\n"
            "  Careful with the index convention: <column> is used directly as a\n"
            "  particle id. That is correct for the curvature columns above, where\n"
            "  the value is a 0-based bead index and ids are 1-based, which shifts\n"
            "  the group onto the three beads that actually form the angle. A\n"
            "  column using a different convention needs its own offset.\n\n"
            "  Attributes available on every frame:\n"
        )
        for key in ("Frame", "frames") + CONST_PARAMS:
            fh.write(f"    {key}\n")
        fh.write("    ... plus every numeric column of the run's measurements CSV\n")
    return path


def run_suffix(run_dir: str) -> str:
    """'RUN_0003' -> '_Run3'. Keeps the five runs of one parameter set apart
    now that every trajectory lands in the same output directory."""
    name = os.path.basename(os.path.normpath(run_dir))
    m = re.search(r"(\d+)\s*$", name)
    return f"_Run{int(m.group(1))}" if m else "_" + sanitize(name)


def output_path(run_dir: str, params: dict[str, float], args) -> str:
    name = (
        f"trajectory_F{fmt_param(params['activity'])}"
        f"_K{fmt_param(params['bending'])}"
        f"_T{fmt_param(params['ChiralityAngle'])}"
        f"{run_suffix(run_dir)}.xyz"
    )
    return os.path.join(args.out_dir, name)


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
def annotate_run(run_dir: str, args) -> dict:
    """Annotate a single run. Returns a result dict; never raises RunError."""
    result = {
        "run": run_dir,
        "status": "ok",
        "message": "",
        "out": None,
        "frames": 0,
        "details": [],
    }
    try:
        params = run_parameters(run_dir)
        out_path = output_path(run_dir, params, args)
        result["out"] = out_path

        if os.path.isfile(out_path) and not args.overwrite:
            result["status"] = "skipped"
            result["message"] = "output exists (use --overwrite)"
            return result

        csv_path = find_csv(run_dir)
        xyz_path = os.path.join(run_dir, XYZ_NAME)

        # ---- load CSV ---------------------------------------------------
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]  # drop index col
        frame_col = detect_frame_column(df, args.frame_column)

        if args.columns:
            wanted = [c.strip() for c in args.columns.split(",") if c.strip()]
            missing = [c for c in wanted if c not in df.columns]
            if missing:
                raise RunError(
                    f"columns not in CSV: {missing}. Available: {list(df.columns)}"
                )
        else:
            wanted = [
                c
                for c in df.columns
                if c != frame_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        if not wanted:
            raise RunError("no columns selected to embed")

        keys = {c: sanitize(c) for c in wanted}
        if len(set(keys.values())) != len(keys):
            raise RunError(f"column names collide after sanitizing: {keys}")
        clash = set(keys.values()) & set(CONST_PARAMS)
        if clash:
            raise RunError(
                f"CSV columns clash with input_values constants: {sorted(clash)}"
            )

        # frame value -> row index
        if frame_col is not None:
            frame_vals = pd.to_numeric(df[frame_col], errors="coerce")
            row_of_frame = {int(v): i for i, v in enumerate(frame_vals) if pd.notna(v)}
            base = min(row_of_frame) if row_of_frame else 0
        else:
            row_of_frame = None
            base = 0

        # ---- pass 1: geometry -------------------------------------------
        n_frames, has_species, n_coords, n_extra, sample, lo, hi = scan_geometry(
            xyz_path, args.layout
        )
        result["frames"] = n_frames
        lattice = (
            "" if args.no_lattice else build_lattice(lo, hi, args.pad) + ' pbc="F F F"'
        )

        props = "species:S:1:pos:R:3" if has_species else "pos:R:3"
        if n_extra > 0:
            props += f":extra:R:{n_extra}"
        if args.ids:
            # Trailing 1..N column -> OVITO 'Particle Identifier'. Makes the
            # position along the chain explicit, and lets Load Trajectory match
            # particles to the topology by ID instead of by storage order.
            props += ":id:I:1"

        # constants, formatted once and reused on every frame
        const_parts = [
            f"{k}={float(params[k]):.{args.precision}f}" for k in CONST_PARAMS
        ]

        if n_frames != len(df):
            result["details"].append(
                f"frame count mismatch: {n_frames} in XYZ, {len(df)} in CSV"
            )
        if hi[0] - lo[0] == 0 or hi[1] - lo[1] == 0:
            result["details"].append(
                "an in-plane axis has zero extent -- columns are probably misread; "
                "try --layout species3"
            )

        # The highlight expressions use the anchor value as a particle id, so a
        # group that runs off the end of the chain would silently come up short
        # in OVITO rather than error. Catch it here instead.
        if args.highlights:
            n_beads = atom_count(xyz_path)
            for h in HIGHLIGHTS:
                if h.anchor not in df.columns:
                    result["details"].append(
                        f"highlight '{h.name}': no '{h.anchor}' column in the CSV"
                    )
                    continue
                vals = pd.to_numeric(df[h.anchor], errors="coerce").dropna()
                if vals.empty:
                    continue
                if vals.min() < 1 or vals.max() + h.span - 1 > n_beads:
                    result["details"].append(
                        f"highlight '{h.name}': {h.anchor} spans "
                        f"[{int(vals.min())}, {int(vals.max())}] + {h.span - 1} bead(s), "
                        f"outside ids 1..{n_beads}"
                    )

        # ---- pass 2: rewrite --------------------------------------------
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        missing_rows = 0
        try:
            with open(out_path, "w") as out:
                for i, (natoms, old_comment, atoms) in enumerate(iter_frames(xyz_path)):
                    if row_of_frame is not None:
                        ridx = row_of_frame.get(i + base)
                    else:
                        ridx = i if i < len(df) else None
                    if ridx is None:
                        missing_rows += 1

                    parts = []
                    if lattice:
                        parts.append(lattice)
                    parts.append(f"Properties={props}")
                    parts.append(f"Frame={i}")
                    if frame_col is not None and ridx is not None:
                        parts.append(
                            f"{sanitize(frame_col)}={int(df[frame_col].iloc[ridx])}"
                        )
                    parts.extend(const_parts)

                    if ridx is not None:
                        for col in wanted:
                            key = keys[col]
                            txt = format_value(df[col].iloc[ridx], args.precision)
                            if txt is not None:
                                parts.append(f"{key}={txt}")
                            if args.string_attrs:
                                raw = df[col].iloc[ridx]
                                if isinstance(raw, (float, np.floating)):
                                    parts.append(
                                        f"{key}_txt=nan"
                                        if not np.isfinite(raw)
                                        else f"{key}_txt={float(raw):.{args.string_precision}f}"
                                    )

                    # preserve any pre-existing keys we did not generate ourselves
                    for tok in re.findall(
                        r'[A-Za-z_][\w.]*=(?:"[^"]*"|\S+)', old_comment or ""
                    ):
                        k = tok.split("=", 1)[0]
                        if k not in ("Lattice", "Properties", "pbc") and not any(
                            pp.startswith(k + "=") for pp in parts
                        ):
                            parts.append(tok)

                    out.write(f"{natoms}\n")
                    out.write(" ".join(parts) + "\n")
                    for k, line in enumerate(atoms, start=1):
                        bead = f" {k}" if args.ids else ""
                        if n_coords == 3:
                            out.write(line + bead + "\n")
                        else:
                            tok = line.split()
                            s = 1 if has_species else 0
                            head = tok[:s]  # species, if any
                            coords = tok[s : s + 2] + ["0.0"]  # x y -> x y 0.0
                            tail = tok[s + 2 :]  # any extra columns
                            out.write(" ".join(head + coords + tail) + bead + "\n")
        except OSError as exc:
            # A half-written trajectory is worse than none: drop it either way.
            _unlink(out_path)
            if exc.errno == errno.ENOSPC:
                raise DiskFull(out_path)
            raise
        except BaseException:
            _unlink(out_path)
            raise

        if missing_rows:
            result["details"].append(
                f"{missing_rows} frame(s) had no matching CSV row; "
                "their attributes were omitted"
            )

        result["attributes"] = (
            ["Frame"]
            + ([sanitize(frame_col)] if frame_col else [])
            + list(CONST_PARAMS)
            + [keys[c] for c in wanted]
        )
        result["sample"] = sample
        result["bbox"] = (lo, hi)
        result["has_species"] = has_species
        result["n_coords"] = n_coords
        result["n_extra"] = n_extra
        result["csv"] = csv_path
        result["params"] = params
    except DiskFull:
        raise
    except RunError as exc:
        result["status"] = "failed"
        result["message"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- one bad run must not stop 1400
        result["status"] = "failed"
        result["message"] = f"{type(exc).__name__}: {exc}"
    return result


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _worker(run_dir: str, args):
    try:
        return annotate_run(run_dir, args)
    except DiskFull as exc:
        return {
            "run": run_dir,
            "status": "diskfull",
            "message": str(exc),
            "out": None,
            "frames": 0,
            "details": [],
        }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024 or unit == "TB":
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024


def print_details(result: dict, root: str) -> None:
    """The old single-file PARSE CHECK, for --verbose."""
    rel = os.path.relpath(result["run"], root)
    print(f"\n  --- {rel} ---")
    print(f"  CSV              : {os.path.basename(result['csv'])}")
    p = result["params"]
    print(
        f"  constants        : activity={p['activity']:g} bending={p['bending']:g} "
        f"ChiralityAngle={p['ChiralityAngle']:.6f} deg"
    )
    sample = result["sample"]
    start = 1 if result["has_species"] else 0
    print(f"  sample atom line : {' '.join(sample)}")
    if result["has_species"]:
        print(f"  species / type   : {sample[0]}")
    for k, axis in enumerate("xyz"[: result["n_coords"]]):
        print(f"  {axis}                : {sample[start + k]}")
    if result["n_coords"] == 2:
        print("  z                : 0.0  (padded)")
    if result["n_extra"]:
        print(f"  extra columns    : {result['n_extra']} (passed through)")
    lo, hi = result["bbox"]
    print(
        f"  bounding box     : x [{lo[0]:.3f}, {hi[0]:.3f}]  "
        f"y [{lo[1]:.3f}, {hi[1]:.3f}]  z [{lo[2]:.3f}, {hi[2]:.3f}]"
    )
    print(
        f"  OVITO attributes : {', '.join('[' + a + ']' for a in result['attributes'])}"
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch-embed per-frame CSV values and run constants into every "
        "config0.xyz under a data root, as OVITO global attributes."
    )
    p.add_argument(
        "--root",
        default=DATA_ROOT,
        help="data root holding <parameter dir>/<run dir> (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        default=OUTPUT_DIR,
        help="single directory receiving every annotated .xyz (default: %(default)s)",
    )
    p.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="PATTERN",
        help="only runs whose '<param dir>/<run dir>' matches this glob "
        "(repeatable), e.g. --only 'F05_*' --only '*/RUN_0001'",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="process at most N pending runs"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite outputs that already exist (default: skip them)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="list what would be written, then stop"
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel worker processes (default: %(default)s)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="start even if the free-space estimate says it will not fit",
    )
    p.add_argument(
        "--verbose", action="store_true", help="print the per-run parse check"
    )
    p.add_argument(
        "--columns",
        default=COLUMNS if isinstance(COLUMNS, str) else None,
        help="comma-separated CSV columns to embed (default: all numeric)",
    )
    p.add_argument(
        "--frame-column",
        default=FRAME_COLUMN,
        help="name of the frame-index column in the CSV",
    )
    p.add_argument(
        "--precision",
        type=int,
        default=6,
        help="decimal places written for floats (default: 6)",
    )
    p.add_argument(
        "--string-attrs",
        action="store_true",
        help="also emit pre-rounded string copies (Rg_txt=4.72) so you can "
        "control precision per value in OVITO, since the Text label "
        "layer applies one numeric format string to all placeholders",
    )
    p.add_argument(
        "--string-precision",
        type=int,
        default=2,
        help="decimals for the _txt string copies (default: 2)",
    )
    p.add_argument(
        "--layout",
        default="auto",
        choices=["auto", "species3", "species2", "plain3", "plain2"],
        help="atom-line column layout. species3 = <type> x y z (the usual "
        "case, including numeric type IDs); species2 = <type> x y; "
        "plain3 = x y z; plain2 = x y. Default: auto-detect.",
    )
    p.add_argument(
        "--no-lattice",
        action="store_true",
        help="do not write a Lattice= key (OVITO will auto-fit a box)",
    )
    p.add_argument(
        "--pad",
        type=float,
        default=0.05,
        help="fractional padding on the generated cell (default: 0.05)",
    )
    p.add_argument(
        "--no-ids",
        dest="ids",
        action="store_false",
        help="do not append the 1..N bead-index column (OVITO Particle Identifier)",
    )
    p.add_argument(
        "--no-topology",
        dest="topology",
        action="store_false",
        help="do not write the topology_N<n>.data chain-connectivity file(s)",
    )
    p.add_argument(
        "--no-highlights",
        dest="highlights",
        action="store_false",
        help="skip the monomer-highlight guide and its range checks",
    )
    p.add_argument(
        "--guide-only",
        action="store_true",
        help="rewrite ovito_setup.txt from the current HIGHLIGHTS list and stop; "
        "annotates nothing, so it is instant even with the disk unplugged",
    )
    args = p.parse_args()

    args.root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(args.root):
        sys.exit(f"ERROR: data root not found: {args.root}")

    args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    # Create the leaf if needed, but never a whole missing tree: on macOS an
    # unmounted /Volumes/DISK would silently become a directory on the boot
    # drive, which is exactly the disk we are trying to stay off.
    if not os.path.isdir(args.out_dir):
        parent = os.path.dirname(args.out_dir)
        if not os.path.isdir(parent):
            sys.exit(
                f"ERROR: output directory not reachable: {args.out_dir}\n"
                f"  '{parent}' does not exist -- is the disk plugged in?\n"
                f"  Edit the OUTPUT_DIR line at the top of this script, "
                f"or pass --out-dir PATH."
            )
        if not args.dry_run:
            os.makedirs(args.out_dir, exist_ok=True)

    if args.guide_only:
        # Take the chain length from whatever topology files are already there,
        # so this works without touching the (possibly remote) data root.
        lengths = set()
        for p_ in glob.glob(os.path.join(args.out_dir, "topology_N*.data")):
            m_ = re.search(r"topology_N(\d+)\.data$", p_)
            if m_:
                lengths.add(int(m_.group(1)))
        print(f"Wrote    : {write_highlight_guide(args.out_dir, lengths)}")
        for h in HIGHLIGHTS:
            print(f"  {h.name}: {highlight_expression(h)}")
        return

    # ---- discover -------------------------------------------------------
    runs = discover_runs(args.root)
    if not runs:
        sys.exit(f"ERROR: no run directory under {args.root} contains {XYZ_NAME}")

    if args.only:
        runs = [
            r
            for r in runs
            if any(
                fnmatch.fnmatch(os.path.relpath(r, args.root), pat) for pat in args.only
            )
        ]
        if not runs:
            sys.exit(f"ERROR: no run matches {args.only}")

    print(f"Root     : {args.root}")
    print(f"Output   : {args.out_dir}")
    print(f"Runs     : {len(runs)} with a {XYZ_NAME}")

    # ---- pending / pre-flight -------------------------------------------
    pending, already, unresolved = [], 0, []
    claimed: dict[str, str] = {}
    for run_dir in runs:
        try:
            out = output_path(run_dir, run_parameters(run_dir), args)
        except RunError as exc:
            unresolved.append((run_dir, str(exc)))
            continue
        # One flat directory means two runs could map onto one filename.
        if out in claimed:
            unresolved.append(
                (
                    run_dir,
                    f"output name collides with "
                    f"{os.path.relpath(claimed[out], args.root)}: "
                    f"{os.path.basename(out)}",
                )
            )
            continue
        claimed[out] = run_dir
        if os.path.isfile(out) and not args.overwrite:
            already += 1
        else:
            pending.append((run_dir, out))

    if args.limit is not None:
        pending = pending[: args.limit]

    est = (
        sum(os.path.getsize(os.path.join(r, XYZ_NAME)) for r, _ in pending)
        * SIZE_GROWTH
    )
    probe = args.out_dir
    while not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    free = shutil.disk_usage(probe).free

    print(
        f"Pending  : {len(pending)}"
        + (f"  ({already} already annotated, skipped)" if already else "")
        + (f"  ({len(unresolved)} unreadable)" if unresolved else "")
    )
    print(f"Estimate : {human(est)} of output, {human(free)} free on {probe}")

    # One topology file per distinct chain length covers every run of that
    # length -- the connectivity 1-2-3-...-N never changes within a run or
    # between runs.
    chain_types: dict[int, int] = {}  # chain length -> particle type to declare
    if args.topology:
        # When everything is already annotated there is still a topology file to
        # (re)write if it went missing, so fall back to scanning every run.
        for run_dir in [r for r, _ in pending] or runs:
            xyz = os.path.join(run_dir, XYZ_NAME)
            try:
                n_ = atom_count(xyz)
                if n_ not in chain_types:
                    chain_types[n_] = first_species(xyz)
            except (RunError, OSError):
                pass  # the run itself will report the problem when it fails
        if chain_types:
            print(
                f"Topology : {len(chain_types)} chain length(s): "
                + ", ".join(f"N={n} (type {t})" for n, t in sorted(chain_types.items()))
            )
    chain_lengths = set(chain_types)

    if args.dry_run:
        for run_dir, out in pending:
            print(
                f"  {os.path.relpath(run_dir, args.root)}  ->  {os.path.basename(out)}"
            )
        for n in sorted(chain_lengths):
            print(f"  (topology)  ->  topology_N{n}.data  [{n} beads, {n - 1} bonds]")
        for run_dir, msg in unresolved:
            print(f"  SKIP {os.path.relpath(run_dir, args.root)}: {msg}")
        return

    if not pending:
        for n in sorted(chain_lengths):
            if not os.path.isfile(os.path.join(args.out_dir, f"topology_N{n}.data")):
                print(
                    f"Wrote    : {os.path.basename(write_topology(n, args.out_dir, atom_type=chain_types[n]))} "
                    f"({n} beads, {n - 1} bonds)"
                )
        if args.highlights:
            print(
                f"Wrote    : "
                f"{os.path.basename(write_highlight_guide(args.out_dir, chain_lengths))}"
            )
        print("\nNothing to do.")
        return

    if est > free * 0.95 and not args.force:
        sys.exit(
            f"\nERROR: not enough free space -- need ~{human(est)}, have {human(free)}.\n"
            f"  Narrow the batch with --only / --limit, write elsewhere with\n"
            f"  --out-dir, free up space, or override with --force."
        )

    for n in sorted(chain_lengths):
        print(
            f"Wrote    : {os.path.basename(write_topology(n, args.out_dir, atom_type=chain_types[n]))} "
            f"({n} beads, {n - 1} bonds)"
        )
    if args.highlights:
        print(
            f"Wrote    : "
            f"{os.path.basename(write_highlight_guide(args.out_dir, chain_lengths))} "
            f"({len(HIGHLIGHTS)} highlight(s))"
        )

    # ---- run -------------------------------------------------------------
    print()
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failures, warnings = [], []
    width = len(str(len(pending)))
    run_dirs = [r for r, _ in pending]
    work = functools.partial(_worker, args=args)

    try:
        if args.jobs > 1 and len(run_dirs) > 1:
            with multiprocessing.Pool(args.jobs) as pool:
                results = pool.imap_unordered(work, run_dirs)
                _consume(
                    results, len(pending), width, args, counts, failures, warnings, pool
                )
        else:
            _consume(
                (work(r) for r in run_dirs),
                len(pending),
                width,
                args,
                counts,
                failures,
                warnings,
                None,
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")

    # ---- summary ---------------------------------------------------------
    print(
        f"\nDone: {counts['ok']} annotated, {counts['skipped']} skipped, "
        f"{counts['failed']} failed."
    )
    if warnings:
        print(f"\n{len(warnings)} run(s) with warnings:")
        for rel, msgs in warnings[:20]:
            print(f"  {rel}: {'; '.join(msgs)}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")
    if failures or unresolved:
        print(f"\n{len(failures) + len(unresolved)} run(s) could not be annotated:")
        for rel, msg in [
            (os.path.relpath(r, args.root), m) for r, m in unresolved
        ] + failures:
            print(f"  {rel}: {msg}")

    if counts["ok"]:
        print("\nOVITO Text label placeholders, e.g.:")
        print(
            "  F = [activity]<br>K = [bending]<br>theta = [ChiralityAngle] deg"
            "<br>Rg = [Rg]<br>Frame [Frame]"
        )
        if chain_lengths:
            n = sorted(chain_lengths)[0]
            print("\nTo draw the chain as a connected line:")
            print(f"  1. File > Load File ... topology_N{n}.data   (beads + bonds)")
            print("     It shows a straight row of beads in a huge box -- expected;")
            print("     both are placeholders that the next step replaces.")
            print("  2. Add modifier > Load trajectory ... trajectory_F*.xyz")
            print("     Beads are matched to the topology by their id column, so the")
            print("     bonds follow the chain however it folds.")
            print("  3. Bonds visual element > set the display width")
        if args.highlights:
            print("\nTo highlight monomers, add per highlight:")
            print("  Expression selection  +  Assign color   (untick both to disable)")
            for h in HIGHLIGHTS:
                r, g, b = h.color
                print(
                    f"  {h.name:<14} RGB({round(r * 255)},{round(g * 255)},"
                    f"{round(b * 255)})  {highlight_expression(h)}"
                )
            print("  Then one Clear selection at the top of the pipeline.")
            print("  Full instructions: ovito_setup.txt (written next to the data)")


def _consume(results, total, width, args, counts, failures, warnings, pool) -> None:
    """Print progress as results arrive; abort the batch if the disk fills."""
    for n, res in enumerate(results, 1):
        rel = os.path.relpath(res["run"], args.root)
        if res["status"] == "diskfull":
            print(f"\nERROR: disk full while writing {res['message']} -- stopping.")
            if pool is not None:
                pool.terminate()
            raise SystemExit(1)
        counts[res["status"]] += 1
        if res["status"] == "ok":
            print(
                f"[{n:>{width}}/{total}] {rel}  ->  "
                f"{os.path.basename(res['out'])}  ({res['frames']} frames)"
            )
            if args.verbose:
                print_details(res, args.root)
            if res["details"]:
                warnings.append((rel, res["details"]))
                for msg in res["details"]:
                    print(f"          WARNING: {msg}")
        elif res["status"] == "skipped":
            print(f"[{n:>{width}}/{total}] {rel}  --  skipped: {res['message']}")
        else:
            failures.append((rel, res["message"]))
            print(f"[{n:>{width}}/{total}] {rel}  !!  FAILED: {res['message']}")


if __name__ == "__main__":
    main()
