#!/usr/bin/env python
"""Expose results stored under legacy step-valued names under their ms names.

DynVision used to name temporal quantities in steps (``tsteps=20``, ``dsteps``,
``stim``, ``intro``, ``interval``, ``idle``); it now names them in milliseconds
(``tsim=40`` at ``dt=2``, ``tseq``, ``tstim``, ``tintro``, ``tinterval``,
``tidle``). Snakemake would therefore look for every existing model and test
result under a new path and recompute it. This script makes the existing files
reachable under the new names without renaming or rewriting anything:

* every file under a legacy-named model directory (``models/``) and test
  directory (``reports/.../{test_identifier}/``) gets a relative symlink at its
  new path, inside real (not linked) directories;
* each link carries its target's mtime, so Snakemake sees exactly the old
  graph's mtime relations under the new names and plans no reruns;
* hashed test directories (compressed mode) are recovered from the frozen
  legacy experiment config, translated, and re-hashed with the workflow's own
  function.

Why file-level links with mirrored mtimes (verified on Snakemake 9.1.11):

* Snakemake keys its metadata by the literal path, so a new name carries no
  recorded params/code/input and only mtime decides.
* For an output it reads the link's own mtime; for an input, the later of link
  and target. Plain ``ln -s`` links get the creation time and trigger reruns
  depending on creation order; mirrored mtimes do not.
* Before a job runs, Snakemake removes its outputs with ``os.remove``. On a
  file link that removes the link only. Through a directory link it would
  delete the real old file.

Not linked by default: stage-2 aggregates (``reports/.../test_data.csv`` next
to the test directories), which Snakemake regenerates cheaply in the new format
(``--include-aggregates`` links them, for mirrors without per-test files);
figures and logs. Not mappable, and reported: a legacy key next to a wildcard
``dt=*`` (a dt sweep at fixed steps has no single ms name), unknown hashes, and
files Snakemake marked incomplete.

Usage::

    python scripts/link_legacy_temporal_paths.py ROOT            # dry run
    python scripts/link_legacy_temporal_paths.py ROOT --apply    # create links
    python scripts/link_legacy_temporal_paths.py ROOT --undo     # remove them

``ROOT`` is the project working directory holding ``models/`` and ``reports/``.
Afterwards, ``snakemake -n`` on the new targets should plan no test_model job
for results that existed; see the summary for what was skipped.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import yaml

from dynvision.utils.timing import (
    format_ms,
    is_legacy_step_key,
    is_ms_name,
    translate_legacy_args,
)
from dynvision.workflow.snake_utils import args_product, test_spec_hash

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_CONFIG = REPO / "dynvision" / "configs" / "legacy" / "config_experiments_steps.yaml"
MANIFEST_NAME = ".legacy_temporal_links.tsv"

#: {name}{sep}{args}_{seed}: DyRCNNx8:tsteps=20+dt=2+..._7000 (older dirs use '_' as sep).
MODEL_IDENTIFIER = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9]*)(?P<sep>[:_])(?P<args>[^/]*=[^/]*?)_(?P<seed>[0-9.]+|\*)$"
)
HASH = re.compile(r"^[0-9a-f]{1,8}$")


class Unmappable(Exception):
    """A legacy name that has no single ms name."""


# --------------------------------------------------------------------------
# Name translation
# --------------------------------------------------------------------------


def _tokens(args: str) -> List[Tuple[str, str]]:
    return [tuple(token.split("=", 1)) for token in args.split("+") if "=" in token]


def _dt_of(tokens: List[Tuple[str, str]]) -> Optional[str]:
    return next((value for key, value in tokens if key == "dt"), None)


def translate_args(args: str, dt: Optional[str]) -> str:
    """Translate the legacy keys of a ``k=v+k=v`` string, keeping order.

    Raises:
        Unmappable: If a legacy key is present but ``dt`` is not a number.
    """
    tokens = _tokens(args)
    if not any(is_legacy_step_key(key) for key, _ in tokens):
        return args
    dt = _dt_of(tokens) or dt  # a spec's own dt overrides the model's
    try:
        dt_value = float(dt)
    except (TypeError, ValueError):
        raise Unmappable(f"legacy keys next to dt={dt!r}") from None
    try:
        translated = translate_legacy_args(dict(tokens), dt_value)
    except ValueError as error:
        raise Unmappable(str(error)) from None
    return "+".join(
        f"{key}={format_ms(value) if is_ms_name(key) else value}"
        for key, value in translated.items()
    )


def translate_model_identifier(component: str) -> Tuple[str, Optional[str]]:
    """Return (new component, its dt) for a model identifier directory name."""
    match = MODEL_IDENTIFIER.match(component)
    if not match:
        return component, None
    args = match["args"]
    dt = _dt_of(_tokens(args))
    new_args = translate_args(args, dt)
    new = f"{match['name']}{match['sep']}{new_args}_{match['seed']}"
    return new, dt


@dataclass
class LegacySpecs:
    """Hash -> (loader, spec) for every test spec a legacy config defines."""

    by_hash: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    @classmethod
    def from_configs(cls, paths: List[Path]) -> "LegacySpecs":
        specs = cls()
        for path in paths:
            experiments = yaml.safe_load(path.read_text())["experiment_config"]
            for entry in experiments.values():
                if not isinstance(entry, dict) or "data_loader" not in entry:
                    continue
                loader = entry["data_loader"]
                for spec in args_product(entry.get("data_args", {})):
                    specs.by_hash.setdefault(test_spec_hash(loader, spec), (loader, spec))
        return specs


def translate_test_identifier(component: str, dt: Optional[str], legacy: LegacySpecs) -> str:
    """Translate a hashed or a full-spec test directory name."""
    if HASH.match(component):
        if component not in legacy.by_hash:
            raise Unmappable(f"hash {component!r} is not in the legacy config(s)")
        loader, spec = legacy.by_hash[component]
        new_spec = ":" + translate_args(spec.lstrip(":"), dt)
        return test_spec_hash(loader, new_spec)
    if ":" in component and "=" in component:
        loader, args = component.split(":", 1)
        return f"{loader}:{translate_args(args, dt)}"
    return component


@dataclass
class Mapping:
    old: Path  # relative to ROOT
    new: Path  # relative to ROOT
    kind: str  # "model", "test", "aggregate"


def map_path(relative: Path, legacy: LegacySpecs) -> Optional[Mapping]:
    """Return the new name of a file, or None if its name has no legacy key.

    Raises:
        Unmappable: As the translate functions.
    """
    parts = list(relative.parts)
    if parts[0] == "models" and len(parts) >= 4:
        parts[2], _ = translate_model_identifier(parts[2])
        kind = "model"
    elif parts[0] == "reports" and len(parts) >= 5:
        parts[2], dt = translate_model_identifier(parts[2])
        if len(parts) >= 6:
            parts[4] = translate_test_identifier(parts[4], dt, legacy)
            kind = "test"
        else:
            kind = "aggregate"
    else:
        return None
    new = Path(*parts)
    return None if new == relative else Mapping(relative, new, kind)


# --------------------------------------------------------------------------
# Filesystem
# --------------------------------------------------------------------------


def incomplete_paths(root: Path) -> Set[str]:
    """Paths Snakemake marked incomplete, decoded from .snakemake/incomplete."""
    marker_dir = root / ".snakemake" / "incomplete"
    paths: Set[str] = set()
    if not marker_dir.is_dir():
        return paths
    for marker in marker_dir.rglob("*"):
        if marker.is_dir():
            continue
        # Long ids are split into '@'-prefixed directory chunks (persistence.py).
        b64 = "".join(part.lstrip("@") for part in marker.relative_to(marker_dir).parts)
        try:
            paths.add(base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).decode())
        except (ValueError, UnicodeDecodeError):
            continue
    return paths


def iter_files(root: Path) -> Iterator[Path]:
    for top in ("models", "reports"):
        base = root / top
        if not base.is_dir():
            continue
        for directory, _, files in os.walk(base, followlinks=False):
            for name in files:
                yield (Path(directory) / name).relative_to(root)


def mirror_mtime(link: Path) -> None:
    """Give a link its target's times, so Snakemake sees the old relations."""
    target = os.stat(link)  # follows the link
    os.utime(link, ns=(target.st_atime_ns, target.st_mtime_ns), follow_symlinks=False)


def plan(root: Path, legacy: LegacySpecs, include_aggregates: bool):
    """Classify every file under models/ and reports/."""
    incomplete = incomplete_paths(root)
    outcome: Dict[str, List] = {
        "link": [], "linked": [], "conflict": [], "unmappable": [],
        "incomplete": [], "aggregate": [],
    }
    own_links = {line.split("\t")[0] for line in read_manifest(root)}
    for relative in iter_files(root):
        if str(relative) in own_links:
            continue  # a link this script made earlier
        try:
            mapping = map_path(relative, legacy)
        except Unmappable as reason:
            outcome["unmappable"].append((relative, str(reason)))
            continue
        if mapping is None:
            continue
        if {str(relative), str(root / relative)} & incomplete:
            outcome["incomplete"].append(mapping)
            continue
        if mapping.kind == "aggregate" and not include_aggregates:
            outcome["aggregate"].append(mapping)
            continue
        new_path = root / mapping.new
        if new_path.is_symlink() and new_path.resolve() == (root / relative).resolve():
            outcome["linked"].append(mapping)
        elif os.path.lexists(new_path):
            outcome["conflict"].append(mapping)
        else:
            outcome["link"].append(mapping)
    return outcome


def read_manifest(root: Path) -> List[str]:
    manifest = root / MANIFEST_NAME
    if not manifest.exists():
        return []
    return [line for line in manifest.read_text().splitlines() if line.strip()]


def apply(root: Path, outcome) -> int:
    created = 0
    with open(root / MANIFEST_NAME, "a") as manifest:
        for mapping in outcome["link"]:
            new_path = root / mapping.new
            new_path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(root / mapping.old, new_path.parent), new_path)
            mirror_mtime(new_path)
            manifest.write(f"{mapping.new}\t{mapping.old}\n")
            created += 1
    for mapping in outcome["linked"]:
        mirror_mtime(root / mapping.new)  # the target may have changed since
    return created


def undo(root: Path) -> Tuple[int, int]:
    """Remove the links in the manifest that still point at their target."""
    removed = kept = 0
    directories = set()
    for line in read_manifest(root):
        new, old = line.split("\t")
        link = root / new
        if link.is_symlink() and link.resolve() == (root / old).resolve():
            link.unlink()
            removed += 1
            directories.add(link.parent)
        else:
            kept += 1
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        while directory != root and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            directory = directory.parent
    (root / MANIFEST_NAME).unlink(missing_ok=True)
    return removed, kept


def summarize(outcome, applied: Optional[int], limit: int) -> None:
    counts = Counter({key: len(value) for key, value in outcome.items()})
    action = "would create" if applied is None else "created"
    print(f"links {action}:        {applied if applied is not None else counts['link']}")
    print(f"already linked:        {counts['linked']}")
    print(f"conflicts (skipped):   {counts['conflict']}")
    print(f"aggregates (skipped):  {counts['aggregate']}  (--include-aggregates to link)")
    print(f"incomplete (skipped):  {counts['incomplete']}")
    print(f"unmappable (skipped):  {counts['unmappable']}")
    for key in ("link", "conflict", "unmappable", "incomplete"):
        for item in outcome[key][:limit]:
            if isinstance(item, Mapping):
                print(f"  [{key}] {item.old}\n      -> {item.new}")
            else:
                print(f"  [{key}] {item[0]}: {item[1]}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root", type=Path, help="project directory with models/ and reports/")
    parser.add_argument(
        "--legacy-config",
        type=Path,
        action="append",
        help="legacy experiment config(s) defining the hashed test specs "
        f"(default: {DEFAULT_LEGACY_CONFIG.relative_to(REPO)})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="create the links")
    mode.add_argument("--undo", action="store_true", help="remove the links in the manifest")
    parser.add_argument(
        "--include-aggregates",
        action="store_true",
        help="also link stage-2 test_data.csv aggregates",
    )
    parser.add_argument("--show", type=int, default=5, help="examples listed per category")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if args.undo:
        removed, kept = undo(root)
        print(f"removed {removed} link(s); left {kept} manifest entr(y/ies) that no longer matched")
        return 0

    legacy = LegacySpecs.from_configs(args.legacy_config or [DEFAULT_LEGACY_CONFIG])
    outcome = plan(root, legacy, args.include_aggregates)
    applied = apply(root, outcome) if args.apply else None
    summarize(outcome, applied, args.show)
    if not args.apply:
        print("\nDry run. Re-run with --apply to create the links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
