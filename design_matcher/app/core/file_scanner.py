"""
Core file scanning and unique label generation utilities for CAD drawings (DXF and DWG).
Provides pure, framework-independent functions for folder traversal, extension filtering,
natural sorting, and collision-free hierarchical display label generation.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def natural_sort_key(s: str) -> List[Any]:
    """
    Transforms string into a natural sort key so that numbers sort numerically
    ('1.dxf', '2.dxf', '10.dxf' rather than '1.dxf', '10.dxf', '2.dxf').
    Case-insensitive.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def scan_cad_files(
    root: Path | str,
    allowed_extensions: Iterable[str],
    recursive: bool = True,
) -> List[Path]:
    """
    Recursively (or non-recursively) scans the given root directory for files matching
    the allowed extensions (case-insensitively).

    Handles permission errors and broken symlinks gracefully without raising exceptions.
    Returns a list of resolved Path objects.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        return []

    # Normalize extensions to lowercase with leading dot (e.g. '.dxf', '.dwg')
    ext_set: Set[str] = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in allowed_extensions
        if ext
    }
    if not ext_set:
        return []

    results: List[Path] = []

    if not recursive:
        try:
            with os.scandir(root_path) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=True):
                            name_lower = entry.name.lower()
                            if any(name_lower.endswith(ext) for ext in ext_set):
                                results.append(Path(entry.path))
                    except (PermissionError, OSError):
                        continue
        except (PermissionError, OSError):
            return []
        return results

    # Recursive scan using os.walk with error suppression
    try:
        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
            # Sort dirnames and filenames for predictable traversal
            dirnames.sort(key=natural_sort_key)
            for fname in filenames:
                fname_lower = fname.lower()
                if any(fname_lower.endswith(ext) for ext in ext_set):
                    full_p = Path(dirpath) / fname
                    results.append(full_p)
    except (PermissionError, OSError):
        pass

    return results


def _get_path_label(path: Path, num_levels: int, root: Optional[Path] = None) -> str:
    """
    Generates a label for a path using the last `num_levels` parts.
    If `root` is provided and `path` is within `root`, levels are drawn from
    `path.relative_to(root)`. If fewer parts exist, returns all available parts.
    """
    if root is not None:
        try:
            rel = path.resolve().relative_to(root.resolve())
            parts = rel.parts
            count = min(num_levels, len(parts))
            return "/".join(parts[-count:])
        except ValueError:
            pass

    parts = path.parts
    count = min(num_levels, len(parts))
    return "/".join(parts[-count:])


def build_unique_display_labels(
    paths: List[Path],
    root: Optional[Path] = None,
) -> Dict[Path, str]:
    """
    Computes unique display names for a list of file paths according to the rules:
    1. If a file name (with extension) is unique across all results, show just 'filename.ext'.
    2. If the same file name exists in more than one folder, show 'parentFolder/filename.ext'.
    3. If immediate parent folder names are also identical, add more parent levels until
       each duplicate label is unique (e.g. 'x/a/1.dxf' vs 'y/a/1.dxf').

    '1.dxf' and '1.dwg' are distinct file names, so they do not conflict with each other.
    """
    if not paths:
        return {}

    # Map each Path to its filename (with extension).
    # Group paths by case-insensitive filename
    name_groups: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        name_groups[p.name.lower()].append(p)

    final_labels: Dict[Path, str] = {}
    colliding_paths: List[Path] = []

    # Pass 1: Files with unique filenames get f.name
    for name_lower, group in name_groups.items():
        if len(group) == 1:
            final_labels[group[0]] = group[0].name
        else:
            colliding_paths.extend(group)

    if not colliding_paths:
        return final_labels

    # Pass 2: For duplicate filenames, progressively add parent levels
    # Start with 2 levels: immediate_parent/filename
    levels: Dict[Path, int] = {p: 2 for p in colliding_paths}
    max_safety_iterations = 20

    for _ in range(max_safety_iterations):
        # Candidate label for each colliding path
        current_candidates: Dict[Path, str] = {
            p: _get_path_label(p, levels[p], root=root) for p in colliding_paths
        }

        # Check for collisions among candidates and existing unique labels
        label_usage: Dict[str, List[Path]] = defaultdict(list)
        for p, lbl in current_candidates.items():
            label_usage[lbl].append(p)

        # Also check collision against already-assigned unique labels
        assigned_label_set = {lbl for p, lbl in final_labels.items()}

        all_unique = True
        for lbl, owners in label_usage.items():
            # If label is shared by >1 owner or collides with an existing single-file label
            if len(owners) > 1 or lbl in assigned_label_set:
                all_unique = False
                for p in owners:
                    levels[p] += 1

        if all_unique:
            for p, lbl in current_candidates.items():
                final_labels[p] = lbl
            break
    else:
        # Fallback if deeply nested identical names: use full relative or resolved path
        for p in colliding_paths:
            if root is not None:
                try:
                    final_labels[p] = str(p.resolve().relative_to(root.resolve()))
                    continue
                except ValueError:
                    pass
            final_labels[p] = str(p)

    return final_labels


def build_tree_hierarchy(
    paths: List[Path],
    root: Path,
) -> Dict[str, Any]:
    """
    Builds a nested tree structure for matching files under root:
    {
        "name": root.name,
        "path": root,
        "is_dir": True,
        "children": {
            "subfolder_name": { ... },
            "filename.dxf": { "name": "filename.dxf", "path": Path, "is_dir": False }
        }
    }
    Folders containing zero matching files at any depth are excluded.
    """
    root_resolved = root.resolve()
    tree: Dict[str, Any] = {
        "name": root_resolved.name or str(root_resolved),
        "path": root_resolved,
        "is_dir": True,
        "children": {},
    }

    for p in paths:
        p_resolved = p.resolve()
        try:
            rel = p_resolved.relative_to(root_resolved)
        except ValueError:
            # File is outside root; attach directly to root
            tree["children"][p.name] = {
                "name": p.name,
                "path": p,
                "is_dir": False,
                "children": {},
            }
            continue

        parts = rel.parts
        curr = tree
        for i, part in enumerate(parts):
            is_leaf = (i == len(parts) - 1)
            if is_leaf:
                curr["children"][part] = {
                    "name": part,
                    "path": p,
                    "is_dir": False,
                    "children": {},
                }
            else:
                if part not in curr["children"]:
                    # Build directory path
                    sub_path = root_resolved.joinpath(*parts[: i + 1])
                    curr["children"][part] = {
                        "name": part,
                        "path": sub_path,
                        "is_dir": True,
                        "children": {},
                    }
                curr = curr["children"][part]

    return tree
