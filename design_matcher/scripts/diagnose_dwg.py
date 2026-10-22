"""
Diagnostic script for inspecting DWG conversion to DXF:
- Entity counts by type (LINE, LWPOLYLINE, SPLINE, ARC, CIRCLE, INSERT, POINT, HATCH, TEXT...)
- Layer names and properties
- Header extents ($EXTMIN / $EXTMAX)
- Actual computed bounding box from visible modelspace entities
- POINT entities and marker inspection
"""

import sys
from pathlib import Path
from collections import Counter
import ezdxf
from ezdxf import path as ez_path

def diagnose_dxf_file(dxf_path: Path):
    print(f"\n=======================================================")
    print(f"DIAGNOSING DXF FILE: {dxf_path.name}")
    print(f"=======================================================")
    
    doc = ezdxf.readfile(str(dxf_path))
    header = doc.header
    extmin = header.get("$EXTMIN", None)
    extmax = header.get("$EXTMAX", None)
    insunits = header.get("$INSUNITS", None)
    print(f"Header $EXTMIN: {extmin}")
    print(f"Header $EXTMAX: {extmax}")
    print(f"Header $INSUNITS: {insunits}")
    
    print("\n--- LAYERS ---")
    for layer in doc.layers:
        print(f"  Layer: '{layer.dxf.name}', Color: {layer.dxf.color}, IsOff: {layer.is_off()}, IsFrozen: {layer.is_frozen()}")

    msp = doc.modelspace()
    all_ents = list(msp)
    print(f"\nTotal Top-Level Modelspace Entities: {len(all_ents)}")

    type_counts = Counter()
    nested_type_counts = Counter()
    point_entities = []
    
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    def inspect_entity(ent, depth=0):
        nonlocal min_x, min_y, max_x, max_y
        t = ent.dxftype()
        if depth == 0:
            type_counts[t] += 1
        else:
            nested_type_counts[t] += 1

        if t == "INSERT":
            try:
                for sub in ent.virtual_entities():
                    inspect_entity(sub, depth + 1)
            except Exception as e:
                print(f"    Warning exploding block {ent.dxf.name}: {e}")
            return

        if t == "POINT":
            pos = ent.dxf.location
            point_entities.append((pos.x, pos.y, getattr(ent.dxf, "layer", "0")))
            min_x = min(min_x, pos.x)
            min_y = min(min_y, pos.y)
            max_x = max(max_x, pos.x)
            max_y = max(max_y, pos.y)
            return

        if t in ("TEXT", "MTEXT"):
            ins = getattr(ent.dxf, "insert", (0.0, 0.0, 0.0))
            min_x = min(min_x, ins[0])
            min_y = min(min_y, ins[1])
            max_x = max(max_x, ins[0])
            max_y = max(max_y, ins[1])
            return

        try:
            p = ez_path.make_path(ent)
            if len(p) > 0:
                min_x = min(min_x, p.start.x)
                min_y = min(min_y, p.start.y)
                max_x = max(max_x, p.start.x)
                max_y = max(max_y, p.start.y)
                for cmd in p:
                    end = cmd.end
                    min_x = min(min_x, end.x)
                    min_y = min(min_y, end.y)
                    max_x = max(max_x, end.x)
                    max_y = max(max_y, end.y)
        except Exception:
            pass

    for ent in all_ents:
        inspect_entity(ent, depth=0)

    print("\n--- ENTITY COUNTS (TOP-LEVEL) ---")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:15s}: {c}")

    if nested_type_counts:
        print("\n--- NESTED ENTITY COUNTS (INSIDE BLOCKS) ---")
        for t, c in sorted(nested_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t:15s}: {c}")

    print("\n--- POINT ENTITIES FOUND ---")
    print(f"  Total POINT entities: {len(point_entities)}")
    for i, (px, py, plyr) in enumerate(point_entities[:10]):
        print(f"  Point #{i+1}: ({px:.4f}, {py:.4f}) on layer '{plyr}'")
    if len(point_entities) > 10:
        print(f"  ... and {len(point_entities) - 10} more.")

    print("\n--- COMPUTED BOUNDING BOX FROM REAL VISIBLE GEOMETRY ---")
    if min_x <= max_x and min_y <= max_y:
        bw = max_x - min_x
        bh = max_y - min_y
        print(f"  X-Range: [{min_x:.4f}, {max_x:.4f}] (Width:  {bw:.4f})")
        print(f"  Y-Range: [{min_y:.4f}, {max_y:.4f}] (Height: {bh:.4f})")
    else:
        print("  No visible geometry bounding box could be computed.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        diagnose_dxf_file(Path(sys.argv[1]))
    else:
        print("Usage: python diagnose_dwg.py <file.dxf>")
