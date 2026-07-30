#!/usr/bin/env python
"""
Step 06 - Render the complete classification trees with counts at every level.

For each hierarchy, emits an indented text tree and a nested JSON where each node
carries n_proteins (or n_ligands) and n_complexes, so you can group / compare
model performance at any depth.

Protein hierarchies:  ChEMBL target class (l1..l6), EC (class/subclass/sub-sub)
Ligand hierarchy:     ClassyFire ChemOnt (kingdom/superclass/class/subclass)

Writes to output/trees/.
"""
import os, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
TREES = os.path.join(OUT, "trees"); os.makedirs(TREES, exist_ok=True)


def build_tree(df, level_cols, entity_col, count_label):
    """Nested dict tree; each node: _n_entities, _n_complexes, children."""
    root = {"_n": 0, "_complexes": 0, "children": {}}
    n_col = "n_complexes"
    for _, r in df.iterrows():
        path = [r[c] for c in level_cols if isinstance(r.get(c), str) and r.get(c).strip()]
        if not path:
            continue
        nc = int(r[n_col]) if pd.notna(r[n_col]) else 0
        node = root
        node["_n"] += 1; node["_complexes"] += nc
        for p in path:
            node = node["children"].setdefault(p, {"_n": 0, "_complexes": 0, "children": {}})
            node["_n"] += 1; node["_complexes"] += nc
    return root


def render(node, name, depth, lines, count_label):
    if depth >= 0:
        pad = "  " * depth
        lines.append(f"{pad}{name}  [{node['_n']} {count_label}, {node['_complexes']} complexes]")
    for child in sorted(node["children"], key=lambda k: -node["children"][k]["_complexes"]):
        render(node["children"][child], child, depth + 1, lines, count_label)


def emit(root, base, count_label, header):
    lines = [header, "=" * len(header)]
    render(root, "ALL", -1, lines, count_label)
    # top line
    lines.insert(2, f"ALL  [{root['_n']} {count_label}, {root['_complexes']} complexes]")
    open(os.path.join(TREES, base + ".txt"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(root, open(os.path.join(TREES, base + ".json"), "w"), indent=1)
    print(f"[06] wrote trees/{base}.txt / .json")


def do_proteins():
    if not os.path.exists(os.path.join(OUT, "protein_taxonomy.csv")):
        return
    t = pd.read_csv(os.path.join(OUT, "protein_taxonomy.csv"))
    emit(build_tree(t, [f"chembl_l{i}" for i in range(1, 7)], "protein_id", "proteins"),
         "protein_chembl_class", "proteins",
         "ChEMBL TARGET CLASS TREE  (protein-ligand complexes grouped by target class)")
    emit(build_tree(t, ["ec_class", "ec_subclass", "ec_sub_subclass", "ec_primary"], "protein_id", "proteins"),
         "protein_ec", "proteins", "ENZYME COMMISSION (EC) TREE")


def do_ligands():
    f = os.path.join(OUT, "ligand_taxonomy.csv")
    if not os.path.exists(f):
        print("[06] ligand_taxonomy.csv not present yet; skipping ligand tree")
        return
    l = pd.read_csv(f)
    emit(build_tree(l, ["kingdom", "superclass", "cf_class", "subclass", "direct_parent"],
                    "ligand_id", "ligands"),
         "ligand_classyfire", "ligands",
         "CLASSYFIRE / ChemOnt TREE  (ligands grouped by chemical class)")


if __name__ == "__main__":
    do_proteins()
    do_ligands()
