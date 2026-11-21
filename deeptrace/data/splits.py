"""Phase 1.3: leakage-safe splits.

Celeb-DF
  official : 518-video test list; train/val from the rest, grouped by identity
             (every identity a video touches stays on one side).
  strict   : identities assigned to train/val/test; a fake whose two identities
             land in different groups is dropped. YouTube-real has no identity
             label, so those videos are assigned at random (stated limitation).

FF++ c23
  No official split JSON ships with this copy. Source IDs are linked into
  connected components (original k + every fake using k as target or source,
  which also joins pairs like 000_003 / 003_000). Whole components go to
  train/val/test at 72/14/14, so no person crosses splits and nothing is dropped.
  DeepFakeDetection is excluded from training/testing: its real actor videos are
  not in this copy, so those scenes would only ever appear as fake (shortcut).
"""
import re
import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def _components(edges, nodes):
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {n: find(n) for n in nodes}


def _assign(groups, fracs, seed):
    g = sorted(set(groups))
    rng = np.random.RandomState(seed)
    rng.shuffle(g)
    n = len(g)
    cut1, cut2 = int(round(n * fracs[0])), int(round(n * (fracs[0] + fracs[1])))
    return {k: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, k in enumerate(g)}


# ---------------------------------------------------------------- FF++
def ffpp_split(inv, seed=0, fracs=(0.72, 0.14, 0.14)):
    ff = inv[(inv.dataset == "ffpp") & (inv.method != "DeepFakeDetection")].copy()
    ids = set(ff.id_a) | set(ff.id_b)
    comp = _components(zip(ff.id_a, ff.id_b), ids)
    role = _assign(set(comp.values()), (fracs[0], fracs[1]), seed)
    ff["split"] = [role[comp[a]] for a in ff.id_a]
    assert all(role[comp[a]] == role[comp[b]] for a, b in zip(ff.id_a, ff.id_b))
    ff["split_name"] = f"ffpp_s{seed}"
    return ff


# ---------------------------------------------------------------- Celeb-DF
def _cdf_ids(rel):
    return set(re.findall(r"id\d+", rel))


def celebdf_official(inv, seed=0, val_frac=0.15):
    """Official 518 test list kept as published (it shares identities with training,
    which is why the strict split exists). Train/val: identities assigned to one side;
    a fake whose two identities fall on different sides is dropped, so validation
    (used for checkpoint and threshold choice) shares no identity with training."""
    c = inv[inv.dataset == "celebdf"].copy()
    test = c.cdf_official_test.astype(bool)
    c.loc[test, "split"] = "test"
    dev = c[~test]
    ids = sorted({i for r in dev.rel for i in _cdf_ids(r)}, key=lambda s: int(s[2:]))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    role = {ids[k]: ("val" if j < int(round(len(ids) * val_frac)) else "train") for j, k in enumerate(perm)}
    def a(rel):
        t = _cdf_ids(rel)
        if not t:
            return "val" if rng.rand() < val_frac else "train"
        r = {role[x] for x in t}
        return r.pop() if len(r) == 1 else "drop"
    c.loc[dev.index, "split"] = [a(r) for r in dev.rel]
    c["split_name"] = f"celebdf_official_s{seed}"
    return c


def celebdf_strict(inv, f_test=0.25, f_val=0.15, max_seed=500):
    """Seeded identity assignment; first seed meeting minimum class counts is used."""
    c0 = inv[inv.dataset == "celebdf"].copy()
    ids = sorted({i for r in c0.rel for i in _cdf_ids(r)}, key=lambda s: int(s[2:]))
    for seed in range(max_seed):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(ids))
        n_te, n_va = int(len(ids) * f_test), int(len(ids) * f_val)
        role = {ids[k]: ("test" if j < n_te else "val" if j < n_te + n_va else "train")
                for j, k in enumerate(perm)}
        def a(rel):
            t = _cdf_ids(rel)
            if not t:
                return rng.choice(SPLITS, p=[1 - f_test - f_val, f_val, f_test])
            r = {role[x] for x in t}
            return r.pop() if len(r) == 1 else "drop"
        c = c0.copy(); c["split"] = [a(r) for r in c.rel]
        n = c[c.split != "drop"].groupby(["split", "label"]).size()
        need = {("test", 1): 300, ("val", 1): 100, ("train", 1): 1500,
                ("test", 0): 100, ("val", 0): 30, ("train", 0): 200}
        if all(n.get(k, 0) >= v for k, v in need.items()):
            c["split_name"] = f"celebdf_strict_s{seed}"
            return c
    raise RuntimeError("no strict seed found")


# ---------------------------------------------------------------- checks
def leakage_report(df, id_cols=("id_a", "id_b")):
    """Count identities that appear in more than one split (should be 0 for strict/ffpp)."""
    d = df[df.split.isin(SPLITS)]
    owner = {}
    for s in SPLITS:
        for col in id_cols:
            for i in d.loc[d.split == s, col]:
                if str(i).startswith("yt_"):
                    continue
                owner.setdefault(i, set()).add(s)
    return {f"{a}-{b}": sum(1 for v in owner.values() if a in v and b in v)
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}


def counts(df):
    return df.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "real", 1: "fake"})


def build_all(inv):
    out = {"ffpp": ffpp_split(inv), "celebdf_official": celebdf_official(inv),
           "celebdf_strict": celebdf_strict(inv)}
    for k, d in out.items():
        print(f"\n== {k} ({d.split_name.iloc[0]})")
        print(counts(d).to_string())
        if k == "ffpp":
            print(d[d.split != "drop"].groupby(["split", "method"]).size().unstack(fill_value=0).to_string())
        print("identity leakage:", leakage_report(d))
    return out
