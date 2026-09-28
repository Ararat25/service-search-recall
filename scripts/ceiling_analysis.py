"""Анализ потолка recall: где теряем 0.49 на valid (seed=42, 500 групп).
Считает: лексический потолок (общие стемы запрос/документ), гео-потолок,
долю релевантных в BM25-пуле top200 (проблема ранжирования vs поиска),
recall по частотности запроса (head/tail) и длине запроса.
Пишет ceiling_report.txt (UTF-8).
"""

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import normalize_text

TOK = re.compile(r"[а-яa-z0-9]+", re.IGNORECASE)
SEED, N_VALID, POOL = 42, 500, 200
OUT = ROOT / "ceiling_report.txt"


def main():
    import bm25s
    from nltk.stem.snowball import RussianStemmer

    stem = RussianStemmer()
    rng = np.random.RandomState(SEED)
    train = pd.read_parquet(ROOT / "dataset/train.parquet")
    items = pd.read_parquet(
        ROOT / "dataset/benchmark_items.parquet",
        columns=[
            "item_id",
            "item_title_raw",
            "item_description_raw",
            "item_infm_params_text",
            "item_location_id",
        ],
    )
    train["item_id"] = train["item_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)
    gk = ["search_query", "search_location_id", "search_category"]
    keys = list(train.groupby(gk).indices.keys())
    vkeys = set(keys[i] for i in rng.choice(len(keys), N_VALID, replace=False))
    is_v = np.asarray(train.set_index(gk).index.map(lambda k: k in vkeys))
    valid = train[is_v].copy()
    gt = valid.groupby(gk)["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
    qframe = valid.drop_duplicates(gk).reset_index(drop=True)

    need = set(valid["item_id"]) - set(items["item_id"])
    extra = valid[valid["item_id"].isin(need)].drop_duplicates("item_id")
    cols = [
        "item_id",
        "item_title_raw",
        "item_description_raw",
        "item_infm_params_text",
        "item_location_id",
    ]
    corpus = pd.concat([items[cols], extra[cols]], ignore_index=True).drop_duplicates(
        "item_id"
    )
    doc_by_id = {str(r["item_id"]): r for _, r in corpus.iterrows()}

    # стемы запроса и документов (title отдельно и все поля)
    L = []
    n_rel = n_title_hit = n_all_hit = n_geo = n_geo_title = 0
    freq = train["search_query"].value_counts().to_dict()
    rows = []
    for _, r in qframe.iterrows():
        key = (r["search_query"], r["search_location_id"], r["search_category"])
        qt = set(
            stem.stem(w) for w in TOK.findall(normalize_text(r["search_query"]).lower())
        )
        for rel in gt[key]:
            d = doc_by_id.get(str(rel))
            if d is None:
                continue
            n_rel += 1
            tt = set(
                stem.stem(w)
                for w in TOK.findall(normalize_text(str(d["item_title_raw"])).lower())
            )
            ta = tt | set(
                stem.stem(w)
                for w in TOK.findall(
                    normalize_text(
                        f"{d['item_description_raw']} {d['item_infm_params_text']}"
                    ).lower()
                )
            )
            th, ah = len(qt & tt) > 0, len(qt & ta) > 0
            g = d["item_location_id"] == r["search_location_id"]
            n_title_hit += th
            n_all_hit += ah
            n_geo += g
            n_geo_title += g and th
            rows.append(
                {
                    "q": r["search_query"],
                    "freq": freq.get(r["search_query"], 0),
                    "qlen": len(qt),
                    "title_hit": th,
                    "all_hit": ah,
                    "geo": g,
                    "rel_title": str(d["item_title_raw"])[:70],
                }
            )
    df = pd.DataFrame(rows)
    L.append(f"релевантных пар: {n_rel}")
    L.append(f"потолок title-лексики (>=1 общий стем): {n_title_hit/n_rel:.3f}")
    L.append(f"потолок all-fields лексики: {n_all_hit/n_rel:.3f}")
    L.append(f"потолок гео (same loc): {n_geo/n_rel:.3f}")
    L.append(f"geo AND title: {n_geo_title/n_rel:.3f}")
    L.append(
        f"НЕ пересекаются по стемам вообще (нужен dense/расширение): {(~df['all_hit']).mean():.3f}"
    )
    L.append("")
    L.append(
        "== recall-верхняя оценка для чистой лексики: "
        f"{df['all_hit'].mean():.3f} (все что без пересечения — мимо BM25) =="
    )
    L.append("")
    L.append("== по частотности запроса в train ==")
    df["band"] = pd.cut(
        df["freq"], [0, 1, 5, 50, 10**9], labels=["1", "2-5", "6-50", "50+"]
    )
    L.append(
        str(
            df.groupby("band", observed=True)[["title_hit", "all_hit", "geo"]]
            .mean()
            .round(3)
        )
    )
    L.append("")
    L.append("== по длине запроса (число стемов) ==")
    df["ql"] = pd.cut(df["qlen"], [0, 1, 2, 4, 100], labels=["1", "2", "3-4", "5+"])
    L.append(
        str(df.groupby("ql", observed=True)[["title_hit", "all_hit"]].mean().round(3))
    )
    L.append("")
    L.append("== примеры релевантных БЕЗ пересечения (цель для dense) ==")
    for _, x in df[~df["all_hit"]].head(15).iterrows():
        L.append(f"  Q={x['q']!r} (freq={x['freq']}) -> {x['rel_title']!r}")
    L.append("")
    L.append(
        "== примеры релевантных С пересечением, но вне топа (цель для ранжирования/весов) =="
    )
    for _, x in df[df["all_hit"]].head(10).iterrows():
        L.append(f"  Q={x['q']!r} -> {x['rel_title']!r}")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
