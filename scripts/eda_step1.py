"""Шаг 1: разведка данных. Пишет eda_output.txt в UTF-8 (эмодзи в консоли Windows ломают вывод)."""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eda_output.txt"

train = pd.read_parquet(ROOT / "dataset/train.parquet")
queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet")

L = []


def log(s=""):
    L.append(s)


log("=== SHAPES ===")
log(f"train {train.shape} | queries {queries.shape} | items {items.shape}")
log("")
log("=== train: сколько релевантных на один (query-text, location, category)? ===")
# Группируем по точному тексту запроса + локация + категория как прокси query
g = train.groupby(["search_query", "search_location_id", "search_category"])[
    "item_id"
].agg(["count", "nunique"])
log(str(g["count"].describe()))
log("топ-5 запросов по числу выборов:")
log(str(train["search_query"].value_counts().head(5)))
log("")
log("=== search_category (train vs benchmark) ===")
log("train:\n" + str(train["search_category"].value_counts()))
log("bench:\n" + str(queries["search_category"].value_counts()))
log("")
log("=== item_category_id (train vs corpus) ===")
log("train:\n" + str(train["item_category_id"].value_counts().head(10)))
log("corpus:\n" + str(items["item_category_id"].value_counts().head(10)))
log("")
log("=== microcat: сколько всего ===")
log(
    f"train microcat nunique={train['item_microcat_id'].nunique()} corpus={items['item_microcat_id'].nunique()}"
)
log("")
log("=== location: совпадение train item_location vs search_location ===")
log(
    f"train match rate={(train['item_location_id'] == train['search_location_id']).mean():.3f}"
)
log(
    f"bench search_location nunique={queries['search_location_id'].nunique()} corpus item_location nunique={items['item_location_id'].nunique()}"
)
log("")
log("=== item_id из train: сколько есть в корпусе benchmark_items? ===")
corpus_ids = set(items["item_id"].astype(str))
train_ids = set(train["item_id"].astype(str))
log(
    f"train unique items={len(train_ids)} overlap={len(train_ids & corpus_ids)} rate={len(train_ids & corpus_ids)/len(train_ids):.3f}"
)
log("")
log("=== запросы benchmark: есть ли такой же текст в train? (точное совпадение) ===")
train_q = set(train["search_query"].astype(str))
bench_q = queries["search_query"].astype(str)
log(f"bench exact-text in train: {(bench_q.isin(train_q)).sum()}/{len(bench_q)}")
# нормализованные (lower+strip)
train_qn = set(train["search_query"].astype(str).str.lower().str.strip())
log(
    f"bench normalized in train: {(bench_q.str.lower().str.strip().isin(train_qn)).sum()}/{len(bench_q)}"
)
log("")
log("=== примеры search_query (bench, первые 20) ===")
for q in queries["search_query"].head(20):
    log(f"  - {q!r}")
log("")
log("=== примеры пар train (первые 5) ===")
for _, r in train.head(5).iterrows():
    log(
        f"  Q={r['search_query']!r} loc={r['search_location_id']} cat={r['search_category']} filt={str(r['search_infm_params_text'])[:60]!r}"
    )
    log(
        f"    -> {r['item_title_raw']!r} micro={r['item_microcat_id']} icat={r['item_category_id']}"
    )
log("")
log("=== длины текстов ===")
log(f"query len chars: {queries['search_query'].str.len().describe().to_dict()}")
log(f"title len chars: {items['item_title_raw'].str.len().describe().to_dict()}")
log(f"descr len chars: {items['item_description_raw'].str.len().describe().to_dict()}")
log(
    f"descr empty rate corpus={(items['item_description_raw'].astype(str).str.strip() == '').mean():.3f}"
)
log("")
log("=== item_price (object! примеры) ===")
log(str(items["item_price"].head(10).tolist()))
log(str(train["item_price"].head(10).tolist()))
log("")
log("=== пропуски ===")
log("items null:\n" + str(items.isna().sum()[items.isna().sum() > 0]))
log("train null:\n" + str(train.isna().sum()[train.isna().sum() > 0]))
log("")
log("=== delivery flag ===")
log("train:\n" + str(train["search_is_delivery_search"].value_counts()))
log("bench:\n" + str(queries["search_is_delivery_search"].value_counts()))

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT} lines={len(L)}")
