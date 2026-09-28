"""Проверка конфига V4 на отложенной части обучения.

Делит обучение по группам "текст запроса, город, категория": 500 случайных
групп уходят в проверку, остальное остаётся обучением. Релевантные объявления
проверки докладываются в корпус (иначе их там нет), а словари истории строятся
только по обучающей части — иначе история подсказала бы ответы.
Считает полноту топ-50 для двух наборов: урезанный V3 и полный V4.

"""

import gc
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import recall_at_50

# Слова: русские и латинские буквы плюс цифры, "ё" приводим к "е" (см. norm).
TOK = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
SEED, N_VALID = 42, 500

# Наборы кандидатов у веток. V3 — урезанный набор для сравнения,
# V4 — полный (те же числа, что в финальном скрипте).
A_POOL = 1200
AE_POOL = 500
PRF_POOL = 400
B_POOL = 300
C_POOL = 600
CF_POOL = 400
D_POOL = 600
DS_POOL = 400
TOPN = 50


def norm(text):
    """Привести строку к нижнему регистру и убрать лишнее."""
    if not isinstance(text, str):
        return ""
    return text.lower().replace("ё", "е").strip()


def full_text(row):
    """Весь текст объявления, заголовок повторён трижды."""
    title = str(row.get("item_title_raw", "") or "")
    descr = str(row.get("item_description_raw", "") or "")
    params = str(row.get("item_infm_params_text", "") or "")
    return norm(" ".join([title] * 3 + [descr, params]))


def short_text_raw(row):
    """Короткий текст: заголовок плюс параметры, без описания."""
    title = str(row.get("item_title_raw", "") or "")
    params = str(row.get("item_infm_params_text", "") or "")
    return norm(f"{title} {params}")


def short_bm25_text(row):
    """Короткий текст для BM25, заголовок повторён дважды."""
    title = str(row.get("item_title_raw", "") or "")
    params = str(row.get("item_infm_params_text", "") or "")
    return norm(" ".join([title] * 2 + [params]))


def title_text(row):
    """Один заголовок."""
    return norm(str(row.get("item_title_raw", "") or ""))


def round_robin_union(lists):
    """Объединить списки по одному элементу из каждого по кругу, без повторов."""
    merged, seen, pos = [], set(), [0] * len(lists)
    while True:
        moved = False
        for li, lst in enumerate(lists):
            while pos[li] < len(lst) and lst[pos[li]] in seen:
                pos[li] += 1
            if pos[li] < len(lst):
                merged.append(lst[pos[li]])
                seen.add(lst[pos[li]])
                pos[li] += 1
                moved = True
        if not moved:
            break
    return merged


def main():
    import bm25s
    from nltk.stem.snowball import RussianStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer

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
            "item_microcat_id",
        ],
    )
    train["item_id"] = train["item_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    # Деление на обучение и проверку целыми группами, чтобы один и тот же
    # запрос с городом не оказался по обе стороны.
    group_keys = ["search_query", "search_location_id", "search_category"]
    all_keys = list(train.groupby(group_keys).indices.keys())
    valid_keys = set(
        all_keys[i] for i in rng.choice(len(all_keys), N_VALID, replace=False)
    )
    is_valid = np.asarray(
        train.set_index(group_keys).index.map(lambda k: k in valid_keys)
    )
    valid = train[is_valid].copy()
    truth = (
        valid.groupby(group_keys)["item_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    # Корпус проверки: бенчмарк плюс недостающие релевантные из проверки.
    cols = [
        "item_id",
        "item_title_raw",
        "item_description_raw",
        "item_infm_params_text",
        "item_location_id",
        "item_microcat_id",
    ]
    missing = set(valid["item_id"]) - set(items["item_id"])
    extra = valid[valid["item_id"].isin(missing)].drop_duplicates("item_id")
    extra = extra[[c for c in cols if c in extra.columns]]
    for col in cols:
        if col not in extra.columns:
            extra[col] = np.nan
    corpus = pd.concat([items[cols], extra[cols]], ignore_index=True).drop_duplicates(
        "item_id"
    )
    corpus_ids = set(corpus["item_id"].astype(str))
    city_of = dict(zip(corpus["item_id"].astype(str), corpus["item_location_id"]))
    subcat_of = dict(
        zip(corpus["item_id"].astype(str), corpus["item_microcat_id"].astype(str))
    )
    cids = corpus["item_id"].astype(str).tolist()
    titles = [title_text(row) for _, row in corpus.iterrows()]

    # История только из обучающей части — честно, без подсмотра в проверку.
    pool_train = train[~is_valid].copy()
    pool_train["nq"] = (
        pool_train["search_query"]
        .astype(str)
        .str.lower()
        .str.replace("ё", "е")
        .str.strip()
    )
    top_titles = (
        pool_train.groupby("nq")[["item_title_raw"]]
        .apply(
            lambda g: [
                t for t, _ in Counter(g["item_title_raw"].astype(str)).most_common(3)
            ]
        )
        .to_dict()
    )
    in_corpus = pool_train["item_id"].isin(corpus_ids)
    log_by_text = (
        pool_train[in_corpus]
        .groupby("nq")["item_id"]
        .apply(lambda s: s.value_counts().index.tolist())
        .to_dict()
    )
    log_by_text_city = (
        pool_train[in_corpus]
        .groupby(["nq", "search_location_id"])["item_id"]
        .apply(lambda s: s.value_counts().index.tolist())
        .to_dict()
    )
    uniq_queries = pool_train.drop_duplicates("nq")[["nq"]].reset_index(drop=True)
    uq_items = (
        pool_train[in_corpus]
        .groupby("nq")["item_id"]
        .apply(lambda s: s.value_counts().index.tolist())
        .to_dict()
    )
    vec_q = TfidfVectorizer(
        token_pattern=r"(?u)[а-яёa-z0-9]+",
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=200000,
        min_df=1,
    )
    query_matrix = vec_q.fit_transform(uniq_queries["nq"].tolist())
    uniq_list = uniq_queries["nq"].tolist()
    del uniq_queries
    gc.collect()
    del train, pool_train
    gc.collect()

    qframe = valid.drop_duplicates(group_keys).reset_index(drop=True)
    qtexts = [norm(q) for q in qframe["search_query"]]
    qcities = qframe["search_location_id"].tolist()

    # Дополнение запроса словами из истории (до 10 новых слов от 4 букв).
    qexp = []
    for q in qtexts:
        known_stems = {stem.stem(w) for w in TOK.findall(q.lower())}
        extra = []
        for title in top_titles.get(q.lower().strip(), []):
            for word in TOK.findall(norm(title).lower()):
                if (
                    len(word) >= 4
                    and stem.stem(word) not in known_stems
                    and word not in extra
                ):
                    extra.append(word)
                if len(extra) >= 10:
                    break
            if len(extra) >= 10:
                break
        qexp.append(q + (" " + " ".join(extra) if extra else ""))

    def stem_tokens(texts):
        """Слова текстов, приведённые к основам, с кэшем по словарю."""
        raw = [TOK.findall(t.lower()) for t in texts]
        vocab = set(w for tokens in raw for w in tokens)
        cache = {w: stem.stem(w) for w in vocab}
        return [[cache[w] for w in tokens] for tokens in raw]

    full_docs = [full_text(row) for _, row in corpus.iterrows()]
    short_docs = [short_bm25_text(row) for _, row in corpus.iterrows()]
    docs_full = stem_tokens(full_docs)
    docs_short = stem_tokens(short_docs)
    del full_docs, short_docs
    gc.collect()
    q_full = [
        [stem.stem(w) for w in TOK.findall(q.lower())] or ["__no_token__"]
        for q in qtexts
    ]
    q_exp = [
        [stem.stem(w) for w in TOK.findall(q.lower())] or ["__no_token__"] for q in qexp
    ]
    q_short = [
        [stem.stem(w) for w in TOK.findall(q.lower())] or ["__no_token__"]
        for q in qtexts
    ]
    index_full = bm25s.BM25()
    index_full.index(docs_full)
    del docs_full
    gc.collect()
    index_short = bm25s.BM25()
    index_short.index(docs_short)
    del docs_short
    gc.collect()
    res_full, _ = index_full.retrieve(q_full, k=A_POOL)
    res_exp, _ = index_full.retrieve(q_exp, k=AE_POOL)
    # Дополнение словами из трёх первых найденных заголовков.
    q_prf = []
    q_prf_texts = []
    for n in range(len(qtexts)):
        known_stems = {stem.stem(w) for w in TOK.findall(qtexts[n].lower())}
        extra = []
        for idx in res_full[n][:3]:
            for word in TOK.findall(titles[int(idx)].lower()):
                if (
                    len(word) >= 4
                    and stem.stem(word) not in known_stems
                    and word not in extra
                ):
                    extra.append(word)
                if len(extra) >= 8:
                    break
            if len(extra) >= 8:
                break
        extended = qtexts[n] + (" " + " ".join(extra) if extra else "")
        q_prf_texts.append(extended)
        q_prf.append(
            [stem.stem(w) for w in TOK.findall(extended.lower())] or ["__no_token__"]
        )
    res_prf, _ = index_full.retrieve(q_prf, k=PRF_POOL)
    del index_full
    gc.collect()
    expanded = sum(1 for a, b in zip(qtexts, q_prf_texts) if a != b)
    print(f"prf expanded {expanded}/{len(qtexts)}", flush=True)
    res_short, _ = index_short.retrieve(q_short, k=B_POOL)
    del index_short
    gc.collect()

    short_raw = [short_text_raw(row) for _, row in corpus.iterrows()]
    vec_words_short = TfidfVectorizer(
        token_pattern=r"(?u)[а-яёa-z0-9]+",
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=500000,
        min_df=1,
    )
    mat_words_short = vec_words_short.fit_transform(short_raw)
    del short_raw
    gc.collect()
    full_raw = [full_text(row) for _, row in corpus.iterrows()]
    vec_words_full = TfidfVectorizer(
        token_pattern=r"(?u)[а-яёa-z0-9]+",
        lowercase=True,
        ngram_range=(1, 1),
        sublinear_tf=True,
        max_features=400000,
        min_df=2,
    )
    mat_words_full = vec_words_full.fit_transform(full_raw)
    del full_raw
    gc.collect()
    title_raw = [title_text(row) for _, row in corpus.iterrows()]
    vec_chars_title = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        max_features=300000,
        min_df=1,
    )
    mat_chars_title = vec_chars_title.fit_transform(title_raw)
    del title_raw
    gc.collect()
    short_for_chars = [short_text_raw(row) for _, row in corpus.iterrows()]
    vec_chars_short = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        lowercase=True,
        sublinear_tf=True,
        max_features=200000,
        min_df=2,
    )
    mat_chars_short = vec_chars_short.fit_transform(short_for_chars)
    del short_for_chars
    gc.collect()
    print(
        f"tfidf {mat_words_short.shape} {mat_words_full.shape}"
        f" {mat_chars_title.shape} {mat_chars_short.shape}",
        flush=True,
    )

    def top_ids(matrix, vectorizer, query, k):
        """До k идентификаторов с ненулевым сходством, лучшие первыми."""
        qv = vectorizer.transform([query])
        scores = (matrix * qv.T).toarray().ravel()
        order = np.argpartition(scores, -k)[-k:]
        order = order[np.argsort(scores[order])[::-1]]
        return [cids[i] for i in order if scores[i] > 0]

    preds_v3, preds_v4 = {}, {}
    for n in range(len(qframe)):
        key = (
            qframe.loc[n, "search_query"],
            qframe.loc[n, "search_location_id"],
            qframe.loc[n, "search_category"],
        )
        u_full = [cids[i] for i in res_full[n]]
        u_exp = [cids[i] for i in res_exp[n]]
        u_prf = [cids[i] for i in res_prf[n]]
        u_short = [cids[i] for i in res_short[n]]
        u_words_short = top_ids(mat_words_short, vec_words_short, qtexts[n], C_POOL)
        u_words_full = top_ids(mat_words_full, vec_words_full, qtexts[n], CF_POOL)
        u_chars_title = top_ids(mat_chars_title, vec_chars_title, qtexts[n], D_POOL)
        u_chars_short = top_ids(mat_chars_short, vec_chars_short, qtexts[n], DS_POOL)

        key_nq = qtexts[n].lower().strip()
        exact = [
            i for i in log_by_text_city.get((key_nq, qcities[n]), []) if i in corpus_ids
        ][:TOPN]
        history = [
            i
            for i in log_by_text.get(key_nq, [])
            if i in corpus_ids and i not in set(exact)
        ]
        history = (
            [i for i in history if city_of.get(i) == qcities[n]]
            + [i for i in history if city_of.get(i) != qcities[n]]
        )[:TOPN]
        qv = vec_q.transform([key_nq])
        sim = (query_matrix * qv.T).toarray().ravel()
        cand = np.argpartition(sim, -3)[-3:]
        cand = cand[np.argsort(sim[cand])[::-1]]
        fuzzy = []
        for j in cand:
            if sim[j] <= 0.05 or uniq_list[j] == key_nq:
                continue
            for iid in uq_items.get(uniq_list[j], [])[:20]:
                if iid in corpus_ids and iid not in set(exact + history + fuzzy):
                    fuzzy.append(iid)
                if len(fuzzy) >= 30:
                    break
            if len(fuzzy) >= 30:
                break
        fuzzy = (
            [i for i in fuzzy if city_of.get(i) == qcities[n]]
            + [i for i in fuzzy if city_of.get(i) != qcities[n]]
        )[:30]
        head = (exact + history + fuzzy)[:TOPN]
        # Подкатегория для приоритета: самая частая среди кандидатов из истории.
        votes = Counter(
            [
                subcat_of.get(i, "")
                for i in head
                if subcat_of.get(i, "") not in ("", "nan")
            ]
        )
        best_subcat = votes.most_common(1)[0][0] if votes else None

        def geo_cut(union, head_ids):
            """История впереди, затем свой город (внутри — своя подкатегория)."""
            union = [i for i in union if i not in set(head_ids)]
            same_city = [i for i in union if city_of.get(i) == qcities[n]]
            other = [i for i in union if city_of.get(i) != qcities[n]]
            if best_subcat:
                same_city = [
                    i for i in same_city if subcat_of.get(i) == best_subcat
                ] + [i for i in same_city if subcat_of.get(i) != best_subcat]
            return (head_ids + same_city + other)[:TOPN]

        preds_v3[key] = geo_cut(
            round_robin_union(
                [u_full[:500], u_exp, u_short, u_words_short[:300], u_chars_title[:300]]
            ),
            (exact + history + fuzzy)[:TOPN],
        )
        full_union = round_robin_union(
            [
                u_full,
                u_exp,
                u_prf,
                u_short,
                u_words_short,
                u_words_full,
                u_chars_title,
                u_chars_short,
            ]
        )
        preds_v4[key] = geo_cut(full_union, head)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(qframe)}", flush=True)

    score_v3 = recall_at_50(preds_v3, truth)
    score_v4 = recall_at_50(preds_v4, truth)
    print(
        f"V3={score_v3:.4f}  V4(+prf+microcat+pools)={score_v4:.4f}  DELTA={score_v4 - score_v3:+.4f} (n={len(truth)})"
    )


if __name__ == "__main__":
    main()
