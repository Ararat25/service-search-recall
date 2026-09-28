"""

Что делает скрипт: для каждого запроса из бенчмарка собирает восемь списков
кандидатов четырьмя видами поиска (BM25 по полному тексту, BM25 по короткому
тексту, TF-IDF по словам, TF-IDF по буквам), добавляет впереди кандидатов
из истории выборов и упорядочивает итог: сначала свой город, внутри него —
своя подкатегория. Обрезает до 50 и пишет файл ответа.

"""

import gc
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "answer_v4.csv"

# Слова запроса и объявлений: русские и латинские буквы плюс цифры.
# Буква "ё" тоже входит в диапазон, но мы всё равно приводим её к "е",
# чтобы одни и те же слова не разъезжались по разным токенам.
TOK = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Сколько кандидатов берёт каждая поисковая ветка до объединения.
# Берём с запасом: порядок внутри итоговых 50 метрикой не считается,
# важно только покрыть как можно больше подходящих объявлений.
A_POOL = 1200  # BM25 по полному тексту, исходный запрос
AE_POOL = 500  # BM25 по полному тексту, запрос с добавлением слов из истории
PRF_POOL = 400  # BM25 по полному тексту, запрос с добавлением слов из топ-3
B_POOL = 300  # BM25 по короткому тексту (заголовок не тонет в описании)
C_POOL = 600  # TF-IDF по словам на коротком тексте
CF_POOL = 400  # TF-IDF по словам на полном тексте
D_POOL = 600  # TF-IDF по сочетаниям букв на заголовке
DS_POOL = 400  # TF-IDF по сочетаниям букв на коротком тексте
TOPN = 50  # бюджет ответа на один запрос по условию задачи


def norm(text):
    """Привести строку к нижнему регистру и убрать лишнее."""
    if not isinstance(text, str):
        return ""
    return text.lower().replace("ё", "е").strip()


def full_text(row):
    """Весь текст объявления. Заголовок повторён трижды, чтобы короткий
    заголовок не терялся на фоне длинного описания."""
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
    """Короткий текст для BM25. Заголовок повторён дважды — та же причина,
    что и утроение в полном тексте, но слабее, описания здесь нет."""
    title = str(row.get("item_title_raw", "") or "")
    params = str(row.get("item_infm_params_text", "") or "")
    return norm(" ".join([title] * 2 + [params]))


def title_text(row):
    """Один заголовок. Нужен для буквенного поиска и для расширения запроса."""
    return norm(str(row.get("item_title_raw", "") or ""))


def round_robin_union(lists):
    """Объединить списки по одному элементу из каждого по кругу.

    Так набор получается разнообразнее, чем при склейке по весам:
    каждая ветка гарантированно отдаёт своих лучших, даже если её веса
    в целом ниже. Повторы убираются, порядок внутри — порядок веток.
    """
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

    items = pd.read_parquet(ROOT / "dataset/benchmark_items.parquet")
    queries = pd.read_parquet(ROOT / "dataset/benchmark_queries.parquet")
    train = pd.read_parquet(
        ROOT / "dataset/train.parquet",
        columns=["search_query", "search_location_id", "item_id", "item_title_raw"],
    )
    # Идентификаторы — строки: числом можно потерять ведущие нули.
    items["item_id"] = items["item_id"].astype(str)
    queries["query_id"] = queries["query_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)
    print(f"items={len(items)} queries={len(queries)} train={len(train)}", flush=True)

    corpus_ids = set(items["item_id"].astype(str))
    city_of = dict(zip(items["item_id"].astype(str), items["item_location_id"]))
    subcat_of = dict(
        zip(items["item_id"].astype(str), items["item_microcat_id"].astype(str))
    )
    cids = items["item_id"].astype(str).tolist()
    titles = [title_text(row) for _, row in items.iterrows()]

    # --- История выборов из обучения ---
    # Нормализованный текст запроса нужен, чтобы "Ремонт " и "ремонт" были одним ключом.
    train["nq"] = (
        train["search_query"].astype(str).str.lower().str.replace("ё", "е").str.strip()
    )
    # Три самых частых заголовка на запрос: из их слов будем дополнять запрос.
    # Слова, а не готовые ответы, потому что сами идентификаторы в 95 процентах
    # случаев отсутствуют в корпусе бенчмарка и как ответы почти бесполезны.
    top_titles = (
        train.groupby("nq")[["item_title_raw"]]
        .apply(
            lambda g: [
                t for t, _ in Counter(g["item_title_raw"].astype(str)).most_common(3)
            ]
        )
        .to_dict()
    )
    in_corpus = train["item_id"].isin(corpus_ids)
    # Готовые ответы из истории: по тексту запроса и по паре "текст + город",
    # отсортированные по частоте выборов. Городные идут первыми.
    log_by_text = (
        train[in_corpus]
        .groupby("nq")["item_id"]
        .apply(lambda s: s.value_counts().index.tolist())
        .to_dict()
    )
    log_by_text_city = (
        train[in_corpus]
        .groupby(["nq", "search_location_id"])["item_id"]
        .apply(lambda s: s.value_counts().index.tolist())
        .to_dict()
    )
    # Для поиска похожих запросов: все разные тексты запросов из истории
    # и их ответы, плюс матрица TF-IDF по текстам запросов.
    uniq_queries = train.drop_duplicates("nq")[["nq"]].reset_index(drop=True)
    uq_items = (
        train[in_corpus]
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
    del train
    gc.collect()

    qtexts = [norm(q) for q in queries["search_query"]]
    qcities = queries["search_location_id"].tolist()

    # --- Дополнение запроса словами из истории ---
    # Берём до 10 слов длиной от 4 букв, которых ещё нет в запросе
    # (сравнение по основам, чтобы не дублировать словоформы).
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
        """Разбить тексты на слова и привести каждое к основе.

        Основы считаются один раз и кладутся в словарь: уникальных слов
        около 600 тысяч, без кэша стеммер считал бы одни и те же слова
        в каждом документе заново.
        """
        raw = [TOK.findall(t.lower()) for t in texts]
        vocab = set(w for tokens in raw for w in tokens)
        cache = {w: stem.stem(w) for w in vocab}
        return [[cache[w] for w in tokens] for tokens in raw]

    print("bm25 index...", flush=True)
    full_docs = [full_text(row) for _, row in items.iterrows()]
    short_docs = [short_bm25_text(row) for _, row in items.iterrows()]
    docs_full = stem_tokens(full_docs)
    docs_short = stem_tokens(short_docs)
    del full_docs, short_docs
    gc.collect()
    # Пустой запрос после токенизации заменяем заглушкой, иначе BM25 падает.
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

    print("bm25 retrieve...", flush=True)
    res_full, _ = index_full.retrieve(q_full, k=A_POOL)
    res_exp, _ = index_full.retrieve(q_exp, k=AE_POOL)
    # Дополнение словами из трёх первых найденных заголовков и второй проход
    # по тому же индексу: находит объявления, где запрос встречается не дословно,
    # а рядом с теми же словами, что у лидеров выдачи.
    q_prf = []
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
        q_prf.append(
            [stem.stem(w) for w in TOK.findall(extended.lower())] or ["__no_token__"]
        )
    res_prf, _ = index_full.retrieve(q_prf, k=PRF_POOL)
    del index_full, q_prf
    gc.collect()
    res_short, _ = index_short.retrieve(q_short, k=B_POOL)
    del index_short
    gc.collect()

    # --- Четыре матрицы TF-IDF ---
    # Словесные ловят фразы, буквенные — окончания, опечатки и сложные слова.
    # Редкие слова не выкидываем (предел частоты 1): для хвостовых запросов
    # редкое слово — самое различительное.
    print("tfidf fit...", flush=True)
    short_raw = [short_text_raw(row) for _, row in items.iterrows()]
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
    full_raw = [full_text(row) for _, row in items.iterrows()]
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
    title_raw = [title_text(row) for _, row in items.iterrows()]
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
    short_for_chars = [short_text_raw(row) for _, row in items.iterrows()]
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
        """Вернуть до k идентификаторов с ненулевым сходством, лучшие первыми."""
        qv = vectorizer.transform([query])
        scores = (matrix * qv.T).toarray().ravel()
        order = np.argpartition(scores, -k)[-k:]
        order = order[np.argsort(scores[order])[::-1]]
        return [cids[i] for i in order if scores[i] > 0]

    # --- Сборка ответа по каждому запросу ---
    print("fuse 2452...", flush=True)
    out = []
    for n in range(len(queries)):
        u_full = [cids[i] for i in res_full[n]]
        u_exp = [cids[i] for i in res_exp[n]]
        u_prf = [cids[i] for i in res_prf[n]]
        u_short = [cids[i] for i in res_short[n]]
        u_words_short = top_ids(mat_words_short, vec_words_short, qtexts[n], C_POOL)
        u_words_full = top_ids(mat_words_full, vec_words_full, qtexts[n], CF_POOL)
        u_chars_title = top_ids(mat_chars_title, vec_chars_title, qtexts[n], D_POOL)
        u_chars_short = top_ids(mat_chars_short, vec_chars_short, qtexts[n], DS_POOL)

        key = qtexts[n].lower().strip()
        # Готовые ответы из истории: сначала "текст + город", потом "текст".
        exact = [
            i for i in log_by_text_city.get((key, qcities[n]), []) if i in corpus_ids
        ][:TOPN]
        history = [
            i
            for i in log_by_text.get(key, [])
            if i in corpus_ids and i not in set(exact)
        ]
        history = (
            [i for i in history if city_of.get(i) == qcities[n]]
            + [i for i in history if city_of.get(i) != qcities[n]]
        )[:TOPN]
        # Похожие запросы из истории: три ближайших по TF-IDF, сходство от 0.05,
        # сам запрос пропускаем. Их ответы идут после точных.
        qv = vec_q.transform([key])
        sim = (query_matrix * qv.T).toarray().ravel()
        cand = np.argpartition(sim, -3)[-3:]
        cand = cand[np.argsort(sim[cand])[::-1]]
        fuzzy = []
        for j in cand:
            if sim[j] <= 0.05 or uniq_list[j] == key:
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
        # Итог: история впереди, затем объединённые ветки — сначала свой город,
        # внутри него сначала своя подкатегория, затем всё остальное.
        union = round_robin_union(
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
        union = [i for i in union if i not in set(head)]
        same_city = [i for i in union if city_of.get(i) == qcities[n]]
        other = [i for i in union if city_of.get(i) != qcities[n]]
        if best_subcat:
            same_city = [i for i in same_city if subcat_of.get(i) == best_subcat] + [
                i for i in same_city if subcat_of.get(i) != best_subcat
            ]
        out.append((head + same_city + other)[:TOPN])
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(queries)}", flush=True)

    # Проверка формата из условия задачи перед записью.
    answer = pd.DataFrame(
        {"query_id": queries["query_id"], "answer": [" ".join(row) for row in out]}
    )
    assert list(answer.columns) == ["query_id", "answer"] and len(answer) == len(
        queries
    )
    assert answer["query_id"].nunique() == len(queries)
    valid_ids = set(items["item_id"])
    for _, row in answer.iterrows():
        ids = str(row["answer"]).split()
        assert (
            len(ids) <= 50
            and len(ids) == len(set(ids))
            and all(i in valid_ids for i in ids)
        ), row["query_id"]
    answer.to_csv(OUT, index=False)
    print(f"WROTE {OUT} rows={len(answer)}", flush=True)


if __name__ == "__main__":
    main()
