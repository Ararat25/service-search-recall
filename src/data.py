"""Загрузка данных и подготовка текстов. Идентификаторы — только строки."""

import pandas as pd

TEXT_QUERY = "search_query"
TEXT_DOC_FIELDS = ["item_title_raw", "item_description_raw", "item_infm_params_text"]


def load_all(train_path, queries_path, items_path):
    """Прочитать три parquet. Вернуть (обучение, запросы, объявления).

    Идентификаторы приводятся к строкам: числом можно потерять ведущие нули
    и получить несовпадение с корпусом (такие ответы метрика молча не засчитает).
    """
    train = pd.read_parquet(train_path)
    queries = pd.read_parquet(queries_path)
    items = pd.read_parquet(items_path)
    if "item_id" in train.columns:
        train["item_id"] = train["item_id"].astype(str)
    queries["query_id"] = queries["query_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)
    return train, queries, items


def normalize_text(text: str) -> str:
    """Базовая чистка строки: нижний регистр и края без пробелов."""
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def build_doc_text(row) -> str:
    """Склеить поля объявления в один текст (для простых поисковых базовых линий)."""
    parts = [str(row.get(field, "") or "") for field in TEXT_DOC_FIELDS]
    return normalize_text(" ".join(parts))
