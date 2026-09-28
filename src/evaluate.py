"""Подсчёт полноты топ-50.

Полнота одного запроса — доля его подходящих объявлений, попавших в ответ.
Итог — среднее по запросам. Порядок объявлений в ответе не важен,
поэтому предсказания сравниваются как множества.
"""


def recall_at_50(predictions: dict, ground_truth: dict) -> float:
    """Посчитать Recall@50.

    predictions: словарь "запрос -> список идентификаторов (первые 50 берутся)".
    ground_truth: словарь "запрос -> множество подходящих идентификаторов".
    """
    scores = []
    for query, relevant in ground_truth.items():
        relevant = set(map(str, relevant))
        if not relevant:
            continue
        predicted = set(map(str, predictions.get(query, [])[:50]))
        scores.append(len(predicted & relevant) / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def make_split(train_df, n_valid=500, seed=42):
    """Разбить обучение на части не реализовано здесь специально.

    Честное деление — целыми группами "текст запроса, город, категория",
    иначе пары "запрос — объявление" утекут в проверку через историю выборов.
    Такое деление живёт прямо в проверочных скриптах (seed 42, 500 групп),
    поэтому заготовка ниже не используется финалом.
    """
    raise NotImplementedError("Деление описано выше; см. scripts/val_lexmax_v4.py")
