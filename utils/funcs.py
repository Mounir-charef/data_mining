import pandas as pd
from itertools import combinations
from pprint import pprint
from typing import Iterable
from collections import defaultdict


def load_data(path: str):
    columns_names = ['Watcher', 'videoCategoryId',
                     'videoCategoryLabel', 'definition']
    return pd.read_csv(path, names=columns_names, header=None, skiprows=1)


def data_preprocessing(df: pd.DataFrame):
    df.drop('definition', inplace=True, axis=1)
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)
    df['videoCategoryId'] = df['videoCategoryId'].astype(int)


def data_study(df: pd.DataFrame):
    print(f'Number of rows: {len(df)}')
    print(f'Number of Items: {len(df["videoCategoryId"].unique())}')
    print(f'Number of transactions: {len(df["Watcher"].unique())}')


def get_transaction_df(df: pd.DataFrame):
    return df.groupby("Watcher").agg({
        'videoCategoryId': lambda x: set(x),
        'videoCategoryLabel': lambda x: set(x)
    }).sort_values(by='videoCategoryId', key=lambda x: x.apply(len), ascending=False).rename(
        columns={'Watcher': 'Transaction', 'videoCategoryId': 'Items', 'videoCategoryLabel': 'Items_Labels'})


def get_unique_items(transactions: Iterable) -> set:
    items = set()
    for transaction in transactions:
        items.update(transaction)
    return items


def remaining_items(transactions: dict):
    items = set()
    for transaction in transactions:
        items.update(transaction)
    return items


def generate_candidates(items, k) -> list[set]:
    return [set(combination) for combination in combinations(items, k)]


def support(transactions: Iterable[set], candidate: set | frozenset):
    return sum(1 for transaction in transactions if candidate.issubset(transaction))


def prune_candidates(transactions: Iterable[set], candidates: list[set], min_support: int) -> dict[dict, int]:
    count = {}
    for candidate in candidates:
        count[frozenset(candidate)] = support(transactions, candidate)
    return {k: v for k, v in count.items() if v >= min_support}


def apriori(transactions: Iterable[set], min_support: int) -> tuple[dict[int, set], int]:
    frequent_items = {}
    items = get_unique_items(transactions)
    candidates = generate_candidates(items, 1)
    support_counts = prune_candidates(transactions, candidates, min_support)
    if not support_counts:
        return {1: set()}, 1

    items = remaining_items(support_counts)
    frequent_items[1] = items
    k = 2
    while True:
        candidates = generate_candidates(items, k)
        if not candidates:
            break

        support_counts = prune_candidates(
            transactions, candidates, min_support)
        if not support_counts:
            break

        items = remaining_items(support_counts)

        frequent_items[k] = items
        k += 1

    return frequent_items, k-1


# noinspection PyTypeChecker
def generate_association_rules(frequent_items: set) -> defaultdict[frozenset, set]:
    max_k = len(frequent_items)
    rules = defaultdict(set)
    for k in range(1, max_k):
        transactions = list(combinations(frequent_items, k))
        for j in range(1, max_k):
            candidates_transaction = list(combinations(frequent_items, j))
            for translation in transactions:
                for candidate in candidates_transaction:
                    if not set(candidate).issubset(translation):
                        rules[frozenset(translation)].add(candidate)
    return rules


def find_confidence(rules: dict[frozenset, set], transactions: Iterable[set], thresh_hold: float = 0.5) -> None:
    for transaction, candidates in rules.items():
        for candidate in candidates:
            union = transaction.union(candidate)
            confidence = support(transactions, union) / \
                support(transactions, transaction)
            if confidence >= thresh_hold:
                print(f'{transaction} => {candidate} : {confidence:.2f}')


def main():
    file_path = 'Dataset-Exos2.csv'
    df = load_data(file_path)
    print(df.head())
    data_preprocessing(df)
    data_study(df)

    transactions_df = get_transaction_df(df)
    transactions = transactions_df['Items']

    frequent_items, k = apriori(transactions, min_support=3)
    print('Frequent items:\n')
    pprint(frequent_items)

    generated_rules = generate_association_rules(frequent_items[k])
    find_confidence(generated_rules, transactions)


if __name__ == "__main__":
    main()
