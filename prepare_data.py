import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


# Настройки
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "labeled.csv"

SEED = 42
MAX_LEN = 64
MAX_VOCAB = 20000
BATCH_SIZE = 32


def tokenize(text):
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def build_loaders():
    # 1. Загрузка данных
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Не найден датасет: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if not {"comment", "toxic"}.issubset(df.columns):
        raise ValueError(
            "CSV должен содержать comment и toxic"
        )

    # 2. Очистка
    df = df[["comment", "toxic"]].dropna().copy()

    df["comment"] = (
        df["comment"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["toxic"] = pd.to_numeric(
        df["toxic"], errors="coerce"
    )

    df = df.dropna()
    df = df[df["toxic"].isin([0, 1])]
    df = df[df["comment"] != ""]
    df = df.drop_duplicates(subset=["comment"])

    df = df[
        df["comment"].apply(lambda text: len(tokenize(text)) > 0)
    ].copy()

    df["toxic"] = df["toxic"].astype(int)

    # 3. Разделение 80% / 10% / 10%
    # Сохраняем примерные пропорции классов
    parts = {"train": [], "val": [], "test": []}

    for label, group in df.groupby("toxic"):
        group = group.sample(
            frac=1, random_state=SEED
        )

        n = len(group)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        parts["train"].append(group.iloc[:n_train])

        parts["val"].append(
            group.iloc[n_train:n_train + n_val]
        )

        parts["test"].append(
            group.iloc[n_train + n_val:]
        )

    train_df = pd.concat(parts["train"])
    val_df = pd.concat(parts["val"])
    test_df = pd.concat(parts["test"])

    # 4. Словарь только по обучающим данным
    counter = Counter()

    for text in train_df["comment"]:
        counter.update(tokenize(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, count in counter.most_common():
        if count < 2 or len(vocab) >= MAX_VOCAB:
            break

        vocab[word] = len(vocab)

    # 5. Сохраняем словарь для будущих предсказаний
    with open(
        ROOT / "data" / "vocab.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(vocab, file, ensure_ascii=False)

    # 6. Превращаем текст в ID токенов
    def encode(text):
        words = tokenize(text)[:MAX_LEN]

        ids = [
            vocab.get(word, vocab["<UNK>"])
            for word in words
        ]

        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))

        return ids

    # 7. Создаём тензоры PyTorch
    def make_dataset(frame):
        x = torch.tensor(
            [encode(text) for text in frame["comment"]],
            dtype=torch.long
        )

        y = torch.tensor(
            frame["toxic"].to_numpy(),
            dtype=torch.float32
        ).unsqueeze(1)

        return TensorDataset(x, y)

    train_dataset = make_dataset(train_df)
    val_dataset = make_dataset(val_df)
    test_dataset = make_dataset(test_df)

    # 8. DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Данные подготовлены:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))
    print("Vocabulary:", len(vocab))

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = (
        build_loaders()
    )

    xb, yb = next(iter(train_loader))

    print("X batch:", xb.shape)
    print("Y batch:", yb.shape)
