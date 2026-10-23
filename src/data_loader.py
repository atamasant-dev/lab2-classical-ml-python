"""Генерація, завантаження та предобробка даних для NIDS (лабораторна №2, варіант 2).

Вхідні ознаки X: PacketSize, IAT, TCP_Flags, WindowSize, ActiveSessions.
Цільова змінна Y: 0 — Normal, 1 — Attack (PortScan / DDoS).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

FEATURES = ["PacketSize", "IAT", "TCP_Flags", "WindowSize", "ActiveSessions"]
TARGET = "Label"
FLAGS = ["SYN", "SYN-ACK", "ACK", "PSH-ACK", "FIN-ACK", "RST"]

# Ознаки зі «степеневим» розподілом: перед масштабуванням беремо log(1+x)
LOG_FEATURES = ["IAT", "ActiveSessions"]
LINEAR_FEATURES = ["PacketSize", "WindowSize"]
CATEGORICAL_FEATURES = ["TCP_Flags"]

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "telemetry_dataset.csv"


# --------------------------------------------------------------------------- #
# Генерація синтетичного набору
# --------------------------------------------------------------------------- #
def _choice_flags(rng: np.random.Generator, size: int, probs: dict[str, float]) -> np.ndarray:
    p = np.array([probs.get(f, 0.0) for f in FLAGS], dtype=float)
    return rng.choice(FLAGS, size=size, p=p / p.sum())


def _normal_traffic(rng: np.random.Generator, n: int) -> pd.DataFrame:
    # Розмір пакета: двомодальний (малі ACK-пакети та великі пакети даних)
    small = rng.random(n) < 0.45
    size = np.where(small, rng.normal(90, 30, n), rng.normal(1100, 350, n))
    iat = rng.lognormal(np.log(25), 1.0, n)  # мс
    flags = _choice_flags(rng, n, {"ACK": .38, "PSH-ACK": .30, "SYN": .10, "SYN-ACK": .08, "FIN-ACK": .09, "RST": .05})
    window = rng.choice([8192, 16384, 29200, 32768, 64240, 65535], size=n) * rng.uniform(0.9, 1.0, n)
    sessions = rng.poisson(18, n) + 1
    busy = rng.random(n) < 0.10  # завантажений легітимний сервер
    sessions = np.where(busy, rng.poisson(120, n) + 1, sessions)
    df = pd.DataFrame({"PacketSize": size, "IAT": iat, "TCP_Flags": flags, "WindowSize": window, "ActiveSessions": sessions})

    # «Складні» нормальні записи: легітимні сплески (CDN, бекапи) схожі на атаку
    burst = rng.random(n) < 0.05
    k = int(burst.sum())
    df.loc[burst, "IAT"] = rng.lognormal(np.log(1.0), 0.8, k)
    df.loc[burst, "ActiveSessions"] = np.exp(rng.normal(np.log(300), 0.5, k)).astype(int)
    df.loc[burst, "TCP_Flags"] = _choice_flags(rng, k, {"SYN": .4, "ACK": .35, "PSH-ACK": .25})
    return df


def _portscan_traffic(rng: np.random.Generator, n: int) -> pd.DataFrame:
    size = rng.normal(54, 8, n)
    iat = rng.lognormal(np.log(3.0), 0.9, n)
    flags = _choice_flags(rng, n, {"SYN": .70, "RST": .15, "FIN-ACK": .08, "ACK": .07})
    typical = rng.random(n) < 0.80  # nmap та подібні сканери використовують малі вікна
    window = np.where(typical, rng.choice([1024, 2048, 4096, 5840], size=n), rng.choice([8192, 29200, 64240], size=n))
    sessions = np.exp(rng.normal(np.log(90), 0.6, n)).astype(int)
    return pd.DataFrame({"PacketSize": size, "IAT": iat, "TCP_Flags": flags, "WindowSize": window, "ActiveSessions": sessions})


def _ddos_traffic(rng: np.random.Generator, n: int) -> pd.DataFrame:
    small = rng.random(n) < 0.60  # SYN-flood (малі пакети) та HTTP-flood (великі)
    size = np.where(small, rng.normal(60, 15, n), rng.normal(700, 400, n))
    iat = rng.lognormal(np.log(0.15), 0.8, n)
    flags = _choice_flags(rng, n, {"SYN": .60, "ACK": .20, "PSH-ACK": .20})
    spoofed = rng.random(n) < 0.50  # підроблені (випадкові) вікна
    window = np.where(spoofed, rng.integers(0, 65536, n), rng.choice([512, 1024, 65535], size=n))
    sessions = np.exp(rng.normal(np.log(1500), 0.7, n)).astype(int)
    return pd.DataFrame({"PacketSize": size, "IAT": iat, "TCP_Flags": flags, "WindowSize": window, "ActiveSessions": sessions})


def generate_synthetic_nids(n_samples: int = 6000, attack_ratio: float = 0.30, seed: int = 42) -> pd.DataFrame:
    """Генерує синтетичний мережевий трафік: Normal (0) та Attack (1: PortScan + DDoS)."""
    rng = np.random.default_rng(seed)
    n_attack = int(n_samples * attack_ratio)
    n_port = n_attack // 2
    n_ddos = n_attack - n_port
    n_norm = n_samples - n_attack

    normal = _normal_traffic(rng, n_norm)
    port = _portscan_traffic(rng, n_port)
    ddos = _ddos_traffic(rng, n_ddos)

    # «Стелс-атаки»: 10% атак маскуються під нормальну швидкість та кількість сесій
    attacks = pd.concat([port, ddos], ignore_index=True)
    stealth = rng.random(len(attacks)) < 0.10
    ks = int(stealth.sum())
    attacks.loc[stealth, "IAT"] = rng.lognormal(np.log(25), 1.0, ks)
    attacks.loc[stealth, "ActiveSessions"] = rng.poisson(25, ks) + 1

    normal[TARGET] = 0
    attacks[TARGET] = 1
    df = pd.concat([normal, attacks], ignore_index=True)

    # Фізичні межі значень
    df["PacketSize"] = df["PacketSize"].clip(40, 1500).round().astype(int)
    df["IAT"] = df["IAT"].clip(0.01, 2000).round(3)
    df["WindowSize"] = df["WindowSize"].clip(0, 65535).round().astype(int)
    df["ActiveSessions"] = df["ActiveSessions"].clip(1, 10000).astype(int)

    # Шум розмітки: 0.5% міток помилкові (реальні дані ніколи не розмічені ідеально)
    flip = rng.random(len(df)) < 0.005
    df.loc[flip, TARGET] = 1 - df.loc[flip, TARGET]

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)[FEATURES + [TARGET]]


def save_dataset(df: pd.DataFrame, path: Path = DEFAULT_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """Завантажує CSV та перевіряє наявність усіх потрібних колонок."""
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"У файлі {path} відсутні колонки: {sorted(missing)}")
    return df


# --------------------------------------------------------------------------- #
# Предобробка (Scaling, Encoding)
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """log(1+x) + StandardScaler для «важких хвостів», StandardScaler для решти, One-Hot для TCP_Flags."""
    log_branch = Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                           ("scale", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("log", log_branch, LOG_FEATURES),
            ("num", StandardScaler(), LINEAR_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Стратифікований поділ train/test (зберігає частку атак у обох частинах)."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


if __name__ == "__main__":
    data = generate_synthetic_nids()
    print("Збережено:", save_dataset(data), data.shape)
