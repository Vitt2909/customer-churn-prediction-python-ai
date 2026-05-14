from pathlib import Path
import random

import numpy as np
import pandas as pd


SEED = 42
N_CLIENTES = 3000
CHURN_RATE = 0.25

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT_DIR / "data" / "clientes_churn_ficticio.csv"

PLANOS = ["Starter", "Pro", "Enterprise"]
CANAIS = ["Indicação", "Google Ads", "LinkedIn", "Evento", "Inbound"]
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


def gerar_base() -> pd.DataFrame:
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    churn = np.array([1] * int(N_CLIENTES * CHURN_RATE) + [0] * int(N_CLIENTES * (1 - CHURN_RATE)))
    rng.shuffle(churn)

    reference_date = pd.Timestamp("2025-12-31")
    rows = []

    for index, churn_flag in enumerate(churn, start=1):
        is_churn = churn_flag == 1

        plano = rng.choice(PLANOS, p=[0.55, 0.35, 0.10] if is_churn else [0.38, 0.43, 0.19])
        canal = rng.choice(CANAIS, p=[0.12, 0.36, 0.23, 0.10, 0.19] if is_churn else [0.30, 0.20, 0.15, 0.10, 0.25])
        regiao = rng.choice(REGIOES, p=[0.10, 0.25, 0.11, 0.38, 0.16])

        tempo_meses = int(np.clip(round(rng.normal(13, 8) if is_churn else rng.normal(32, 15)), 1, 72))
        data_cadastro = reference_date - pd.to_timedelta(int(tempo_meses * 30 + rng.integers(0, 30)), unit="D")

        numero_produtos = int(
            rng.choice(
                [1, 2, 3, 4, 5],
                p=[0.55, 0.28, 0.12, 0.04, 0.01] if is_churn else [0.20, 0.32, 0.25, 0.16, 0.07],
            )
        )

        tickets_abertos = int(np.clip(rng.poisson(4.8 if is_churn else 1.3), 0, 15))
        taxa_resolucao = 0.42 if is_churn else 0.82
        tickets_resolvidos = int(rng.binomial(tickets_abertos, taxa_resolucao)) if tickets_abertos > 0 else 0

        nps_score = int(np.clip(round(rng.normal(34, 18) if is_churn else rng.normal(74, 14)), 0, 100))
        uso_medio = float(np.clip(rng.normal(9, 5) if is_churn else rng.normal(31, 11), 0.5, 80))
        atraso_pagamento = int(np.clip(round(rng.gamma(2.2, 5.5) if is_churn else rng.gamma(1.2, 2.0)), 0, 45))
        desconto_recebido = int(rng.random() < (0.44 if is_churn else 0.24))

        rows.append(
            {
                "cliente_id": f"CLI{index:05d}",
                "data_cadastro": data_cadastro.date().isoformat(),
                "plano": plano,
                "tempo_como_cliente_meses": tempo_meses,
                "numero_produtos": numero_produtos,
                "tickets_abertos_ultimo_mes": tickets_abertos,
                "tickets_resolvidos": tickets_resolvidos,
                "nps_score": nps_score,
                "uso_medio_mensal_horas": round(uso_medio, 2),
                "atraso_pagamento_dias": atraso_pagamento,
                "desconto_recebido": desconto_recebido,
                "canal_aquisicao": canal,
                "regiao": regiao,
                "churn": int(churn_flag),
            }
        )

    df = pd.DataFrame(rows)
    return df[
        [
            "cliente_id",
            "data_cadastro",
            "plano",
            "tempo_como_cliente_meses",
            "numero_produtos",
            "tickets_abertos_ultimo_mes",
            "tickets_resolvidos",
            "nps_score",
            "uso_medio_mensal_horas",
            "atraso_pagamento_dias",
            "desconto_recebido",
            "canal_aquisicao",
            "regiao",
            "churn",
        ]
    ]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = gerar_base()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    churn_rate = df["churn"].mean()
    print(f"Arquivo gerado: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Distribuição churn: {df['churn'].value_counts().to_dict()}")
    print(f"Taxa de churn: {churn_rate:.2%}")


if __name__ == "__main__":
    main()
