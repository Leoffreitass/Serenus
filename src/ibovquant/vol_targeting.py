"""Volatility targeting no nível da carteira.

    w_t = clip(σ_alvo / σ̂_t, 0, w_max)

onde σ̂_t é a vol prevista (GARCH, EWMA ou realizada) na data de formação e a
fração (1 − w_t) fica em caixa remunerado a CDI. Long-only: w_max = 1 por
default (sem alavancagem).

Ref.: Moreira & Muir (2017); Harvey et al. (2018); Barroso & Santa-Clara
(2015). Ver docs/research/vt_garch_notes.md.
"""
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data'


def exposicao(vol_prevista: pd.Series, sigma_alvo: float = 0.15,
              w_max: float = 1.0) -> pd.Series:
    """Fração da carteira em risco a cada data de formação.

    Pseudocódigo:
        função EXPOSIÇÃO(σ̂, σ_alvo=15%, w_max=1):
            para cada formação t:
                w[t] ← clip( σ_alvo / σ̂[t],  0,  w_max )
            # (1 − w) fica em caixa remunerado ao CDI
    """
    return (sigma_alvo / vol_prevista).clip(upper=w_max, lower=0.0)


def load_cdi(path: Path | str | None = None) -> pd.Series:
    """CDI mensal (fração, ex.: 0.0105 = 1,05% no mês), indexado por fim de mês.
    Fonte: BCB/SGS série 4391 (acumulada no mês)."""
    d = json.load(open(path or DATA / 'cdi_mensal.json'))
    s = pd.Series({pd.Period(k, 'M').to_timestamp('M'): v / 100.0
                   for k, v in d['series'].items()}).sort_index()
    s.name = 'cdi'
    return s
