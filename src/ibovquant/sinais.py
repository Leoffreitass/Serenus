"""Sinal defensivo de baixa volatilidade e seleção top-N com histerese.

Todos os sinais devolvem um DataFrame (datas de formação x tickers) em que
"maior valor = melhor"; `select_top` compra os N maiores. Por isso o sinal
de baixa volatilidade é devolvido com sinal trocado (-vol).

Convenções:
- Seleção: top-N do universo elegível; histerese: papel em carteira só sai
  quando seu rank cai abaixo de `band` (>N), reduzindo turnover.
- Sem look-ahead: o sinal do fim do mês m forma a carteira mantida no mês m+1.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def sinal_low_vol(wide_daily: pd.DataFrame, datas: pd.DatetimeIndex,
                  window: int = 252, min_obs: int = 126) -> pd.DataFrame:
    """Sinal defensivo de baixa volatilidade (Blitz & van Vliet, 2007;
    Ang et al., 2006): em cada data de formação, a vol realizada anualizada
    dos últimos `window` pregões, COM SINAL TROCADO — assim `select_top`
    (que compra os maiores valores) seleciona os papéis MENOS voláteis.

    Pseudocódigo:
        função LOW_VOL(P diário, datas, janela=252, min_obs=126):
            para cada formação t:
                vol_i ← desvio(retornos diários de i nos últimos 252
                               pregões ATÉ t) · √252
                se nº de observações < min_obs: vol_i ← NaN
                sinal[t,i] ← −vol_i        # trocado: maior = mais calmo
    """
    dret = wide_daily.pct_change(fill_method=None)
    out = {}
    for dt in datas:
        jan = dret.loc[:dt].tail(window)
        vol = jan.std() * np.sqrt(252)
        vol[jan.notna().sum() < min_obs] = np.nan
        out[dt] = -vol
    return pd.DataFrame(out).T


def cross_ranks(signal: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """Rank cross-sectional (1 = melhor sinal) apenas entre elegíveis."""
    s = signal.where(elig)
    return s.rank(axis=1, ascending=False, method='first')


@dataclass
class Selecao:
    holdings: pd.DataFrame          # bool, datas de formação x tickers
    ranks: pd.DataFrame             # ranks cross-sectional
    turnover: pd.Series = field(default=None)  # fração da carteira trocada por mês


def select_top(signal: pd.DataFrame, elig: pd.DataFrame,
               n: int = 15, band: int | None = 30) -> Selecao:
    """Seleciona top-N por mês com banda de histerese.

    band=None desliga a histerese (carteira = top-N puro todo mês).
    Regra com histerese: mantém posições atuais com rank <= band; completa as
    vagas com os melhores ranqueados fora da carteira.

    Pseudocódigo:
        função SELECT_TOP(sinal, elegíveis, n=15, band=30):
            carteira ← vazia
            para cada formação t (cronológico):
                ranks ← ordenar elegíveis por sinal (1 = melhor)
                mantidos ← papéis da carteira com rank ≤ band   # histerese
                novos ← melhores fora da carteira, até completar n
                carteira[t] ← mantidos + novos
    """
    ranks = cross_ranks(signal, elig)
    hold_prev: set = set()
    linhas, turn = {}, {}
    for dt, row in ranks.iterrows():
        r = row.dropna().sort_values()
        if len(r) < n:                       # sem universo suficiente ainda
            hold_prev = set()
            continue
        if band is None or not hold_prev:
            atual = list(r.index[:n])
        else:
            mantidos = [t for t in hold_prev if t in r.index and r[t] <= band]
            novos = [t for t in r.index if t not in mantidos][: n - len(mantidos)]
            atual = mantidos + novos
        turn[dt] = 1.0 - len(set(atual) & hold_prev) / n if hold_prev else np.nan
        linhas[dt] = atual
        hold_prev = set(atual)

    holdings = pd.DataFrame(False, index=list(linhas), columns=ranks.columns)
    for dt, ticks in linhas.items():
        holdings.loc[dt, ticks] = True
    return Selecao(holdings=holdings, ranks=ranks, turnover=pd.Series(turn))


