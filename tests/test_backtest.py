# -*- coding: utf-8 -*-
"""Testes de sanidade do motor de backtest e dos módulos críticos.

Rodar da raiz do repositório:  pytest tests/ -q
Cobrem as propriedades que sustentam as conclusões do projeto:
sem look-ahead, contabilidade de custos, histerese, convenção de sinal
do low-vol, pesos HRP válidos e clip do vol targeting.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ibovquant import backtest as bt, hrp, sinais, vol_targeting as vt  # noqa: E402


# ------------------------------------------------------------------ fixtures

def _painel_sintetico(n_meses=14, tickers=('AAA', 'BBB', 'CCC'), seed=7):
    """Preços mensais sintéticos (fins de mês) com retornos conhecidos."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2020-01-31', periods=n_meses, freq='ME')
    rets = pd.DataFrame(rng.normal(0.01, 0.05, (n_meses, len(tickers))),
                        index=idx, columns=list(tickers))
    return 100.0 * (1 + rets).cumprod()


# ------------------------------------------------------------------ backtest

def test_backtest_sem_lookahead():
    """O retorno creditado ao mês m+1 deve ser exatamente o retorno dos
    ativos ENTRE m e m+1, com os pesos formados em m."""
    mclose = _painel_sintetico()
    dt_form = mclose.index[5]
    pesos = pd.DataFrame(0.0, index=[dt_form], columns=mclose.columns)
    pesos.loc[dt_form, ['AAA', 'BBB']] = 0.5
    cdi = pd.Series(0.0, index=mclose.index)

    res = bt.backtest(pesos, mclose, cdi, w_vt=1.0, custo_lado=0.0)
    nxt = mclose.index[6]
    esperado = 0.5 * (mclose.loc[nxt, 'AAA'] / mclose.loc[dt_form, 'AAA'] - 1) \
             + 0.5 * (mclose.loc[nxt, 'BBB'] / mclose.loc[dt_form, 'BBB'] - 1)
    assert res.liquido.index[0] == nxt
    assert np.isclose(res.liquido.iloc[0], esperado)


def test_backtest_custo_na_montagem_inicial():
    """No primeiro mês, o giro é a compra da carteira inteira: custo =
    custo_lado * w_vt (Σ|W| = w_vt)."""
    mclose = _painel_sintetico()
    dt_form = mclose.index[5]
    pesos = pd.DataFrame(0.0, index=[dt_form], columns=mclose.columns)
    pesos.loc[dt_form, ['AAA', 'BBB']] = 0.5
    cdi = pd.Series(0.0, index=mclose.index)

    custo_lado = 0.0015
    w = 0.8
    res = bt.backtest(pesos, mclose, cdi, w_vt=w, custo_lado=custo_lado)
    assert np.isclose(res.turnover.iloc[0], w)
    assert np.isclose(res.custos.iloc[0], custo_lado * w)
    assert np.isclose(res.bruto.iloc[0] - res.liquido.iloc[0], custo_lado * w)


def test_backtest_turnover_com_drift():
    """Mesma carteira-alvo em dois meses seguidos: o giro do 2º mês é
    exatamente o custo de desfazer o drift — Σ|W_alvo − W_end|, com
    W_end_i = W_i·(1+r_i)/(1+r_carteira). Sem drift seria zero."""
    mclose = _painel_sintetico()
    d1, d2, d3 = mclose.index[5], mclose.index[6], mclose.index[7]
    pesos = pd.DataFrame(0.0, index=[d1, d2], columns=mclose.columns)
    pesos.loc[[d1, d2], ['AAA', 'BBB']] = 0.5
    cdi = pd.Series(0.0, index=mclose.index)

    res = bt.backtest(pesos, mclose, cdi, w_vt=1.0, custo_lado=0.0015)

    r = mclose.loc[d2, ['AAA', 'BBB']] / mclose.loc[d1, ['AAA', 'BBB']] - 1
    rb = 0.5 * r['AAA'] + 0.5 * r['BBB']
    w_end = 0.5 * (1 + r) / (1 + rb)              # pesos efetivos pós-drift
    giro_esperado = float((0.5 - w_end).abs().sum())

    assert giro_esperado > 0                       # drift existe de fato
    assert np.isclose(res.turnover.iloc[1], giro_esperado)
    assert np.isclose(res.custos.iloc[1], 0.0015 * giro_esperado)


def test_backtest_caixa_rende_cdi():
    """Com w_vt = 0, o retorno bruto do mês deve ser exatamente o CDI."""
    mclose = _painel_sintetico()
    dt_form = mclose.index[5]
    pesos = pd.DataFrame(0.0, index=[dt_form], columns=mclose.columns)
    pesos.loc[dt_form, 'AAA'] = 1.0
    nxt = mclose.index[6]
    cdi = pd.Series(0.01, index=mclose.index)

    res = bt.backtest(pesos, mclose, cdi, w_vt=0.0, custo_lado=0.0)
    assert np.isclose(res.bruto.loc[nxt], 0.01)


def test_backtest_delistado_congela_sem_lookahead():
    """Papel sem preço no mês seguinte fica CONGELADO (retorno 0, peso não
    migra) — redistribuir usaria a disponibilidade de m+1 na formação
    (look-ahead) e concentraria nos sobreviventes."""
    mclose = _painel_sintetico()
    dt_form, nxt = mclose.index[5], mclose.index[6]
    mclose.loc[nxt:, 'BBB'] = np.nan          # delistou depois da formação
    pesos = pd.DataFrame(0.0, index=[dt_form], columns=mclose.columns)
    pesos.loc[dt_form, ['AAA', 'BBB']] = 0.5
    cdi = pd.Series(0.0, index=mclose.index)

    res = bt.backtest(pesos, mclose, cdi, w_vt=1.0, custo_lado=0.0)
    esperado = 0.5 * (mclose.loc[nxt, 'AAA'] / mclose.loc[dt_form, 'AAA'] - 1) \
             + 0.5 * 0.0                      # perna delistada rende 0
    assert np.isclose(res.liquido.loc[nxt], esperado)


# ------------------------------------------------------------------ sinais

def test_sinal_low_vol_prefere_o_mais_calmo():
    """Convenção -vol: o papel menos volátil deve ter o MAIOR sinal."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range('2020-01-01', periods=400)
    wide = pd.DataFrame({
        'CALMO':   100 * (1 + rng.normal(0, 0.005, 400)).cumprod(),
        'NERVOSO': 100 * (1 + rng.normal(0, 0.030, 400)).cumprod(),
    }, index=idx)
    sig = sinais.sinal_low_vol(wide, pd.DatetimeIndex([idx[-1]]))
    assert sig.loc[idx[-1], 'CALMO'] > sig.loc[idx[-1], 'NERVOSO']


def test_select_top_histerese_segura_posicao():
    """Papel em carteira com rank entre n e band deve ser MANTIDO;
    sem histerese (band=None) ele sairia."""
    datas = pd.DatetimeIndex(['2020-01-31', '2020-02-29'])
    tickers = [f'T{i:02d}' for i in range(6)]
    elig = pd.DataFrame(True, index=datas, columns=tickers)
    sig = pd.DataFrame(index=datas, columns=tickers, dtype=float)
    sig.loc[datas[0]] = [6, 5, 4, 3, 2, 1]          # top-3: T00, T01, T02
    sig.loc[datas[1]] = [6, 5, 1, 3, 4, 2]          # T02 cai p/ rank 6... mas band=4 -> rank<=4? não: sai
    sig.loc[datas[1]] = [6, 5, 3.5, 3, 4, 2]        # T02 vira rank 4 (entre n=3 e band=4): mantém

    com_hist = sinais.select_top(sig, elig, n=3, band=4)
    sem_hist = sinais.select_top(sig, elig, n=3, band=None)
    assert bool(com_hist.holdings.loc[datas[1], 'T02'])       # histerese segura
    assert not bool(sem_hist.holdings.loc[datas[1], 'T02'])   # top-N puro troca


# ------------------------------------------------------------------ hrp / vt

def test_hrp_pesos_validos():
    """Pesos HRP: não-negativos, somam 1, e favorecem o ativo de menor risco."""
    rng = np.random.default_rng(11)
    n = 300
    base = rng.normal(0, 0.01, n)
    rets = pd.DataFrame({
        'LV':  0.5 * base + rng.normal(0, 0.004, n),   # baixa vol
        'HV1': base + rng.normal(0, 0.02, n),          # alta vol, correlacionados
        'HV2': base + rng.normal(0, 0.02, n),
    })
    w = hrp.hrp_weights(rets.cov() * 252)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    assert w['LV'] == w.max()


def test_vt_exposicao_clip():
    """w = σ_alvo/σ̂ com teto em w_max (sem alavancagem) e piso em 0."""
    prev = pd.Series([0.10, 0.15, 0.30, 0.75])
    w = vt.exposicao(prev, sigma_alvo=0.15, w_max=1.0)
    assert np.allclose(w.values, [1.0, 1.0, 0.5, 0.2])
