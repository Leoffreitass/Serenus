"""Motor de backtest mensal: sinal -> HRP -> vol targeting, com custos.

Linha do tempo (sem look-ahead):
  fim do mês m: sinal de seleção, covariância (HRP) e previsão de vol (GARCH)
                calculados SÓ com dados até o fechamento de m -> pesos totais
                W_m = w_vt(m) · pesos_hrp(m)
  mês m+1:      a carteira rende Σ W_m·r_i + (1-w_vt)·CDI, menos custos de
                transação sobre o giro |W_m − W_end_{m-1}|, onde W_end é a
                carteira EFETIVA ao fim do mês anterior (pós-drift): os pesos
                derivam com os retornos, W_end_i = W_i·(1+r_i)/(1+r_carteira),
                e o rebalance paga o giro real de volta ao alvo.

Convenções:
  - Retornos com `fill_method=None` (explícito): preço ausente vira retorno
    NaN, nunca um 0% fantasma por forward-fill (o default do pandas < 3.0
    faria pad e mascararia suspensões).
  - Papel em carteira SEM retorno no mês seguinte (suspensão/delisting
    intra-mês): o peso NÃO é redistribuído — redistribuir usaria a
    disponibilidade de m+1 na formação (look-ahead) e concentraria nos
    sobreviventes. A posição fica congelada com retorno 0 no mês
    (aproximação otimista para falências; documentada e monitorada — na
    amostra do Serenus o caso nunca ocorre).
  - Formação pulada NO MEIO da série (w_vt NaN ou cesta vazia) emite
    RuntimeWarning: o mês fica sem P&L e o giro seguinte é medido contra
    uma carteira defasada. Pulos no INÍCIO (warm-up do GARCH) são normais.

Custo default: 15 bps por lado (corretagem+emolumentos+slippage) sobre o
notional girado na perna de ações; caixa/CDI sem custo.

Pseudocódigo (os marcadores (a)..(e) aparecem no corpo de `backtest`):

    função BACKTEST(W_alvo [M×N], preços mensais P, cdi, w_vt [M], custo_lado):
        R ← retornos_mensais(P)              # sem forward-fill (NaN honesto)
        W_fim ← vazio                        # carteira efetiva pós-drift
        para cada formação m em ordem cronológica:
            # (a) validação da formação
            se w_vt[m] ausente ou cesta vazia: avisar se no meio; pular
            w ← w_vt[m];  b ← W_alvo[m] > 0, normalizado (soma 1)
            m+1 ← primeiro fim de mês após m; se não existe: parar
            # (b) retorno do mês m+1 (formado em m, carregado em m+1)
            r ← R[m+1, b];  r[NaN] ← 0       # congela delistado (sem look-ahead)
            rb ← w·(b'r) + (1−w)·cdi[m+1]    # bruto: ações + caixa
            # (c) giro real e custo, contra a carteira driftada
            W ← b·w
            giro ← Σ|W − W_fim|              # 1º mês: giro = w (montagem)
            custo ← custo_lado · giro
            # (d) registrar o mês
            bruto[m+1] ← rb;  líquido[m+1] ← rb − custo
            turnover[m] ← giro;  custos[m+1] ← custo
            # (e) drift intra-mês: a carteira herdada é o alvo DEPOIS de render
            W_fim ← W·(1+r)/(1+rb)
        retornar (líquido, bruto, w_vt, turnover, custos)
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ResultadoBacktest:
    liquido: pd.Series      # retorno mensal líquido
    bruto: pd.Series        # retorno mensal antes de custos
    w_vt: pd.Series         # exposição da perna de ações por mês de formação
    turnover: pd.Series     # Σ|W_alvo − W_end| (giro real, pós-drift) por formação
    custos: pd.Series       # custo do mês (fração)


def backtest(pesos_ativos: pd.DataFrame, mclose: pd.DataFrame,
             cdi: pd.Series, w_vt: pd.Series | float = 1.0,
             custo_lado: float = 0.0015) -> ResultadoBacktest:
    """Roda o backtest mensal.

    pesos_ativos: pesos da cesta (formação × ticker), somando 1 por linha.
    mclose:       fechamentos mensais ajustados.
    cdi:          retorno mensal do CDI (fração), indexado por fim de mês.
    w_vt:         exposição por data de formação (Series) ou constante.

    Pseudocódigo completo no docstring do módulo; os estágios (a)..(e)
    de lá estão marcados nos comentários do corpo abaixo.
    """
    mret = mclose.pct_change(fill_method=None)         # R ← retornos_mensais(P)  # NaN honesto, sem forward-fill
    if np.isscalar(w_vt):
        w_vt = pd.Series(float(w_vt), index=pesos_ativos.index)

    W_prev = pd.Series(dtype=float)                    # W_fim ← vazio  # carteira efetiva pós-drift
    iniciado = False
    bruto, liquido, turn, custos, wq = {}, {}, {}, {}, {}

    def _pulo(dt, motivo):
        if iniciado:
            warnings.warn(f'backtest: formação {dt:%Y-%m} pulada ({motivo}) '
                          'no meio da série — mês sem P&L e giro seguinte '
                          'medido contra carteira defasada', RuntimeWarning)

    for dt in pesos_ativos.index:                      # para cada formação m (ordem cronológica):
        # ---- (a) validação da formação ------------------------------------
        if dt not in w_vt.index or pd.isna(w_vt.loc[dt]):
            _pulo(dt, 'w_vt ausente')                  #   se w_vt[m] ausente: avisar se no meio; pular m
            continue
        wv = float(w_vt.loc[dt])                       #   w ← w_vt[m]  # exposição da perna de ações
        base = pesos_ativos.loc[dt]                    #   b ← W_alvo[m, :]
        base = base[base > 0]                          #   b ← b restrito a pesos > 0
        if base.empty:
            _pulo(dt, 'cesta vazia')                   #   se cesta vazia: avisar se no meio; pular m
            continue
        nxt_idx = mret.index[mret.index > dt]          #   m+1 ← primeiro fim de mês em R após m
        if len(nxt_idx) == 0:
            continue                                   #   se não existe m+1: parar
        nxt = nxt_idx[0]

        # ---- (b) retorno do mês m+1 (formado em m, carregado em m+1) ------
        wb = base / base.sum()                         #   b ← b / soma(b)  # cesta de ações soma 1
        r = mret.loc[nxt, wb.index].fillna(0.0)        #   r ← R[m+1, b];  r[NaN] ← 0  # congela delistado:
                                                       #     redistribuir usaria info de m+1 (look-ahead)
        r_acao = float((wb * r).sum())                 #   r_ação ← b' · r
        r_cdi = float(cdi.get(nxt, 0.0))               #   r_cdi ← cdi[m+1]  # 0 se ausente (conservador)
        rb = wv * r_acao + (1 - wv) * r_cdi            #   rb ← w·r_ação + (1−w)·r_cdi  # retorno BRUTO

        # ---- (c) giro real e custo, contra a carteira driftada ------------
        W = wb * wv                                    #   W ← b · w  # pesos totais alvo
        uniao = W.index.union(W_prev.index)            #   união dos ativos (peso 0 onde ausente)
        giro = float((W.reindex(uniao, fill_value=0.0)
                      - W_prev.reindex(uniao, fill_value=0.0)).abs().sum())
                                                       #   giro ← Σ_i |W_i − W_fim_i|
                                                       #     1º mês: W_fim vazio → giro = w (montagem)
        c = custo_lado * giro                          #   custo ← custo_lado · giro

        # ---- (d) registrar o mês ------------------------------------------
        bruto[nxt] = rb                                #   bruto[m+1] ← rb
        liquido[nxt] = rb - c                          #   líquido[m+1] ← rb − custo
        turn[dt], custos[nxt], wq[dt] = giro, c, wv    #   turnover[m] ← giro;  custos[m+1] ← custo

        # ---- (e) drift intra-mês ------------------------------------------
        W_prev = W * (1.0 + r) / (1.0 + rb)            #   W_fim ← W·(1+r)/(1+rb)  # a carteira herdada
                                                       #     é o alvo DEPOIS de render: cada posição
                                                       #     cresce com o próprio retorno, o todo com rb
        iniciado = True

    return ResultadoBacktest(
        liquido=pd.Series(liquido).sort_index(),
        bruto=pd.Series(bruto).sort_index(),
        w_vt=pd.Series(wq).sort_index(),
        turnover=pd.Series(turn).sort_index(),
        custos=pd.Series(custos).sort_index(),
    )
