"""Hierarchical Risk Parity (López de Prado, 2016).

Os três passos do paper (ver docs/research/hrp_notes.md):
  1. clustering hierárquico com a distância de correlação d = sqrt((1-ρ)/2);
  2. quasi-diagonalização — reordenação da covariância pela ordem das folhas
     do dendrograma (seriation);
  3. bisseção recursiva — divide a lista seriada ao meio e aloca entre as
     metades α = 1 − V₁/(V₁+V₂), com variância de cluster calculada com pesos
     inversos à variância (IVP) dentro de cada metade.

Pesos resultantes são não-negativos e somam 1 (long-only por construção).
"""
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return pd.DataFrame(np.clip(corr, -1, 1), index=cov.index, columns=cov.columns)


def corr_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """d_ij = sqrt((1 - ρ_ij)/2) — métrica de distância própria (LdP 2016)."""
    return np.sqrt(0.5 * (1.0 - corr)).fillna(0.0)


def linkage_matrix(cov: pd.DataFrame, method: str = 'single') -> np.ndarray:
    dist = corr_distance(cov_to_corr(cov))
    condensed = squareform(dist.values, checks=False)
    return sch.linkage(condensed, method=method)


def quasi_diag_order(link: np.ndarray) -> list[int]:
    """Ordem das folhas do dendrograma (seriation)."""
    return list(sch.leaves_list(link))


def ivp_weights(cov: pd.DataFrame) -> pd.Series:
    """Inverse-variance portfolio (baseline do paper)."""
    iv = 1.0 / np.diag(cov)
    return pd.Series(iv / iv.sum(), index=cov.index)


def _cluster_var(cov: pd.DataFrame, itens: list) -> float:
    sub = cov.loc[itens, itens]
    w = ivp_weights(sub).values
    return float(w @ sub.values @ w)


def hrp_weights(cov: pd.DataFrame, method: str = 'single') -> pd.Series:
    """Pesos HRP para uma matriz de covariância (tickers no índice/colunas).

    Pseudocódigo (López de Prado, 2016):
        função HRP(Σ):
            ρ ← correlação(Σ);  D ← sqrt((1−ρ)/2)          # (a) clustering
            link ← linkage(D, 'single')
            ordem ← folhas_do_dendrograma(link)             # (b) quasi-diag
            w ← vetor de 1s;  pilha ← [ordem]               # (c) bisseção
            enquanto houver cluster com len > 1:
                (L1, L2) ← dividir a lista seriada ao meio
                V_j ← w̃_j' Σ_j w̃_j,  com w̃_j ∝ 1/diag(Σ_j)  # var. IVP
                α ← 1 − V1/(V1+V2)
                w[L1] ·= α;  w[L2] ·= 1−α
            retornar w        # soma 1, todos ≥ 0
    """
    if len(cov) == 1:
        return pd.Series([1.0], index=cov.index)
    link = linkage_matrix(cov, method=method)
    ordem = [cov.index[i] for i in quasi_diag_order(link)]

    w = pd.Series(1.0, index=ordem)
    clusters = [ordem]
    while clusters:
        # divide cada cluster ao meio (bisseção sobre a lista seriada)
        clusters = [c[i:j] for c in clusters
                    for i, j in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for k in range(0, len(clusters), 2):
            c1, c2 = clusters[k], clusters[k + 1]
            v1, v2 = _cluster_var(cov, c1), _cluster_var(cov, c2)
            alpha = 1.0 - v1 / (v1 + v2)
            w[c1] *= alpha
            w[c2] *= 1.0 - alpha
    return w.reindex(cov.index)


# ------------------------------------------------------------------ pipeline

def pesos_mensais(holdings: pd.DataFrame, wide_daily: pd.DataFrame,
                  metodo: str = 'hrp', window: int = 252, min_obs: int = 126,
                  linkage: str = 'single') -> pd.DataFrame:
    """Para cada data de formação, estima a covariância na janela de `window`
    pregões e calcula os pesos da cesta selecionada.

    metodo: 'hrp' | 'ivp' | 'ew'. Tickers com menos de `min_obs` observações
    na janela são excluídos da cesta naquele mês (pesos renormalizados).

    Pseudocódigo:
        função PESOS_MENSAIS(carteiras, P diário, método):
            para cada formação t:
                R ← retornos diários da cesta nos últimos 252 pregões ATÉ t
                descartar papéis com < min_obs observações na janela
                Σ ← cov(R) · 252
                w[t] ← EW  |  IVP(Σ)  |  HRP(Σ)     # conforme `método`
    """
    linhas = {}
    for dt in holdings.index:
        cesta = list(holdings.columns[holdings.loc[dt]])
        rets = (wide_daily.loc[:dt, cesta].tail(window + 1)
                .pct_change(fill_method=None).iloc[1:])
        ok = [t for t in cesta if rets[t].notna().sum() >= min_obs]
        if len(ok) < 2:
            continue
        r = rets[ok].dropna(how='all')
        if metodo == 'ew':
            w = pd.Series(1.0 / len(ok), index=ok)
        else:
            cov = r.cov() * 252.0
            w = hrp_weights(cov, method=linkage) if metodo == 'hrp' else ivp_weights(cov)
        linhas[dt] = w
    pesos = pd.DataFrame(linhas).T.reindex(columns=holdings.columns)
    return pesos.fillna(0.0)


def retorno_carteira(pesos: pd.DataFrame, mclose: pd.DataFrame) -> pd.Series:
    """Retorno mensal da carteira com pesos formados no fim do mês m e
    carregados no mês m+1 (papéis sem preço são excluídos com renormalização)."""
    mret = mclose.pct_change(fill_method=None)
    out = {}
    for dt in pesos.index:
        w = pesos.loc[dt]
        w = w[w > 0]
        nxt = mret.index[mret.index > dt]
        if len(nxt) == 0:
            continue
        r = mret.loc[nxt[0], w.index]
        m = r.notna()
        if m.sum() == 0:
            continue
        ww = w[m] / w[m].sum()
        out[nxt[0]] = float((ww * r[m]).sum())
    return pd.Series(out)


def retorno_diario_carteira(pesos: pd.DataFrame, wide_daily: pd.DataFrame) -> pd.Series:
    """Retorno DIÁRIO da carteira: pesos formados no fim do mês m valem para
    todos os pregões do mês m+1 (pesos fixos intra-mês; papéis sem preço no dia
    são excluídos com renormalização). Insumo do vol targeting/GARCH.

    Pseudocódigo:
        função RET_DIÁRIO(pesos, P diário):
            para cada bloco entre formações consecutivas (t, t+1]:
                em cada dia d do bloco:
                    w_d ← pesos de t renormalizados entre papéis
                          com preço em d
                    retorno[d] ← Σ_i w_d[i] · r_d[i]
    """
    dret = wide_daily.pct_change(fill_method=None)
    datas_form = list(pesos.index)
    out = []
    for i, dt in enumerate(datas_form):
        fim = datas_form[i + 1] if i + 1 < len(datas_form) else dret.index[-1]
        w = pesos.loc[dt]
        w = w[w > 0]
        bloco = dret.loc[(dret.index > dt) & (dret.index <= fim), w.index]
        if bloco.empty:
            continue
        ww = bloco.notna().mul(w, axis=1)
        ww = ww.div(ww.sum(axis=1), axis=0)
        out.append((ww * bloco.fillna(0.0)).sum(axis=1))
    return pd.concat(out).sort_index()


def hhi(pesos: pd.DataFrame) -> pd.Series:
    """Índice de concentração de Herfindahl por mês (1/N = mínimo)."""
    return (pesos ** 2).sum(axis=1)
