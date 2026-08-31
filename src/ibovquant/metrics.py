"""Métricas de avaliação de estratégias (frequência mensal)."""
import numpy as np
import pandas as pd


def _stats(r: pd.Series, cdi: pd.Series | None = None,
           bench: pd.Series | None = None) -> dict:
    r = r.dropna()
    anos = len(r) / 12.0
    cagr = (1 + r).prod() ** (1 / anos) - 1
    vol = r.std() * np.sqrt(12)
    curva = (1 + r).cumprod()
    dd = curva / curva.cummax() - 1
    mdd = dd.min()
    downside = r[r < 0].std() * np.sqrt(12)

    out = {
        'CAGR': cagr, 'Vol a.a.': vol,
        'Sortino': cagr / downside if downside > 0 else np.nan,
        'Máx. DD': mdd,
        'Calmar': cagr / abs(mdd) if mdd < 0 else np.nan,
        '% meses +': (r > 0).mean(),
    }
    if cdi is not None:
        ex = (r - cdi.reindex(r.index)).dropna()
        er_aa = (1 + ex).prod() ** (1 / (len(ex) / 12)) - 1
        out['Sharpe (vs CDI)'] = er_aa / (ex.std() * np.sqrt(12))
    else:
        out['Sharpe (vs CDI)'] = np.nan
    if bench is not None:
        db = pd.concat([r, bench.reindex(r.index)], axis=1, keys=['r', 'b']).dropna()
        beta = db.r.cov(db.b) / db.b.var()
        alpha_m = db.r.mean() - beta * db.b.mean()
        out['Beta (IBOV)'] = beta
        out['Alfa a.a.'] = (1 + alpha_m) ** 12 - 1
    return out


def tabela(retornos: dict[str, pd.Series], cdi: pd.Series | None = None,
           bench: pd.Series | None = None) -> pd.DataFrame:
    """Tabela comparativa formatada de métricas (linhas = estratégias)."""
    raw = pd.DataFrame({k: _stats(v, cdi, bench) for k, v in retornos.items()}).T
    fmt = raw.copy()
    for c in ['CAGR', 'Vol a.a.', 'Máx. DD', '% meses +', 'Alfa a.a.']:
        if c in fmt:
            fmt[c] = raw[c].map(lambda v: f'{v:.1%}' if pd.notna(v) else '—')
    for c in ['Sortino', 'Calmar', 'Sharpe (vs CDI)', 'Beta (IBOV)']:
        if c in fmt:
            fmt[c] = raw[c].map(lambda v: f'{v:.2f}' if pd.notna(v) else '—')
    return fmt


def drawdown(r: pd.Series) -> pd.Series:
    curva = (1 + r.dropna()).cumprod()
    return curva / curva.cummax() - 1
