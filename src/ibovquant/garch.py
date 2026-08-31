"""Previsão de volatilidade GARCH(1,1) via MLE (pacote `arch`).

Modelo (Bollerslev, 1986):
    r_t = μ + ε_t,  ε_t = σ_t z_t,  z_t ~ Student-t(ν)
    σ²_t = ω + α ε²_{t-1} + β σ²_{t-1},   estacionário se α + β < 1

Para o vol targeting mensal usamos a previsão AGREGADA de h=21 pregões:
    σ̂²_{t,t+h} = Σ_{k=1..h} σ̂²_{t+k|t}   (a soma das variâncias multi-passo,
que revertem geometricamente à variância incondicional ω/(1-α-β)), anualizada
por 252/h. Baselines: vol realizada em janela móvel e EWMA/RiskMetrics λ=0,94.

Ref.: Engle (1982); Bollerslev (1986); Hansen & Lunde (2005).
Ver docs/research/vt_garch_notes.md.
"""
import numpy as np
import pandas as pd
from arch import arch_model

ANN = 252
SCALE = 100.0  # retornos em % para estabilidade numérica do MLE


def fit_garch(ret_diario: pd.Series, dist: str = 'studentst'):
    """Ajusta GARCH(1,1) por MLE. `ret_diario` em fração (0.01 = 1%).

    Pseudocódigo:
        função FIT_GARCH(r):
            r ← arredondar(r, 10 casas)      # fit bit-reprodutível
            candidatos ← { fit com starting values default,
                           fits com grade fixa de (α₀, β₀),
                           fit Student-t SEMEADO pelo fit Normal }
            retornar candidato de MAIOR log-verossimilhança
            # multi-start determinístico: escapa de ótimos locais
    """
    import warnings
    # Quantiza a entrada a 10 casas decimais: o MLE amplifica ruído de
    # ponto flutuante de ~1e-16 (ordem de operações varia entre execuções)
    # em ótimos locais diferentes nas janelas difíceis. O arredondamento
    # está muitas ordens de grandeza abaixo de qualquer significado
    # econômico e torna o fit bit-reprodutível entre execuções/máquinas.
    y = ret_diario.dropna().round(10) * SCALE
    with warnings.catch_warnings():
        # Janelas difíceis emitem ConvergenceWarning do SLSQP; a seleção por
        # log-verossimilhança abaixo cuida disso (um fit "code 8" pode ser o
        # ótimo global — não descartamos por warning, e sim por LL).
        # Estimação MULTI-START determinística: o fit direto, um fit Student-t
        # SEMEADO pelos parâmetros do fit Normal e uma grade fixa de starting
        # values plausíveis (α, β); vence o de maior log-verossimilhança.
        # Motivo: os starting values default podem cair em ótimos locais que
        # subestimam drasticamente a vol prevista, e o ótimo local atingido
        # pode variar entre máquinas/BLAS — a grade fixa torna a seleção
        # estável entre ambientes.
        warnings.simplefilter('ignore')
        am = arch_model(y, mean='Constant', vol='GARCH', p=1, q=1, dist=dist)
        candidatos = []
        try:
            candidatos.append(am.fit(disp='off', show_warning=False))
        except Exception:
            pass
        mu0, var0 = float(y.mean()), float(y.var())
        for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.03, 0.94)):
            sv = [mu0, var0 * (1 - a0 - b0), a0, b0]
            if dist == 'studentst':
                sv.append(8.0)
            try:
                candidatos.append(am.fit(disp='off', show_warning=False,
                                         starting_values=np.array(sv),
                                         options={'maxiter': 1000}))
            except Exception:
                pass
        if dist == 'studentst':
            try:
                res_n = arch_model(y, mean='Constant', vol='GARCH',
                                   p=1, q=1, dist='normal').fit(disp='off', show_warning=False)
                sv = np.append(res_n.params.values, 8.0)  # + nu inicial
                candidatos.append(am.fit(disp='off', show_warning=False, starting_values=sv,
                                         options={'maxiter': 1000}))
            except Exception:
                pass
        if not candidatos:
            raise RuntimeError('nenhum fit GARCH convergiu')
        return max(candidatos, key=lambda r: r.loglikelihood)


def params(res) -> dict:
    """ω, α, β (e ν se Student-t), persistência e vol incondicional anualizada."""
    p = res.params
    omega, alpha, beta = p['omega'], p['alpha[1]'], p['beta[1]']
    pers = alpha + beta
    uncond = np.sqrt(omega / (1 - pers) * ANN) / SCALE if pers < 1 else np.nan
    out = {'omega': omega, 'alpha': alpha, 'beta': beta,
           'persistencia': pers, 'vol_incond_aa': uncond}
    if 'nu' in p:
        out['nu'] = p['nu']
    return out


def forecast_vol_aa(res, horizon: int = 21) -> float:
    """Vol anualizada prevista para os próximos `horizon` pregões (agregada).

    Pseudocódigo:
        função PREV_VOL(fit, h=21):
            σ²[k] ← E_t[σ²_{t+k}], k = 1..h    # forma fechada: reversão
                                               # geométrica à média
            retornar sqrt( média(σ²) · 252 )   # vol anualizada do bloco
    """
    f = res.forecast(horizon=horizon, reindex=False)
    var_h = f.variance.values[0]              # σ²_{t+1..t+h}, em %²/dia
    return float(np.sqrt(var_h.sum() / horizon * ANN) / SCALE)


def rolling_garch(ret_diario: pd.Series, datas: pd.DatetimeIndex,
                  horizon: int = 21, window: int = 1008, min_obs: int = 504,
                  dist: str = 'studentst') -> pd.Series:
    """Previsões out-of-sample: em cada data (fim de mês), ajusta o modelo só
    com dados até a data e prevê a vol anualizada dos próximos `horizon` dias.

    Pseudocódigo:
        função ROLLING_GARCH(r diário, datas):
            para cada fim de mês t em datas:
                amostra ← últimos 1008 pregões ATÉ t (mín. 504)
                prev[t] ← PREV_VOL(FIT_GARCH(amostra), h=21)
            # sem look-ahead: só dados até t entram no fit de t
    """
    r = ret_diario.dropna()
    out = {}
    for dt in datas:
        amostra = r.loc[:dt].tail(window)
        if len(amostra) < min_obs:
            continue
        try:
            res = fit_garch(amostra, dist=dist)
            out[dt] = forecast_vol_aa(res, horizon)
        except Exception:
            continue
    return pd.Series(out, name='garch')


# ------------------------------------------------------------- baselines

def vol_realizada(ret_diario: pd.Series, window: int = 21) -> pd.Series:
    """Vol realizada anualizada em janela móvel (estimador ingênuo)."""
    return ret_diario.rolling(window).std() * np.sqrt(ANN)


def vol_ewma(ret_diario: pd.Series, lam: float = 0.94) -> pd.Series:
    """EWMA/RiskMetrics: σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}, anualizada."""
    var = ret_diario.dropna().pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * ANN)


def vol_realizada_futura(ret_diario: pd.Series, datas: pd.DatetimeIndex,
                         horizon: int = 21) -> pd.Series:
    """Alvo da avaliação: vol realizada nos `horizon` pregões SEGUINTES a cada data."""
    r = ret_diario.dropna()
    out = {}
    for dt in datas:
        fut = r.loc[r.index > dt].head(horizon)
        if len(fut) >= int(horizon * 0.7):
            out[dt] = float(fut.std() * np.sqrt(ANN))
    return pd.Series(out, name='realizada_fwd')


# ------------------------------------------------------------- avaliação

def qlike(prev: pd.Series, real: pd.Series) -> float:
    """Perda QLIKE média (robusta p/ proxy de variância; menor = melhor)."""
    df = pd.concat([prev, real], axis=1, keys=['f', 'r']).dropna()
    f2, r2 = df.f ** 2, df.r ** 2
    return float((np.log(f2) + r2 / f2).mean())


def mincer_zarnowitz(prev: pd.Series, real: pd.Series) -> dict:
    """Regressão real = a + b·prev: ideal a=0, b=1; devolve R² também."""
    df = pd.concat([prev, real], axis=1, keys=['f', 'r']).dropna()
    b, a = np.polyfit(df.f, df.r, 1)
    r2 = np.corrcoef(df.f, df.r)[0, 1] ** 2
    return {'alpha': float(a), 'beta': float(b), 'r2': float(r2), 'n': len(df)}


def avalia(previsoes: dict[str, pd.Series], real: pd.Series) -> pd.DataFrame:
    """Tabela comparativa: RMSE, QLIKE e Mincer-Zarnowitz por previsor.

    A comparação usa a AMOSTRA COMUM (datas em que todos os previsores e o
    alvo existem) — previsores com exigência de histórico maior (GARCH)
    começam depois, e métricas em amostras diferentes não são comparáveis.
    """
    base = pd.concat({**previsoes, '_real': real}, axis=1).dropna()
    linhas = {}
    for nome in previsoes:
        f, r = base[nome], base['_real']
        b, a = np.polyfit(f, r, 1)
        linhas[nome] = {
            'RMSE (p.p.)': float(np.sqrt(((f - r) ** 2).mean()) * 100),
            'QLIKE': float((np.log(f ** 2) + (r ** 2) / (f ** 2)).mean()),
            'MZ alpha': float(a), 'MZ beta': float(b),
            'MZ R²': float(np.corrcoef(f, r)[0, 1] ** 2), 'n': len(base),
        }
    return pd.DataFrame(linhas).T
