# -*- coding: utf-8 -*-
"""Monta o painel final de preços ajustados do universo IBOV (2015 -> 2026).

1. Ajusta os preços brutos do COTAHIST (tickers delistados) por eventos de
   capital (splits, grupamentos, bonificações) e proventos (div/JCP), na mesma
   convenção de back-adjustment do Yahoo Finance:
       adj_t = close_t * prod{ f_e : eventos e com data > t }
       split/bonificação com fator r  -> f = 1/r
       provento D com ex em d        -> f = 1 - D / close(último pregão < d)
2. Faz o splice de cadeias de ticker que atravessam as duas fontes
   (NTCO3 <- NATU3-antiga; ELET6+AXIA6).
3. Funde com o painel do Yahoo e valida cobertura vs composição mensal.

Saídas: data/processed/painel_precos.parquet (longo) e
        data/processed/painel_adjclose_wide.csv (datas x tickers, ajustado).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = PROC = ROOT / 'data'  # distribuição enxuta: pasta única

# ---------------------------------------------------------------- carga
yahoo = pd.read_parquet(RAW / 'precos_ibov.parquet')
yahoo['ticker'] = yahoo['ticker'].str.replace('.SA', '', regex=False).replace({'^BVSP': 'IBOV'})
b3 = pd.read_csv(RAW / 'cotahist_faltantes.csv', parse_dates=['date'])

events = []
for lote in (1, 2):
    d = json.load(open(PROC / f'eventos_societarios_lote{lote}.json'))
    events += d['events']
ev = pd.DataFrame(events)
ev['date'] = pd.to_datetime(ev['date'])

CAPITAL = {'split', 'inplit', 'bonificacao'}
CASH = {'dividendo', 'jcp', 'dividendo_anual'}

# tickers cuja série virá do COTAHIST (Yahoo não tem / tem só fragmento)
ALVOS = ['AZUL4', 'BIDI11', 'BIDI4', 'BPAN4', 'BRFS3', 'BRML3', 'CIEL3', 'CRFB3',
         'ENBR3', 'GETT11', 'GOLL4', 'HGTX3', 'JBSS3', 'JPSA3', 'LCAM3', 'NTCO3',
         'PETZ3', 'SOMA3', 'STBP3', 'SULA11', 'ELET6', 'AXIA6']

# ------------------------------------------------- ajuste por eventos

def ajusta(serie_close, eventos, contexto=''):
    """serie_close: Series indexada por data (raw). Retorna adj_close e log."""
    fatores = []  # (data_ex, fator_preco)
    log = []
    for _, e in eventos.iterrows():
        d = e['date']
        if not (serie_close.index.min() < d <= serie_close.index.max()):
            continue  # eventos após o último pregão (ex.: conversão em BDR) não afetam retornos
        if e['type'] in CAPITAL:
            r = float(e.get('factor') or 1.0)
            if e['ticker'] == 'AXIA6' and abs(r - 1.0) < 1e-9:
                # bonificação em OUTRA classe (AXIA7): tratar como provento em valor
                axia7 = yahoo[(yahoo.ticker == 'AXIA7')].set_index('date')['close'].sort_index()
                if len(axia7) == 0:
                    continue
                D = 0.2628378881074 * float(axia7.iloc[0])
                prev = serie_close[serie_close.index < d]
                if len(prev) == 0:
                    continue
                f = 1 - D / float(prev.iloc[-1])
                log.append(f'{d.date()} bonif. AXIA7 valor≈{D:.2f} f={f:.4f}')
            else:
                f = 1.0 / r
                log.append(f'{d.date()} {e["type"]} r={r} f={f:.4f}')
        elif e['type'] in CASH:
            D = float(e.get('value_per_share') or 0)
            if D <= 0:
                continue
            prev = serie_close[serie_close.index < d]
            if len(prev) == 0:
                continue
            f = 1 - D / float(prev.iloc[-1])
            if not (0.5 < f <= 1.0):
                log.append(f'{d.date()} PROVENTO SUSPEITO D={D} f={f:.3f} — IGNORADO')
                continue
        else:
            continue
        fatores.append((d, f))
    # produto acumulado dos fatores FUTUROS a cada data
    adj = serie_close.astype(float).copy()
    for d, f in fatores:
        adj[adj.index < d] *= f
    return adj, log

print('=== Ajustando séries COTAHIST ===')
ajustadas, logs = {}, {}

# cadeia ELET6 + AXIA6 (renomeação em 10/11/2025; mesma empresa/classe)
chain = pd.concat([
    b3[b3.ticker == 'ELET6'].set_index('date')['close'],
    b3[b3.ticker == 'AXIA6'].set_index('date')['close'],
]).sort_index()
ev_chain = ev[ev.ticker.isin(['ELET6', 'AXIA6'])]
adj, lg = ajusta(chain, ev_chain, 'ELET6/AXIA6')
ajustadas['ELET6'] = adj
ajustadas['AXIA6'] = adj
logs['ELET6+AXIA6'] = lg

for tk in ALVOS:
    if tk in ('ELET6', 'AXIA6'):
        continue
    s = b3[b3.ticker == tk].set_index('date')['close'].sort_index()
    adj, lg = ajusta(s, ev[ev.ticker == tk], tk)
    ajustadas[tk] = adj
    logs[tk] = lg

# cadeia NTCO3: prefixa com NATU3-antiga do Yahoo (troca 1:1 em dez/2019)
natu_old = yahoo[(yahoo.ticker == 'NATU3') & (yahoo.date < '2019-12-18')].set_index('date')['adj_close'].sort_index()
ntco = ajustadas['NTCO3']
overlap = natu_old.index.intersection(ntco.index)
if len(overlap) > 0:
    escala = float(ntco[overlap].mean() / natu_old[overlap].mean())
else:  # emenda pelo primeiro ponto
    escala = float(ntco.iloc[0] / natu_old.iloc[-1])
natu_scaled = (natu_old * escala)[~natu_old.index.isin(ntco.index)]
ajustadas['NTCO3'] = pd.concat([natu_scaled, ntco]).sort_index()
print(f'NTCO3: emenda com NATU3-antiga, escala {escala:.4f}, overlap {len(overlap)} pregões')

for k in sorted(logs):
    if logs[k]:
        print(f'  {k}: ' + '; '.join(logs[k]))

# ------------------------------------------------- montagem do painel longo
frames = [yahoo[['date', 'ticker', 'close', 'adj_close', 'volume']].assign(source='yahoo')]
for tk, adj in ajustadas.items():
    raw = pd.concat([b3[b3.ticker.isin(['ELET6', 'AXIA6'])] if tk in ('ELET6', 'AXIA6')
                     else b3[b3.ticker == tk]]).set_index('date')['close'].sort_index()
    raw = raw[~raw.index.duplicated()]
    dfx = pd.DataFrame({'date': adj.index, 'ticker': tk,
                        'close': raw.reindex(adj.index).values,
                        'adj_close': adj.values, 'volume': np.nan, 'source': 'b3_ajustado'})
    frames.append(dfx)

painel = pd.concat(frames, ignore_index=True)
# em duplicidade (date,ticker), preferir yahoo
painel = (painel.sort_values(['ticker', 'date', 'source'])
                .drop_duplicates(['ticker', 'date'], keep='last'))  # b3_ajustado < yahoo alfabeticamente -> keep last = yahoo
painel = painel.sort_values(['ticker', 'date']).reset_index(drop=True)
painel.to_parquet(PROC / 'painel_precos.parquet', index=False)

wide = painel.pivot_table(index='date', columns='ticker', values='adj_close')
wide.to_csv(PROC / 'painel_adjclose_wide.csv')
print(f'\nPainel: {len(painel)} linhas, {painel.ticker.nunique()} séries, '
      f'{painel.date.min().date()} -> {painel.date.max().date()}')

# ------------------------------------------------- validação de cobertura
comp = pd.read_csv(PROC / 'composicao_ibov_long.csv', parse_dates=['date'])
comp['ym'] = comp['date'].dt.to_period('M')
ALIAS = {'VIIA3': 'BHIA3', 'ALSO3': 'ALOS3', 'RRRP3': 'BRAV3', 'TRPL4': 'ISAE4',
         'CCRO3': 'MOTV3', 'EMBR3': 'EMBJ3', 'ELET3': 'AXIA3', 'CPLE6': 'CPLE3',
         'ARZZ3': 'AZZA3', 'MRFG3': 'MBRF3'}
comp['pt'] = comp['ticker'].map(ALIAS).fillna(comp['ticker'])
painel['ym'] = painel['date'].dt.to_period('M')
tem = set(zip(painel['ticker'], painel['ym']))
comp['ok'] = [(t, m) in tem for t, m in zip(comp['pt'], comp['ym'])]
print(f'\n=== Cobertura final: {comp.ok.mean():.2%} dos membro-meses ===')
miss = comp[~comp.ok].groupby('ticker')['ym'].agg(['min', 'max', 'count'])
if len(miss):
    print(miss.to_string())

# ------------------------------------------------- sanity: retornos extremos
print('\n=== Sanity check: maiores |retornos| diários nas séries ajustadas ===')
for tk in ['CIEL3', 'JBSS3', 'BRFS3', 'SULA11', 'BIDI4', 'LCAM3', 'ELET6']:
    s = painel[painel.ticker == tk].set_index('date')['adj_close'].sort_index()
    r = s.pct_change().dropna()
    top = r.abs().nlargest(3)
    print(f'  {tk}: ' + ', '.join(f'{d.date()}: {r[d]:+.1%}' for d in top.index))
