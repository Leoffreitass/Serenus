"""[Proveniência] Este script documenta a construção da base; seus insumos (planilha original e pesquisa de composições) não são versionados nesta distribuição — os dados finais já acompanham o repositório.


Constrói a matriz mensal de composição do IBOVESPA estendida: dez/2022 -> jun/2026.
Lógica: para cada mês-fim, parte da carteira quadrimestral vigente e aplica os
eventos extraordinários com data <= mês, dentro da janela.
Valida a consistência: (janela N ajustada por eventos) + adds - removes == janela N+1.
"""
import json
import re
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / 'data'
SRC_XLSX = REPO / 'data' / 'composicao_ibov_original.xlsx'  # insumo não versionado
OUT_XLSX = DATA / 'composicao_ibov_estendida.xlsx'
OUT_LONG = DATA / 'composicao_ibov_long.csv'
OUT_TKMAP = DATA / 'ticker_changes.csv'

research = json.load(open(DATA / 'research_composicoes_2023_2026.json'))

# ---------- original file ----------
orig = pd.read_excel(SRC_XLSX, sheet_name='comp_historica', header=None)
orig_dates = pd.to_datetime(orig.iloc[0])
print(f'Original: {orig_dates.iloc[0].date()} -> {orig_dates.iloc[-1].date()} ({orig.shape[1]} cols)')

VALID = re.compile(r'^[A-Z0-9]{4}\d{1,2}$')

def clean_new_ticker(s):
    if not s:
        return None
    tok = str(s).split()[0].strip()
    return tok if VALID.match(tok) else None

# ---------- windows ----------
windows = []  # (start Timestamp, set(tickers))
baseline = set(research['baseline_2022_09'])
windows.append((pd.Timestamp('2022-09-01'), baseline))
for w in research['windows']:
    windows.append((pd.Timestamp(w['effective'] + '-01'), set(w['tickers'])))

# ---------- events ----------
events = []
for e in research['extraordinary_events']:
    em = pd.Timestamp(e['date'] + '-01')
    etype = e['event']
    tk = e['ticker']
    new = clean_new_ticker(e.get('new_ticker'))
    events.append({'month': em, 'type': etype, 'ticker': tk, 'new': new, 'detail': e.get('detail', '')})
events_df = pd.DataFrame(events).sort_values('month')

REMOVE_TYPES = {'removal', 'delisting', 'merger'}
ADD_TYPES = {'spinoff_inclusion', 'bonus_class_inclusion'}

def window_at(month_end):
    cur = windows[0]
    for start, ticks in windows:
        if start <= month_end:
            cur = (start, ticks)
    return cur

def composition_at(month_end):
    """Carteira no fechamento do mês: janela vigente + eventos da janela com data <= mês."""
    wstart, ticks = window_at(month_end)
    comp = set(ticks)
    applied = []
    for _, ev in events_df.iterrows():
        if not (wstart <= ev['month'] <= month_end):
            continue
        t, typ, new = ev['ticker'], ev['type'], ev['new']
        if typ == 'ticker_change':
            if t in comp:
                comp.discard(t)
                comp.add(new)
                applied.append(f"{t}->{new}")
        elif typ in REMOVE_TYPES:
            if t in comp:
                comp.discard(t)
                applied.append(f"-{t}")
            # fusão que cria papel incluído no índice (AZZA3, MBRF3)
            if typ == 'merger' and new and 'fora do indice' not in ev['detail'].lower():
                if new not in comp:
                    comp.add(new)
                    applied.append(f"+{new}")
        elif typ in ADD_TYPES:
            if new and new not in comp:
                comp.add(new)
                applied.append(f"+{new}")
    return comp, applied

# ---------- consistency validation between windows ----------
print('\n=== Validação janela->janela ===')
research_windows = research['windows']
prev_start, prev_set = windows[0][0], set(windows[0][1])
ok = True
for i, w in enumerate(research['windows']):
    wstart = pd.Timestamp(w['effective'] + '-01')
    # aplica eventos da janela anterior até o mês anterior ao início desta
    last_me = wstart - pd.offsets.MonthEnd(1)
    derived, applied = composition_at(last_me)
    expected = derived | set(w['adds']) if False else (derived - set(w.get('removes', []))) | set(w.get('adds', []))
    actual = set(w['tickers'])
    diff_plus = actual - expected
    diff_minus = expected - actual
    flag = 'OK' if not diff_plus and not diff_minus else 'DIVERGE'
    if flag != 'OK':
        ok = False
        print(f"{w['effective']}: {flag} | faltam_no_derivado={sorted(diff_plus)} sobram={sorted(diff_minus)} | eventos aplicados antes: {applied}")
    else:
        print(f"{w['effective']}: OK (n={len(actual)})")
print('Cadeia consistente!' if ok else 'ATENÇÃO: divergências acima.')

# ---------- build monthly columns Dec/2022 -> Jun/2026 ----------
new_months = pd.date_range('2022-12-31', '2026-06-30', freq='ME')
cols = {}
print('\n=== Colunas novas ===')
for me in new_months:
    comp, applied = composition_at(me)
    cols[me] = sorted(comp)
    note = f" [{', '.join(applied)}]" if applied else ''
    print(f"{me.date()}: {len(comp)} tickers{note}")

# ---------- assemble extended matrix in the original format ----------
max_len = max(orig.shape[0] - 1, max(len(v) for v in cols.values()))
ext = orig.copy()
for me, ticks in cols.items():
    col = [me] + ticks + [None] * (max_len - len(ticks))
    ext[ext.shape[1]] = pd.Series(col)
# universo total (para lista_acoes) — tudo que já apareceu em qualquer coluna
universe = set()
for c in range(ext.shape[1]):
    universe |= set(x for x in ext.iloc[1:, c].dropna().astype(str) if VALID.match(x))
lista = pd.DataFrame({'tickers': sorted(t + '.SA' for t in universe)})

with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as xw:
    ext.to_excel(xw, sheet_name='comp_historica', header=False, index=False)
    lista.to_excel(xw, sheet_name='lista_acoes', index=False)

# ---------- tidy long format (dedup do original incluído) ----------
rows = []
for c in range(ext.shape[1]):
    d = pd.to_datetime(ext.iloc[0, c])
    ticks = sorted(set(x for x in ext.iloc[1:, c].dropna().astype(str) if VALID.match(x)))
    rows += [{'date': d.date(), 'ticker': t} for t in ticks]
long_df = pd.DataFrame(rows)
long_df.to_csv(OUT_LONG, index=False)

# ---------- ticker change map (para emendar séries de preço) ----------
tk_rows = [{'month': ev['month'].strftime('%Y-%m'), 'old': ev['ticker'], 'new': ev['new'], 'type': ev['type'], 'detail': ev['detail'][:120]}
           for ev in events if ev['type'] == 'ticker_change' and ev['new']]
pd.DataFrame(tk_rows).to_csv(OUT_TKMAP, index=False)

print(f"\nSalvos:\n {OUT_XLSX}\n {OUT_LONG} ({len(long_df)} linhas, {long_df.date.nunique()} datas)\n {OUT_TKMAP} ({len(tk_rows)} trocas)")
print(f"Universo total: {len(universe)} tickers")
