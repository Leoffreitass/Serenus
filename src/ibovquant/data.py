"""Carga e preparo de dados: painel de preços, composição mensal e elegibilidade."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'

#: composição usa o ticker vigente à época; o painel de preços concentra a
#: história no código que possui a série completa.
PRICE_ALIAS = {
    'VIIA3': 'BHIA3', 'ALSO3': 'ALOS3', 'RRRP3': 'BRAV3', 'TRPL4': 'ISAE4',
    'CCRO3': 'MOTV3', 'EMBR3': 'EMBJ3', 'ELET3': 'AXIA3', 'CPLE6': 'CPLE3',
    'ARZZ3': 'AZZA3', 'MRFG3': 'MBRF3',
}


def load_painel(path: Path | str | None = None) -> pd.DataFrame:
    """Painel longo de preços: date, ticker, close, adj_close, volume, source."""
    return pd.read_parquet(path or DATA / 'painel_precos.parquet')


def adjclose_wide(painel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Matriz diária de preços ajustados (datas x tickers)."""
    painel = load_painel() if painel is None else painel
    return painel.pivot_table(index='date', columns='ticker', values='adj_close')


def monthly_closes(wide: pd.DataFrame) -> pd.DataFrame:
    """Último preço ajustado disponível em cada mês (datas viram fins de mês)."""
    return wide.resample('ME').last()


def load_composicao(path: Path | str | None = None) -> pd.DataFrame:
    """Composição mensal longa com colunas: date, ticker, ym, price_ticker."""
    comp = pd.read_csv(path or DATA / 'composicao_ibov_long.csv', parse_dates=['date'])
    comp['ym'] = comp['date'].dt.to_period('M')
    comp['price_ticker'] = comp['ticker'].map(PRICE_ALIAS).fillna(comp['ticker'])
    return comp


def eligibility(comp: pd.DataFrame, mclose: pd.DataFrame) -> pd.DataFrame:
    """Matriz booleana (fim de mês x price_ticker): membro do IBOV no mês E com
    preço disponível no mês. É o universo elegível para formação de carteira."""
    memb = (comp.drop_duplicates(['ym', 'price_ticker'])
                .assign(v=True)
                .pivot(index='ym', columns='price_ticker', values='v')
                .fillna(False))
    memb.index = memb.index.to_timestamp('M')          # PeriodIndex -> fim de mês
    memb = memb.reindex(index=mclose.index, columns=mclose.columns, fill_value=False)
    return memb & mclose.notna()
