# -*- coding: utf-8 -*-
"""
Baixa e parseia os arquivos oficiais COTAHIST da B3 (2015 -> 2026), extraindo
apenas os tickers que o Yahoo Finance nao possui (delistados/purgados).
=============================================================================
COMO USAR (na sua maquina):
    pip install pandas requests
    python download_cotahist.py

Baixa ~12 zips (60-100 MB cada) para ./cotahist_zips/ (mantidos em cache:
se rodar de novo, nao baixa novamente). Gera:
    cotahist_faltantes.csv  -> date, ticker, open, high, low, close, volume, trades
    relatorio_cotahist.txt  -> cobertura por ticker

Depois envie 'cotahist_faltantes.csv' na conversa do Claude.

Formato COTAHIST: registros de largura fixa, layout oficial da B3
(SeriesHistoricas_Layout.pdf). Precos SEM ajuste por proventos — o ajuste
sera feito depois, com a tabela de eventos societarios pesquisada a parte.
"""
import io
import os
import sys
import zipfile

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("Instale as dependencias: pip install pandas requests")

ANOS = range(2015, 2027)  # 2015..2026
URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"
PASTA_ZIPS = "cotahist_zips"

# Tickers ausentes no Yahoo (sem sufixo .SA) + janelas de interesse nao sao
# necessarias aqui: extraimos a serie inteira de cada um e filtramos depois.
TICKERS = {
    "AZUL4", "BIDI11", "BIDI4", "BPAN4", "BRFS3", "BRML3", "CIEL3", "CRFB3",
    "ELET6", "AXIA6", "ENBR3", "GETT11", "GOLL4", "HGTX3", "JBSS3", "JPSA3",
    "LCAM3", "NTCO3", "PETZ3", "SOMA3", "STBP3", "SULA11",
    # redundancias uteis para validacao cruzada com o Yahoo:
    "VVAR11", "VALE5",
}

def baixar_zip(ano):
    os.makedirs(PASTA_ZIPS, exist_ok=True)
    destino = os.path.join(PASTA_ZIPS, f"COTAHIST_A{ano}.ZIP")
    if os.path.exists(destino) and os.path.getsize(destino) > 1_000_000:
        print(f"  {ano}: ja em cache")
        return destino
    url = URL.format(ano=ano)
    print(f"  {ano}: baixando {url} ...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    with requests.get(url, headers=headers, stream=True, timeout=300, verify=False) as r:
        r.raise_for_status()
        with open(destino, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"       ok ({os.path.getsize(destino)/1e6:.0f} MB)")
    return destino

def parse_zip(caminho):
    """Le o TXT dentro do zip e retorna linhas dos tickers desejados."""
    linhas = []
    with zipfile.ZipFile(caminho) as z:
        nome_txt = z.namelist()[0]
        with z.open(nome_txt) as fh:
            for raw in io.TextIOWrapper(fh, encoding="latin-1"):
                if not raw.startswith("01"):
                    continue
                codneg = raw[12:24].strip()
                if codneg not in TICKERS:
                    continue
                tpmerc = raw[24:27]
                if tpmerc != "010":  # somente mercado a vista
                    continue
                fatcot = int(raw[210:217] or 1)  # fator de cotacao (qtd de acoes por cotacao)
                linhas.append({
                    "date": raw[2:10],
                    "ticker": codneg,
                    "open": int(raw[56:69]) / 100 / fatcot,
                    "high": int(raw[69:82]) / 100 / fatcot,
                    "low": int(raw[82:95]) / 100 / fatcot,
                    "close": int(raw[108:121]) / 100 / fatcot,
                    "volume": int(raw[170:188]) / 100,
                    "trades": int(raw[147:152]),
                })
    return linhas

def main():
    # o certificado da B3 as vezes falha em Windows; suprimimos o aviso do verify=False
    import urllib3
    urllib3.disable_warnings()

    todas = []
    print("Baixando/parseando COTAHIST:")
    for ano in ANOS:
        try:
            caminho = baixar_zip(ano)
        except Exception as e:
            print(f"  {ano}: ERRO no download ({e}) — pulando")
            continue
        linhas = parse_zip(caminho)
        print(f"       {len(linhas)} registros dos tickers-alvo")
        todas.extend(linhas)

    if not todas:
        sys.exit("Nenhum registro extraido — verifique os downloads.")

    df = pd.DataFrame(todas)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_csv("cotahist_faltantes.csv", index=False)

    with open("relatorio_cotahist.txt", "w", encoding="utf-8") as f:
        cov = df.groupby("ticker")["date"].agg(["min", "max", "count"])
        f.write("Cobertura por ticker (precos BRUTOS, sem ajuste):\n")
        f.write(cov.to_string())
        faltando = sorted(TICKERS - set(df["ticker"].unique()))
        f.write(f"\n\nTickers sem nenhum registro: {faltando}\n")

    print(f"\nConcluido: {len(df)} linhas, {df['ticker'].nunique()} tickers")
    print("-> cotahist_faltantes.csv  (envie este arquivo na conversa do Claude)")
    print("-> relatorio_cotahist.txt")

if __name__ == "__main__":
    main()
