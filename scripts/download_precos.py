# -*- coding: utf-8 -*-
"""
Download de precos para o universo IBOVESPA (2015 -> hoje) via Yahoo Finance.
=============================================================================
COMO USAR (na sua maquina, com internet livre):
    pip install yfinance pandas pyarrow
    python download_precos.py

Gera na mesma pasta:
    precos_ibov.parquet   -> formato longo: date, ticker, open, high, low, close, adj_close, volume
    precos_ibov_wide.csv  -> matriz de Adj Close (datas x tickers), inclui ^BVSP
    relatorio_download.txt-> tickers OK, falhas e cobertura de datas

Depois envie o arquivo 'precos_ibov.parquet' (ou o .csv) de volta na conversa do Claude.
"""
import sys
import time
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Instale as dependencias: pip install yfinance pandas pyarrow")

START = "2015-01-01"  # 1 ano antes de dez/2015 para as janelas de estimação (vol 252d)

UNIVERSO = [
    "ABEV3.SA","ALOS3.SA","ALPA4.SA","ALSO3.SA","AMER3.SA","AMOB3.SA","ARZZ3.SA","ASAI3.SA",
    "AURE3.SA","AXIA3.SA","AXIA6.SA","AXIA7.SA","AZUL4.SA","AZZA3.SA","B3SA3.SA","BBAS3.SA",
    "BBDC3.SA","BBDC4.SA","BBSE3.SA","BEEF3.SA","BHIA3.SA","BIDI11.SA","BIDI4.SA","BPAC11.SA",
    "BPAN4.SA","BRAP4.SA","BRAV3.SA","BRFS3.SA","BRKM5.SA","BRML3.SA","BRPR3.SA","CASH3.SA",
    "CCRO3.SA","CEAB3.SA","CIEL3.SA","CMIG4.SA","CMIN3.SA","COGN3.SA","CPFE3.SA","CPLE3.SA",
    "CPLE6.SA","CRFB3.SA","CSAN3.SA","CSMG3.SA","CSNA3.SA","CURY3.SA","CVCB3.SA","CXSE3.SA",
    "CYRE3.SA","CYRE4.SA","DIRR3.SA","DXCO3.SA","ECOR3.SA","EGIE3.SA","ELET3.SA","ELET6.SA",
    "EMBJ3.SA","EMBR3.SA","ENBR3.SA","ENEV3.SA","ENGI11.SA","EQTL3.SA","EZTC3.SA","FIBR3.SA",
    "FLRY3.SA","GETT11.SA","GGBR4.SA","GOAU4.SA","GOLL4.SA","HAPV3.SA","HGTX3.SA","HYPE3.SA",
    "IGTI11.SA","IRBR3.SA","ISAE4.SA","ITSA4.SA","ITUB4.SA","JBSS3.SA","JHSF3.SA","JPSA3.SA",
    "KLBN11.SA","LCAM3.SA","LOGG3.SA","LREN3.SA","LWSA3.SA","MBRF3.SA","MGLU3.SA","MOTV3.SA",
    "MRFG3.SA","MRVE3.SA","MULT3.SA","NATU3.SA","NTCO3.SA","OIBR3.SA","PCAR3.SA","PETR3.SA",
    "PETR4.SA","PETZ3.SA","POMO4.SA","POSI3.SA","PRIO3.SA","PSSA3.SA","QUAL3.SA","RADL3.SA",
    "RAIL3.SA","RAIZ4.SA","RDOR3.SA","RECV3.SA","RENT3.SA","RENT4.SA","RRRP3.SA","SANB11.SA",
    "SAPR11.SA","SBSP3.SA","SLCE3.SA","SMFT3.SA","SMTO3.SA","SOMA3.SA","STBP3.SA","SULA11.SA",
    "SUZB3.SA","TAEE11.SA","TIMS3.SA","TOTS3.SA","TRPL4.SA","UGPA3.SA","USIM5.SA","VALE3.SA",
    "VALE5.SA","VAMO3.SA","VBBR3.SA","VIIA3.SA","VIVA3.SA","VIVT3.SA","VVAR11.SA","WEGE3.SA",
    "YDUQ3.SA",
]
INDICE = "^BVSP"
ALL = UNIVERSO + [INDICE]

def baixar(ticker, tentativas=3):
    for i in range(tentativas):
        try:
            df = yf.download(ticker, start=START, progress=False, auto_adjust=False, threads=False)
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
        except Exception as e:
            print(f"  tentativa {i+1} falhou para {ticker}: {e}")
        time.sleep(1 + i)
    return None

def main():
    frames, ok, falhas = [], [], []
    for n, tk in enumerate(ALL, 1):
        print(f"[{n}/{len(ALL)}] {tk} ...", end=" ")
        df = baixar(tk)
        if df is None or df.empty:
            print("FALHOU")
            falhas.append(tk)
            continue
        d = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        }).reset_index().rename(columns={"Date": "date"})
        if "adj_close" not in d.columns:  # yfinance novo pode nao trazer Adj Close
            d["adj_close"] = d["close"]
        d["ticker"] = tk
        frames.append(d[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]])
        ok.append(tk)
        print(f"OK ({len(d)} linhas, {d['date'].min().date()} -> {d['date'].max().date()})")
        time.sleep(0.3)  # gentileza com o servidor

    longo = pd.concat(frames, ignore_index=True)
    longo.to_parquet("precos_ibov.parquet", index=False)
    wide = longo.pivot_table(index="date", columns="ticker", values="adj_close")
    wide.to_csv("precos_ibov_wide.csv")

    with open("relatorio_download.txt", "w", encoding="utf-8") as f:
        f.write(f"Baixados: {len(ok)}/{len(ALL)}\n")
        f.write(f"Periodo: {longo['date'].min()} -> {longo['date'].max()}\n")
        f.write(f"Linhas: {len(longo)}\n\nFALHAS ({len(falhas)}):\n")
        for tk in falhas:
            f.write(f"  {tk}\n")
        f.write("\nCOBERTURA POR TICKER:\n")
        cov = longo.groupby("ticker")["date"].agg(["min", "max", "count"])
        f.write(cov.to_string())

    print(f"\nConcluido: {len(ok)} OK, {len(falhas)} falhas -> ver relatorio_download.txt")
    print("Envie 'precos_ibov.parquet' de volta na conversa do Claude.")

if __name__ == "__main__":
    main()
