# Volatility Targeting e GARCH(1,1) — Notas de Pesquisa

**Contexto:** documentação metodológica para estratégia quantitativa em ações brasileiras (universo IBOVESPA) com controle de volatilidade no nível do portfólio e previsão de volatilidade via GARCH(1,1). Rebalanceamento mensal (~21 pregões).

**Data:** julho/2026

---

## 1. Literatura de Volatility Targeting (VT) / Portfólios Geridos por Volatilidade

### 1.1 Moreira & Muir (2017) — "Volatility-Managed Portfolios" (Journal of Finance)

Trabalho seminal. A estratégia escala a exposição a um fator de risco de forma **inversamente proporcional à variância realizada recente** (no paper, a variância realizada do mês anterior):

```
w_t = c / σ̂²_{t-1}
```

onde `c` é uma constante de normalização (calibrada para igualar a volatilidade incondicional do ativo original).

**Principais achados:**

- A gestão por volatilidade gera **alfas grandes e positivos** e **aumenta substancialmente o Sharpe** para o fator de mercado (equities), momentum, valor, profitability, ROE, investment e carry de moedas.
- Ganhos de utilidade expressivos para um investidor média-variância.
- Resultado contraintuitivo: a estratégia **assume menos risco em recessões e crises** e ainda assim entrega retornos médios altos — porque variações de volatilidade **não são compensadas por variações proporcionais no retorno esperado**. Ou seja, a razão retorno esperado/variância é maior quando a volatilidade está baixa, o que justifica *market timing* via volatilidade.
- Mecanismo econômico: volatilidade é altamente previsível no curto prazo (clustering), retornos esperados quase não são; logo, reduzir exposição quando σ sobe melhora o trade-off risco-retorno.

### 1.2 Harvey, Hoyle, Korgaonkar, Rattray, Sargaison & van Hemert (2018) — "The Impact of Volatility Targeting" (Journal of Portfolio Management)

Estudo abrangente (Man Group + Duke): **mais de 60 ativos**, dados diários desde 1926, cobrindo ações, títulos (bonds), crédito, commodities e moedas; também carteiras 60/40 e risk parity.

**Principais achados (headline):**

- **VT melhora o Sharpe em "ativos de risco": ações e crédito.** Justificativa: nesses ativos existe o **leverage effect** — correlação negativa entre retorno e volatilidade (volatilidade sobe quando preços caem). Reduzir exposição quando a volatilidade sobe induz um comportamento tipo *momentum* (vende após quedas), historicamente lucrativo nesses mercados.
- **Em bonds (Tesouro), moedas e commodities, o efeito sobre o Sharpe é insignificante/negligível** — nesses mercados a relação retorno-volatilidade é fraca ou inexistente.
- **Benefícios que valem para TODAS as classes:**
  - Redução da **vol-of-vol** (volatilidade da volatilidade): a volatilidade realizada da estratégia fica muito mais estável — no exemplo de equities do paper, a vol-of-vol cai de ~4,6% para ~1,8%. A carteira "entrega o risco que promete".
  - **Caudas esquerdas mais curtas e drawdowns menores**: eventos extremos negativos ocorrem tipicamente em regimes de volatilidade alta, quando a estratégia já está com exposição reduzida. Máximos drawdowns melhoram, inclusive em carteiras 60/40 e risk parity.
- Mecânica: alavanca em períodos de vol baixa e desalavanca em vol alta, mirando volatilidade constante em vez de exposição nocional constante. Estimador de vol usado: média móvel exponencial com **meia-vida de ~20 dias**.

### 1.3 Refinamentos e críticas

- **Cederburg, O'Doherty, Wang & Yan (2020) — "On the performance of volatility-managed portfolios" (Journal of Financial Economics):** crítica importante. Os alfas das regressões de spanning de Moreira-Muir usam parâmetros conhecidos só no fim da amostra; **versões implementáveis em tempo real** frequentemente entregam retornos equivalentes-certos menores. Fora da amostra, a gestão por volatilidade funciona de forma mais consistente para **momentum** (8 de 9 estratégias melhoram) e betting-against-beta/profitability, mas para o fator de **mercado** o ganho real-time é pequeno (Sharpe 0,42 vs 0,46 do original). Implicação prática: calibrar a constante de escala com dados apenas passados (expanding window) e reportar resultados líquidos de custos.
- **Bongaerts, Kang & van Dijk (2020) — "Conditional Volatility Targeting" (Financial Analysts Journal):** propõem aplicar o de-risking apenas em regimes extremos de volatilidade (condicional), reduzindo turnover e custos e preservando os ganhos.
- **Barroso & Santa-Clara (2015) — "Momentum has its moments" (JFE):** escalar momentum pela volatilidade realizada praticamente elimina os crashes de momentum — evidência clássica de VT no nível de fator.

**Síntese para o nosso caso (portfólio long-only de ações do IBOVESPA):** ações são exatamente a classe em que a literatura encontra melhora de Sharpe com VT, além dos benefícios "universais" (vol-of-vol menor, caudas e drawdowns reduzidos). A crítica de Cederburg et al. recomenda backtest honesto (parâmetros estimados só com informação disponível) e atenção a custos de transação.

---

## 2. Mecânica do Volatility Targeting

### 2.1 Regra de exposição

A cada rebalanceamento (mensal), a fração investida no portfólio de risco é:

```
w_t = min( σ_target / σ̂_t , w_max )
```

- `σ_target`: volatilidade-alvo anualizada;
- `σ̂_t`: previsão de volatilidade anualizada do portfólio para o horizonte até o próximo rebalanceamento (21 pregões);
- `w_max`: teto de alavancagem (leverage cap).

O restante `(1 − w_t)` fica em caixa — no Brasil, **CDI/Selic** (custo de oportunidade e taxa livre de risco local). Se `w_t < 1`, a carteira fica parcialmente em CDI; se `w_max > 1`, há alavancagem (ver §5.3).

### 2.2 Escolha do target

- Prática de mercado: **10%–15% a.a.** para estratégias de ações vol-managed (Harvey et al. usam alvos nessa faixa em suas ilustrações; fundos "managed volatility" tipicamente 10%–12%).
- O IBOVESPA tem volatilidade histórica de longo prazo na casa de **20%–30% a.a.** Um alvo de 12%–15% a.a. implica exposição média bem abaixo de 100% (com muito CDI), enquanto um alvo próximo de 20% mantém exposição média perto de 1. A escolha é um trade-off entre suavização de risco e *cash drag* (mitigado no Brasil pelo carrego alto do CDI).
- Boa prática: fixar o alvo ex-ante (não otimizá-lo no backtest) e reportar sensibilidade a alvos alternativos.

### 2.3 Estimadores de volatilidade — baselines padrão

1. **Desvio-padrão móvel (rolling std):** janela de 21 ou 63 dias sobre retornos diários; anualização por `√252`. Simples, mas reage com atraso e sofre de "efeito fantasma" (choque sai da janela abruptamente).

```
σ̂²_t = (1/N) · Σ_{i=1..N} r²_{t-i}        (média zero assumida)
σ̂_anual = σ̂_diária · √252
```

2. **EWMA / RiskMetrics (J.P. Morgan, 1996):** média móvel exponencial com fator de decaimento **λ = 0,94** para dados diários (Technical Document RiskMetrics):

```
σ̂²_t = λ·σ̂²_{t-1} + (1−λ)·r²_{t-1}
```

Equivale a um GARCH(1,1) restrito com ω = 0 e α + β = 1 (IGARCH sem constante) — captura clustering, mas **não tem reversão à média**: a previsão multi-passos é flat (σ̂²_{t+h} = σ̂²_{t+1} ∀h), o que é uma limitação para horizontes de 21 dias.

3. **GARCH(1,1)** (nosso estimador principal — §3): tem constante e reversão à média, gerando previsões multi-passos que convergem para a variância incondicional.

---

## 3. GARCH(1,1) como previsor de volatilidade

### 3.1 Origem

- **Engle (1982)** introduziu o **ARCH** (Autoregressive Conditional Heteroskedasticity): variância condicional como função de choques passados ao quadrado — primeiro modelo a formalizar o *volatility clustering* (Nobel 2003).
- **Bollerslev (1986)** generalizou para o **GARCH**, adicionando defasagens da própria variância condicional — parcimônia: um GARCH(1,1) substitui um ARCH de ordem alta.

### 3.2 O modelo

Retornos: `r_t = μ + ε_t`, com `ε_t = σ_t · z_t`, `z_t ~ iid(0,1)`.

**Equação da variância condicional:**

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

com `ω > 0`, `α ≥ 0`, `β ≥ 0`.

- **α** (ARCH): reação a choques recentes;
- **β** (GARCH): persistência/memória da volatilidade;
- **α + β**: persistência total. Em índices de ações (incl. Ibovespa) tipicamente 0,95–0,99.

**Estacionariedade (covariância):** `α + β < 1`.

**Variância incondicional (longo prazo):**

```
σ̄² = ω / (1 − α − β)
```

O modelo captura os dois fatos estilizados essenciais: **clustering** (choques grandes → variância alta persistente) e **reversão à média** (a variância converge para σ̄² à taxa `α + β`).

### 3.3 Previsão multi-passos (essencial para o horizonte de 21 dias)

Previsão 1 passo à frente (feita em t):

```
σ̂²_{t+1} = ω + α·ε²_t + β·σ²_t
```

Para h ≥ 1, por recursão (E[ε²_{t+h}] = σ²_{t+h}):

```
E_t[σ²_{t+h}] = σ̄² + (α + β)^{h−1} · ( σ̂²_{t+1} − σ̄² )
```

ou seja, decaimento geométrico da variância prevista em direção à incondicional, com meia-vida `ln(0,5)/ln(α+β)` dias.

**Variância acumulada no horizonte de rebalanceamento H = 21 pregões:**

```
σ̂²_{t,t+H} = Σ_{h=1..H} E_t[σ²_{t+h}]
           = H·σ̄² + (σ̂²_{t+1} − σ̄²) · (1 − (α+β)^H) / (1 − (α+β))
```

**Volatilidade anualizada usada no VT:**

```
σ̂_t (a.a.) = √( σ̂²_{t,t+H} · 252 / H )
```

Esta é a quantidade que entra em `w_t = σ_target / σ̂_t`. Nota: agregação temporal de retornos GARCH gera caudas menos pesadas no horizonte agregado; a fórmula acima (soma das variâncias esperadas, ignorando autocorrelação de retornos) é o padrão de mercado.

### 3.4 Estimação

- **Máxima verossimilhança (MLE/QMLE)** sobre a janela de estimação, maximizando a log-verossimilhança condicional. Com inovações normais, o QMLE é consistente mesmo sob não-normalidade (erros-padrão robustos de Bollerslev-Wooldridge).
- **Inovações Student-t** (Bollerslev, 1987): recomendadas para mercados emergentes — retornos do Ibovespa têm caudas pesadas mesmo após filtrar o GARCH; a t-Student estima os graus de liberdade ν (tipicamente 4–8 em ações BR) e melhora o ajuste da verossimilhança e a robustez dos parâmetros.
- Boas práticas para o backtest: janela de estimação de 500–1000 pregões (expanding ou rolling), reestimação a cada rebalanceamento, **usando apenas dados até t** (sem look-ahead); checar `α+β < 1` e tratar não-convergência (fallback para EWMA).

### 3.5 Implementação em Python — pacote `arch`

Pacote **`arch`** de Kevin Sheppard (https://arch.readthedocs.io / https://github.com/bashtage/arch):

```python
from arch import arch_model

# retornos em % (escala recomendada pelo pacote para convergência numérica)
am  = arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
res = am.fit(disp="off")
fc  = res.forecast(horizon=21, reindex=False)
var_21d = fc.variance.iloc[-1].sum()           # variância acumulada em 21 dias
vol_ann = (var_21d * 252 / 21) ** 0.5 / 100    # vol anualizada (fração)
```

Suporta GARCH, EGARCH, GJR-GARCH (assimetria/leverage), distribuições Normal, t, skew-t; previsão analítica, por simulação ou bootstrap.

---

## 4. Evidência: GARCH vs vol realizada/EWMA para previsão

### 4.1 Hansen & Lunde (2005) — "A Forecast Comparison of Volatility Models: Does Anything Beat a GARCH(1,1)?" (Journal of Applied Econometrics)

- Compararam **330 modelos da família ARCH/GARCH** usando o teste de Superior Predictive Ability, com volatilidade realizada intradiária como proxy do alvo.
- **Câmbio (DM/USD): nada supera o GARCH(1,1)** de forma estatisticamente significativa — modelos mais sofisticados não agregam.
- **Ações (IBM): modelos com efeito alavancagem** (assimetria, e.g. GJR/EGARCH) superam o GARCH(1,1) simples.
- Leitura prática: GARCH(1,1) é um **benchmark extremamente difícil de bater** e uma escolha defensável como modelo-base; para ações, vale testar GJR-GARCH como extensão (a queda do mercado eleva a vol mais do que a alta).

### 4.2 Evidência brasileira (Ibovespa / ações BR)

- **Moreira & Lemgruber (2004), "Desempenho de estimadores de volatilidade na Bolsa de Valores de São Paulo" (Revista Brasileira de Economia):** avaliação de estimadores de volatilidade (janelas móveis, EWMA, GARCH) para o mercado acionário brasileiro; modelos condicionais simples apresentam bom desempenho para previsão de curto prazo.
- **Moreira & Lemgruber (2002), Trabalho para Discussão nº 61 do Banco Central do Brasil — "O Uso de Dados de Alta Frequência na Estimação da Volatilidade e do Valor em Risco para o IBOVESPA":** GARCH com filtragem de sazonalidade intradiária para vol intradiária; para VaR **diário**, desvio-padrão em janela móvel e **EWMA tiveram desempenho sólido** — métodos simples funcionam bem no horizonte diário para o Ibovespa.
- **Val, Pinto & Klotzle (2014), "Volatilidade e Previsão de Retorno com Modelos de Alta Frequência e GARCH: Evidências para o Mercado Brasileiro" (Revista Contabilidade & Finanças–USP, v. 25, n. 65), sobre PETR4 e VALE5:** modelos de alta frequência (HAR-RV e 2-Comp) **superam a família GARCH** dentro e fora da amostra nos horizontes de 1, 5 e **22 dias**. Implicação: se houver dados intradiários confiáveis, HAR-RV é um upgrade natural; com dados apenas diários, GARCH(1,1)-t permanece o estado da prática.
- Literatura brasileira adicional (VaR com EWMA/GARCH/vol estocástica; range-based) corrobora: alta persistência (α+β ≈ 0,95–0,99), caudas pesadas (t-Student com ν baixo) e bom desempenho relativo do GARCH(1,1) no Ibovespa.

**Conclusão para a estratégia:** usar **GARCH(1,1) com inovações t-Student** como previsor principal (justificado por Hansen-Lunde e pela evidência local), com **EWMA λ=0,94** e rolling std 63d como baselines de comparação no backtest; considerar GJR-GARCH como teste de robustez (assimetria documentada em ações).

---

## 5. Notas práticas de implementação

### 5.1 VT no nível do PORTFÓLIO vs no nível do ativo

- **Nível do portfólio (nossa escolha):** primeiro constroem-se os pesos da carteira de ações (sinal/ranking dentro do universo IBOVESPA); depois estima-se o GARCH sobre a **série de retornos do próprio portfólio** e escala-se a exposição total entre carteira e **caixa/CDI**: `posição = w_t · carteira + (1 − w_t) · CDI`. Vantagens: (i) captura automaticamente correlações entre ações (a vol do portfólio já embute diversificação); (ii) não distorce os pesos relativos do sinal; (iii) um único modelo a estimar; (iv) é o desenho de Moreira-Muir e Harvey et al.
- **Nível do ativo:** escalar cada ação por sua própria vol (inverse-vol / vol targeting por papel). Muda a composição relativa (tilt para ações de baixa vol — mistura o efeito VT com o fator low-vol) e exige matriz de covariância para controlar a vol total. Útil como esquema de ponderação, mas é outra decisão de desenho; não substitui o overlay de portfólio.
- Prática comum: combinar os dois (pesos ~inverse-vol dentro da carteira + overlay de VT no total), documentando cada camada separadamente.

### 5.2 Interação com rebalanceamento mensal

- O peso `w_t` é recalculado **a cada rebalanceamento mensal**, usando a previsão GARCH **acumulada para os próximos 21 pregões** (§3.3) — coerência entre horizonte de previsão e período de manutenção da posição.
- Rebalanceamento mensal (vs diário) reduz turnover e custos, ao preço de reação mais lenta a choques de vol intra-mês. Mitigações opcionais: banda de tolerância (só ajustar se `|w_novo − w_atual|` > x p.p.) ou gatilho extraordinário de de-risking se a vol prevista dobrar intra-mês (no espírito do VT condicional de Bongaerts et al., 2020).
- Como α+β é alto (persistência), a previsão média de 21 dias fica entre a vol spot e a incondicional — o VT mensal com GARCH é naturalmente mais suave do que com EWMA spot, reduzindo turnover do overlay.
- Custos: o turnover do overlay é proporcional a `|Δw_t|`; reportar Sharpe líquido de custos (crítica de Cederburg et al., 2020).

### 5.3 Restrição de alavancagem (long-only)

- **Cap em w_max = 1,0 (sem alavancagem):** implementação mais simples para um produto long-only: a estratégia só **reduz** risco (fica entre 100% ações e 100% CDI). Perde-se o "lado de cima" do VT (alavancar em vol baixa), mas preservam-se os principais benefícios (corte de caudas, drawdowns, vol-of-vol). No Brasil, o CDI alto torna o estado desinvestido pouco custoso.
- **Alavancagem modesta (w_max = 1,3–1,5):** mais fiel à literatura (Moreira-Muir e Harvey et al. permitem alavancar; caps de 1,5x–2x são comuns na literatura aplicada). No Brasil é viável via **futuro de Ibovespa (IND/WIN)** ou aluguel/termo, mas com custo de funding ~CDI sobre a parte alavancada — o que erode o ganho esperado da alavancagem quando o prêmio de risco local é comparável ao CDI.
- Recomendação: reportar as duas versões (cap 1,0 como base; cap 1,5 como variante), com `w_t = clip(σ_target/σ̂_t, 0, w_max)`.
- Cuidado adicional: piso opcional (ex.: w_min = 0,2) para evitar sair 100% do mercado por picos espúrios de previsão.

---

## Resumo das fórmulas-chave

| Objeto | Fórmula |
|---|---|
| Peso VT | `w_t = clip(σ_target / σ̂_t, w_min, w_max)` |
| GARCH(1,1) | `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}` |
| Estacionariedade | `α + β < 1` |
| Variância incondicional | `σ̄² = ω/(1−α−β)` |
| Previsão h passos | `E_t[σ²_{t+h}] = σ̄² + (α+β)^{h−1}(σ̂²_{t+1} − σ̄²)` |
| Variância 21d acumulada | `σ̂²_{t,t+21} = 21·σ̄² + (σ̂²_{t+1}−σ̄²)·(1−(α+β)^21)/(1−(α+β))` |
| Anualização | `σ̂_aa = √(σ̂²_{t,t+H}·252/H)` |
| EWMA (RiskMetrics) | `σ̂²_t = 0,94·σ̂²_{t-1} + 0,06·r²_{t-1}` |

---

## Referências

1. **Moreira, A.; Muir, T. (2017).** "Volatility-Managed Portfolios". *The Journal of Finance*, 72(4), 1611–1644. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513) · [PDF do autor](https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf) · [NBER w22208](https://www.nber.org/papers/w22208)
2. **Harvey, C. R.; Hoyle, E.; Korgaonkar, R.; Rattray, S.; Sargaison, M.; van Hemert, O. (2018).** "The Impact of Volatility Targeting". *The Journal of Portfolio Management*, 45(1), 14–33. [JPM](https://jpm.pm-research.com/content/45/1/14.abstract) · [SSRN 3175538](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3175538) · [resumo QuantPedia](https://quantpedia.com/the-impact-of-volatility-targeting-on-equities-bonds-commodities-and-currencies/)
3. **Cederburg, S.; O'Doherty, M. S.; Wang, F.; Yan, X. S. (2020).** "On the performance of volatility-managed portfolios". *Journal of Financial Economics*, 138(1), 95–117. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X) · [PDF](https://www.lehigh.edu/~xuy219/research/COWY.pdf)
4. **Bongaerts, D.; Kang, X.; van Dijk, M. (2020).** "Conditional Volatility Targeting". *Financial Analysts Journal*, 76(4). [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/0015198X.2020.1790853)
5. **Barroso, P.; Santa-Clara, P. (2015).** "Momentum has its moments". *Journal of Financial Economics*, 116(1), 111–120.
6. **Engle, R. F. (1982).** "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation". *Econometrica*, 50(4), 987–1007.
7. **Bollerslev, T. (1986).** "Generalized Autoregressive Conditional Heteroskedasticity". *Journal of Econometrics*, 31(3), 307–327.
8. **Bollerslev, T. (1987).** "A Conditionally Heteroskedastic Time Series Model for Speculative Prices and Rates of Return". *Review of Economics and Statistics*, 69(3), 542–547. (GARCH-t)
9. **Hansen, P. R.; Lunde, A. (2005).** "A Forecast Comparison of Volatility Models: Does Anything Beat a GARCH(1,1)?". *Journal of Applied Econometrics*, 20(7), 873–889. [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/jae.800) · [SSRN 264571](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264571)
10. **J.P. Morgan/Reuters (1996).** *RiskMetrics — Technical Document*, 4ª ed. (EWMA, λ = 0,94 diário).
11. **Moreira, J. M. S.; Lemgruber, E. F. (2002).** "O Uso de Dados de Alta Frequência na Estimação da Volatilidade e do Valor em Risco para o IBOVESPA". Banco Central do Brasil, Trabalho para Discussão nº 61. [EconPapers](https://econpapers.repec.org/RePEc:bcb:wpaper:61) · [RBE (versão 2004)](https://periodicos.fgv.br/rbe/article/view/870)
12. **Moreira, J. M. S.; Lemgruber, E. F. (2004).** "Desempenho de estimadores de volatilidade na Bolsa de Valores de São Paulo". *Revista Brasileira de Economia*, 58(3). [SciELO](https://scielo.br/scielo.php?pid=S0034-71402004000300006&script=sci_arttext)
13. **Val, F. F.; Pinto, A. C. F.; Klotzle, M. C. (2014).** "Volatilidade e Previsão de Retorno com Modelos de Alta Frequência e GARCH: Evidências para o Mercado Brasileiro". *Revista Contabilidade & Finanças – USP*, 25(65), 189–201. [SciELO](https://www.scielo.br/j/rcf/a/CtBqHvyZ6MT3fphZFN94f4C/?lang=pt) · [DOAJ](https://doaj.org/article/623cde41980f4085b7e4df0407fdc495)
14. **Sheppard, K.** *arch* — ARCH/GARCH models in Python. [Docs](https://arch.readthedocs.io) · [GitHub](https://github.com/bashtage/arch)
