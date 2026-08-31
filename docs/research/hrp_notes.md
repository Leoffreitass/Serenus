# Hierarchical Risk Parity (HRP) — Notas de Pesquisa

**Contexto:** documentação metodológica da etapa de alocação do modelo quantitativo em ações brasileiras (universo IBOVESPA). O HRP é aplicado **após** o filtro de baixa volatilidade, sobre uma cesta pré-selecionada de ~10–20 ações, em regime long-only.

**Data:** julho/2026.

---

## 1. O paper original e o problema que ele resolve

**Referência principal:** Marcos López de Prado (2016), *"Building Diversified Portfolios that Outperform Out-of-Sample"*, **The Journal of Portfolio Management**, vol. 42, n. 4, pp. 59–69. Primeira versão em working paper (SSRN, 2015). Disponível em: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>. O capítulo 16 de *Advances in Financial Machine Learning* (Wiley, 2018), do mesmo autor, reproduz e expande o método com código Python.

### 1.1 Instabilidade de Markowitz

A otimização média-variância clássica (Markowitz, 1952, *"Portfolio Selection"*, Journal of Finance) e sua implementação via **Critical Line Algorithm (CLA)** exigem a **inversão da matriz de covariância**. Problemas conhecidos:

- **Erros de estimação amplificados:** pequenas variações nos retornos esperados ou nas covariâncias estimadas produzem carteiras completamente diferentes (soluções instáveis e concentradas).
- **Número de condição (condition number):** razão entre o maior e o menor autovalor (em módulo) da matriz de correlação/covariância. Quando os ativos são muito correlacionados (caso típico de ações de um mesmo índice, como o IBOVESPA), a matriz se aproxima da singularidade, o número de condição explode e a inversa se torna numericamente instável — cada erro de estimação é multiplicado ao inverter.
- **"Maldição de Markowitz" (Markowitz's curse):** quanto mais correlacionados os ativos, maior a necessidade teórica de diversificação — e maior a instabilidade da solução ótima. Justamente quando mais precisamos da otimização, menos confiável ela é.
- **Demanda de dados:** para uma matriz de covariância invertível e bem condicionada de N=50 ativos são necessárias, no mínimo, ~5 anos de dados diários; mas as correlações financeiras não são estáveis por períodos tão longos.

### 1.2 "Error maximization" de Michaud

Richard Michaud (1989), *"The Markowitz Optimization Enigma: Is 'Optimized' Optimal?"*, **Financial Analysts Journal** 45(1), pp. 31–42, cunhou a crítica de que otimizadores média-variância são, na prática, **"maximizadores de erro"**: eles concentram peso exatamente nos ativos com maiores erros de estimação (retornos superestimados, riscos subestimados, correlações mal medidas). Resultado: carteiras "ótimas" in-sample que têm desempenho ruim out-of-sample — frequentemente perdendo para o ingênuo 1/N (ver DeMiguel, Garlappi & Uppal, 2009, *"Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?"*, Review of Financial Studies 22(5)).

### 1.3 A proposta do HRP

López de Prado substitui a inversão da matriz por **aprendizado de máquina não supervisionado (clustering hierárquico) + teoria de grafos**:

1. Não requer inversão nem positividade-definida da matriz de covariância (funciona até com matriz singular, ex.: N > T).
2. Usa a **estrutura hierárquica** dos ativos (árvore) em vez de tratar todos como substitutos perfeitos uns dos outros (grafo completo implícito na média-variância).
3. Aloca capital **de cima para baixo** ao longo da árvore, de forma que erros de estimação não se propaguem globalmente.

**Resultado central do paper (Monte Carlo, 10.000 simulações, 10 ativos, 520 observações, rebalanceamento mensal):** a variância out-of-sample do HRP foi **~72,5% menor que a do CLA** (mínima variância) e **~38,2% menor que a do IVP** (inverse-variance portfolio), o que equivale a um ganho de ~31% em índice de Sharpe out-of-sample frente ao CLA — ainda que o CLA minimize variância *in-sample*. Fontes: paper original (SSRN) e síntese em [Wikipedia — Hierarchical Risk Parity](https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity).

---

## 2. O algoritmo em três etapas (detalhe de implementação)

Insumos: matriz de correlação `ρ` (N×N) e matriz de covariância `Σ` (N×N) dos retornos dos N ativos.

### 2.1 Etapa (a): Tree clustering (clusterização hierárquica)

1. **Métrica de distância entre ativos:**

   `d(i,j) = sqrt( (1 − ρ(i,j)) / 2 )`,  com `d ∈ [0, 1]`.

   - `ρ = +1 → d = 0` (ativos idênticos); `ρ = 0 → d ≈ 0,707`; `ρ = −1 → d = 1`.
   - É uma **métrica verdadeira** (satisfaz não-negatividade, simetria e desigualdade triangular), o que justifica seu uso em clustering.
2. **Distância de segundo nível (usada por López de Prado):** para robustez, define-se a distância euclidiana entre as **colunas** da matriz D:

   `d̃(i,j) = sqrt( Σ_k (d(k,i) − d(k,j))² )`

   Ou seja, dois ativos são próximos se têm perfis de distância semelhantes em relação a *todos* os demais ativos do sistema, não apenas entre si.
3. **Aglomeração (linkage):** parte-se de N clusters unitários; a cada passo unem-se os dois clusters mais próximos, atualizando a matriz de distâncias, até restar um único cluster. O resultado é uma **árvore binária (dendrograma)** com N−1 nós internos — na prática, a matriz de linkage `(N−1)×4` do `scipy.cluster.hierarchy.linkage`.
4. **Escolha do método de linkage:**
   - **Single linkage** (distância mínima entre clusters): é o do paper original. Crítica: sofre de *chaining effect* — produz árvores desbalanceadas, encadeadas, pouco intuitivas.
   - **Ward** (mínima variância intra-cluster): gera clusters compactos e balanceados; é a escolha preferida em boa parte da literatura posterior (Raffinot 2017; índices institucionais como o FIVE Robust Multi-Asset usam Ward). **Average** e **complete** são intermediários.
   - Não há consenso definitivo: alguns estudos reportam deterioração ao trocar o single linkage, outros preferem Ward pela estabilidade ([Portfolio Optimizer, discussão de linkage](https://portfoliooptimizer.io/blog/hierarchical-risk-parity-introducing-graph-theory-and-machine-learning-in-portfolio-optimizer/)). **Recomendação prática para nosso caso (10–20 ações): testar single e ward em backtest; ward tende a ser mais estável em universos pequenos e correlacionados.**

### 2.2 Etapa (b): Quasi-diagonalização (seriação da matriz)

Reordenam-se linhas e colunas de `Σ` (e `ρ`) segundo a **ordem das folhas do dendrograma**, de modo que ativos semelhantes fiquem adjacentes e os maiores valores de covariância se concentrem perto da diagonal — a matriz fica em blocos "quase diagonais", revelando a estrutura de clusters. Não há mudança de base (ao contrário de PCA): apenas **permutação**.

Implementação: percorrer recursivamente a matriz de linkage a partir do nó raiz, substituindo cada cluster por seus dois filhos, até restarem apenas folhas (ativos originais); a sequência final de folhas é a ordem seriada (equivalente a `scipy.cluster.hierarchy.leaves_list`). Observação: a ordenação de folhas não é única — há 2^(N−1) ordenações compatíveis com a mesma árvore; a maioria das implementações usa a ordem padrão do hclust/scipy.

### 2.3 Etapa (c): Recursive bisection (alocação top-down)

Aloca-se o capital descendo a lista seriada por **bisseções sucessivas**:

1. Inicializar `w_i = 1` para todos os ativos; lista de itens `L = {lista seriada completa}`.
2. Para cada item `L_k` com mais de 1 ativo, **dividir ao meio** (na versão original de López de Prado, a bisseção corta a lista seriada em duas metades de tamanhos ⌈n/2⌉ e ⌊n/2⌋ — ela segue a *ordem* das folhas, não os cortes exatos do dendrograma) em sub-listas `L1` (esquerda) e `L2` (direita).
3. Para cada sub-lista `Lj`, calcular a **variância do cluster** usando pesos de variância inversa dentro do cluster:
   - `w̃ = diag(Σ_j)^(−1) / trace(diag(Σ_j)^(−1))`  (pesos IVP intra-cluster)
   - `V_j = w̃' Σ_j w̃`, onde `Σ_j` é a sub-matriz de covariância dos ativos de `Lj`.
4. **Fator de split:** `α = 1 − V_1 / (V_1 + V_2)`. Multiplicar os pesos de todos os ativos de `L1` por `α` e os de `L2` por `(1 − α)` — ou seja, alocação **inversamente proporcional à variância** entre os dois sub-clusters (paridade de risco simplificada em cada nó).
5. Repetir recursivamente até que todas as sub-listas tenham 1 ativo. Os pesos finais são produto dos fatores ao longo do caminho da raiz até cada folha; somam 1 e são todos ∈ [0, 1] por construção → **naturalmente long-only, sem restrições adicionais**.

### 2.4 Pseudocódigo

```text
função HRP(retornos R [T×N]):
    ρ  ← correlação(R);  Σ ← covariância(R)

    # (a) clustering
    D  ← sqrt((1 − ρ) / 2)                    # distância elemento a elemento
    D̃  ← dist_euclidiana_entre_colunas(D)     # distância de 2º nível
    link ← linkage(D̃, método = 'single' | 'ward')

    # (b) quasi-diagonalização
    ordem ← folhas_do_dendrograma(link)       # permutação dos N ativos

    # (c) bisseção recursiva
    w ← vetor de 1's [N]
    pilha ← [ordem]
    enquanto pilha não vazia:
        itens ← remover itens com len > 1
        para cada item em itens:
            L1 ← primeira metade(item);  L2 ← segunda metade(item)
            para j em {1, 2}:
                w̃_j ← 1/diag(Σ[Lj,Lj]);  w̃_j ← w̃_j / soma(w̃_j)
                V_j ← w̃_j' · Σ[Lj,Lj] · w̃_j
            α ← 1 − V_1 / (V_1 + V_2)
            w[L1] ← w[L1] · α
            w[L2] ← w[L2] · (1 − α)
            empilhar L1 e L2 (se len > 1)
    retornar w                                 # soma 1, todos ≥ 0
```

Complexidade: da ordem de O(N²) a O(N log N) — irrelevante para N ≈ 10–20. Guias de implementação passo a passo: [Hudson & Thames — An Introduction to the Hierarchical Risk Parity Algorithm](https://hudsonthames.org/an-introduction-to-the-hierarchical-risk-parity-algorithm/); [MATLAB — Create Hierarchical Risk Parity Portfolio](https://www.mathworks.com/help/finance/create-hierarchical-risk-parity-portfolio.html). Bibliotecas prontas: `riskfolio-lib` ([HCPortfolio](https://riskfolio-lib.readthedocs.io/en/latest/hcportfolio.html)), `PyPortfolioOpt`, `skfolio`, PortfolioLab/mlfinlab.

---

## 3. Comparações na literatura e refinamentos

### 3.1 HRP vs. benchmarks

| Estudo | Comparação | Resultado-chave |
|---|---|---|
| López de Prado (2016) | HRP vs CLA (mín. variância) e IVP, Monte Carlo | Variância OOS do HRP 72,5% menor que CLA e 38,2% menor que IVP; Sharpe OOS superior |
| Raffinot (2017), *Hierarchical Clustering-Based Asset Allocation*, JPM Multi-Asset Special Issue ([SSRN 2840729](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729)) | Carteiras hierárquicas vs 1/N, mín. variância, ERC | Carteiras baseadas em clustering hierárquico obtêm Sharpe ajustado a risco OOS superior aos métodos tradicionais; 1/N segue benchmark difícil de bater |
| Lohre, Rother & Schäfer (2020), *HRP: Accounting for Tail Dependencies in Multi-Asset Multi-Factor Allocations* (cap. de livro, [ResearchGate](https://www.researchgate.net/publication/345360937_Hierarchical_Risk_Parity_Accounting_for_Tail_Dependencies_in_Multi-Asset_Multi-Factor_Allocations)) | HRP vs risk parity/ERC em multi-ativos | HRP competitivo com ERC, com melhor comportamento em caudas quando se usam medidas de dependência de cauda no clustering |
| Jain & Jain (2019), *"Can Machine Learning-Based Portfolios Outperform Traditional Risk-Based Portfolios?"*, Risks 7(3) | HRP vs 1/N, mín. var., ERC (ações NIFTY) | Resultados mistos: HRP bom em drawdown/estabilidade, mas nem sempre vence 1/N em Sharpe |
| Pfitzinger & Katzke (2019), *A Constrained Hierarchical Risk Parity Algorithm with Cluster-based Capital Allocation* ([PDF](https://www.fmx.nfkatzke.com/Projects/HRP.pdf)) | HRP vs variantes com bisseção pelo dendrograma | Propõem parâmetro τ interpolando entre bisseção "por contagem" e divisão fiel ao dendrograma; melhora coerência estrutural |
| Estudos comparativos recentes (ex.: [arXiv 2210.00984](https://arxiv.org/pdf/2210.00984); [Empirical Economics 2026](https://link.springer.com/article/10.1007/s00181-026-02900-x); [ScienceDirect — The equally weighted portfolio still remains a challenging benchmark](https://www.sciencedirect.com/science/article/pii/S2110701724000489)) | HRP/HERC vs risk-based tradicionais | Padrão recorrente: HRP entrega **volatilidade e drawdowns menores e turnover menor** que mínima variância; ganho de Sharpe sobre 1/N existe mas é modesto e depende do universo/período |

**Turnover:** por não depender de inversão de matriz, os pesos do HRP variam suavemente entre rebalanceamentos; o turnover costuma ficar bem abaixo do da mínima variância irrestrita e próximo (um pouco acima) do 1/N — relevante no Brasil, onde custos de transação e impacto de mercado em ações menos líquidas do IBOVESPA não são desprezíveis.

### 3.2 Refinamentos posteriores

- **HERC — Hierarchical Equal Risk Contribution** (Thomas Raffinot, 2018, *The Hierarchical Equal Risk Contribution Portfolio*, [SSRN 3237540](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237540); obs.: o trabalho seminal de Raffinot sobre alocação por clustering hierárquico é de 2017): combina HRP com o Gap Index (Tibshirani et al.) para escolher o **número ótimo de clusters**, faz a divisão de capital **seguindo a estrutura real do dendrograma** (e não bisseções por contagem) e iguala a contribuição de risco entre clusters, permitindo medidas de risco alternativas (CVaR, CDaR). Ver também [Hudson & Thames — Beyond Risk Parity: HERC](https://hudsonthames.org/beyond-risk-parity-the-hierarchical-equal-risk-contribution-algorithm/).
- **NCO — Nested Clustered Optimization** (López de Prado, 2019, *A Robust Estimator of the Efficient Frontier*, [SSRN 3469961](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961); depois em *Machine Learning for Asset Managers*, Cambridge, 2020): clusteriza a matriz de correlação, otimiza (Markowitz/mín. var.) **dentro** de cada cluster, colapsa cada cluster num "ativo sintético" e otimiza **entre** clusters. Controla a instabilidade intra-cluster (fonte principal do erro, segundo o autor) mantendo a otimalidade média-variância. Ver [mlfinlab — NCO](https://random-docs.readthedocs.io/en/latest/portfolio_optimisation/nested_clustered_optimisation.html).
- **Outros:** *Adaptive Seriational Risk Parity* (Jaeger et al., 2021, Journal of Financial Data Science); HRP com dependência de cauda (Lohre et al., 2020); **Schur Complementary Allocation** (Cotton, 2024, [arXiv 2411.05807](https://ideas.repec.org/p/arx/papers/2411.05807.html)), que unifica HRP e mínima variância num contínuo, mostrando que o HRP é um caso extremo que ignora informação de covariância entre clusters.

---

## 4. Críticas e limitações conhecidas do HRP

1. **A bisseção recursiva original ignora o dendrograma.** O corte "metade a metade" da lista seriada pode separar ativos do mesmo cluster e juntar ativos de clusters distintos — o clustering é usado só para ordenar, não para dividir. É a crítica central de Raffinot (HERC) e de Pfitzinger & Katzke (2019).
2. **Sensibilidade a escolhas de projeto:** método de linkage (single vs ward), métrica de distância, janela de estimação e ordenação de folhas (2^(N−1) ordenações válidas) alteram os pesos finais; o método é heurístico, sem função-objetivo explícita, logo não há "ótimo" contra o qual validar.
3. **Ignora retornos esperados.** Como toda família risk-based, o HRP só usa risco; num pipeline com sinal de momentum isso é aceitável (o alfa vem do screen), mas o HRP não vai sobreponderar os papéis de maior sinal.
4. **Usa apenas variância/correlação (dependência linear).** Caudas pesadas, assimetria e correlações que disparam em crises não são capturadas — motivação das extensões com lower tail dependence (Lohre et al., 2020) e medidas CVaR/CDaR (HERC).
5. **Desconto teórico:** o HRP não usa a informação de covariância *entre* clusters ao alocar (trata os dois ramos como independentes); Cotton (2024) mostra que isso o afasta da fronteira eficiente quando a matriz é bem estimada. Em matrizes mal condicionadas, porém, essa "cegueira" é justamente a fonte da robustez.
6. **Instabilidade do próprio clustering:** com poucas observações, a árvore pode mudar bastante entre janelas, gerando algum turnover "estrutural"; single linkage agrava isso via chaining.
7. **Evidência empírica mista vs 1/N:** em universos pequenos e homogêneos, o ganho sobre equal weight pode ser pequeno ou nulo em Sharpe (o ganho aparece mais em vol/drawdown). Ver [comparativo EW vs HRP](https://www.researchgate.net/publication/380047000_Comparative_Study_of_the_Equal-Weight_Method_and_Hierarchical_Risk_Parity_in_Portfolio_Construction) e [The equally weighted portfolio still remains a challenging benchmark](https://www.sciencedirect.com/science/article/pii/S2110701724000489).

---

## 5. Notas práticas para o nosso caso (IBOVESPA, pós-screen de momentum, 10–20 ações, long-only)

### 5.1 Janela de estimação da covariância

- Prática comum na literatura/indústria: **126 a 504 pregões** (6 meses a 2 anos) de retornos diários. López de Prado usa ~260 obs. (1 ano) no experimento de Monte Carlo, com rebalanceamento mensal.
- Trade-off: janelas curtas (63–126d) reagem a mudanças de regime (relevante no Brasil: choques de política monetária, eventos políticos) mas geram árvores/pesos instáveis; janelas longas (252–504d) estabilizam a árvore mas carregam correlações defasadas.
- Sugestão para o modelo: **252 pregões** como base, com teste de robustez em 126 e 504; alternativamente, covariância com **decaimento exponencial (EWMA, λ≈0,97)** ou encolhimento de Ledoit–Wolf antes do clustering — o HRP não exige matriz invertível, mas estimadores encolhidos deixam a árvore e os pesos mais estáveis.
- Com N=15 e T=252, T/N ≈ 17 — confortável; é exatamente o cenário em que a mínima variância clássica já sofre e o HRP se comporta bem.

### 5.2 HRP sobre uma cesta pré-filtrada por momentum

- **Papéis separados:** o screen de momentum define *o que* comprar (alfa); o HRP define *quanto* de cada um (risco). Essa separação evita reintroduzir estimativas de retorno esperado — justamente o insumo mais ruidoso da média-variância.
- **Long-only automático:** os pesos do HRP são não-negativos por construção, sem necessidade de restrições — adequado a mandato long-only e às limitações de aluguel/short no Brasil.
- **Universo pequeno (10–20):** o clustering ainda é útil porque a cesta de momentum tende a concentrar setores "da moda" (ex.: commodities — VALE/PETR/SUZB — ou domésticas sensíveis a juros — varejo/construção). O HRP evita que 5 papéis do mesmo cluster setorial recebam 5× o risco de um papel isolado, algo que o 1/N faria.
- **Cuidados:**
  - Com N pequeno, considerar **ward linkage** (árvores mais balanceadas) e comparar com single em backtest.
  - Pesos do HRP podem ainda assim concentrar em papéis de baixíssima vol (ex.: utilities); se necessário, aplicar **caps por ativo** (ex.: máx. 15–20%) via renormalização iterativa ou pela variante restrita de Pfitzinger & Katzke.
  - **Rebalanceamento mensal** alinhado ao ciclo do screen de momentum; aplicar bandas de tolerância (ex.: rebalancear só se |Δw| > 1–2 p.p.) para conter turnover e custos na B3.
  - Como a composição da cesta muda a cada rebalanceamento (entra/sai papel do screen), parte do turnover vem do screen, não do HRP; medir os dois separadamente no backtest.
  - Benchmarks internos obrigatórios do backtest: **1/N, IVP (1/σ²), mínima variância (com shrinkage) e ERC**, todos sobre a mesma cesta pós-screen — é o teste que a literatura recomenda antes de adotar HRP.

---

## Referências

1. **López de Prado, M. (2016).** "Building Diversified Portfolios that Outperform Out-of-Sample". *The Journal of Portfolio Management*, 42(4), 59–69. SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678> (slides: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2713516>).
2. **López de Prado, M. (2018).** *Advances in Financial Machine Learning*. Wiley. (Cap. 16: implementação Python do HRP.)
3. **López de Prado, M. (2019).** "A Robust Estimator of the Efficient Frontier" (NCO). SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961>.
4. **López de Prado, M. (2020).** *Machine Learning for Asset Managers*. Cambridge University Press.
5. **Markowitz, H. (1952).** "Portfolio Selection". *The Journal of Finance*, 7(1), 77–91.
6. **Michaud, R. (1989).** "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?". *Financial Analysts Journal*, 45(1), 31–42.
7. **DeMiguel, V.; Garlappi, L.; Uppal, R. (2009).** "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?". *Review of Financial Studies*, 22(5), 1915–1953.
8. **Raffinot, T. (2017).** "Hierarchical Clustering-Based Asset Allocation". *The Journal of Portfolio Management*, 44(2). SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729>.
9. **Raffinot, T. (2018).** "The Hierarchical Equal Risk Contribution Portfolio". SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237540>.
10. **Pfitzinger, J.; Katzke, N. (2019).** "A Constrained Hierarchical Risk Parity Algorithm with Cluster-based Capital Allocation". Stellenbosch University Working Paper. <https://www.fmx.nfkatzke.com/Projects/HRP.pdf>.
11. **Lohre, H.; Rother, C.; Schäfer, K. A. (2020).** "Hierarchical Risk Parity: Accounting for Tail Dependencies in Multi-Asset Multi-Factor Allocations". In *Machine Learning for Asset Management*. Wiley. <https://www.researchgate.net/publication/345360937_Hierarchical_Risk_Parity_Accounting_for_Tail_Dependencies_in_Multi-Asset_Multi-Factor_Allocations>.
12. **Jaeger, M. et al. (2021).** "Adaptive Seriational Risk Parity and Other Extensions for Heuristic Portfolio Construction Using Machine Learning and Graph Theory". *The Journal of Financial Data Science*. <https://jfds.pm-research.com/content/early/2021/10/06/jfds.2021.1.078>.
13. **Cotton, P. (2024).** "Schur Complementary Allocation: A Unification of Hierarchical Risk Parity and Minimum Variance Portfolios". arXiv:2411.05807. <https://ideas.repec.org/p/arx/papers/2411.05807.html>.
14. **Wikipedia.** "Hierarchical Risk Parity". <https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity>.
15. **Hudson & Thames.** "The Hierarchical Risk Parity Algorithm: An Introduction". <https://hudsonthames.org/an-introduction-to-the-hierarchical-risk-parity-algorithm/>; "Beyond Risk Parity: The Hierarchical Equal Risk Contribution Algorithm". <https://hudsonthames.org/beyond-risk-parity-the-hierarchical-equal-risk-contribution-algorithm/>.
16. **Portfolio Optimizer (blog).** "Hierarchical Risk Parity: Introducing Graph Theory and Machine Learning". <https://portfoliooptimizer.io/blog/hierarchical-risk-parity-introducing-graph-theory-and-machine-learning-in-portfolio-optimizer/>.
17. **MathWorks.** "Create Hierarchical Risk Parity Portfolio". <https://www.mathworks.com/help/finance/create-hierarchical-risk-parity-portfolio.html>.
18. **Riskfolio-Lib.** "Hierarchical Clustering Portfolio Optimization". <https://riskfolio-lib.readthedocs.io/en/latest/hcportfolio.html>.
19. **Comparativos empíricos:** arXiv:2210.00984 (<https://arxiv.org/pdf/2210.00984>); "Hierarchical risk clustering versus traditional risk-based portfolios" (*Empirical Economics*, 2026, <https://link.springer.com/article/10.1007/s00181-026-02900-x>); "The equally weighted portfolio still remains a challenging benchmark" (<https://www.sciencedirect.com/science/article/pii/S2110701724000489>); "Comparative Study of the Equal-Weight Method and HRP" (<https://www.researchgate.net/publication/380047000_Comparative_Study_of_the_Equal-Weight_Method_and_Hierarchical_Risk_Parity_in_Portfolio_Construction>).
