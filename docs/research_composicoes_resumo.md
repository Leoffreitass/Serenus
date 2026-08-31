# Reconstrução histórica da composição do IBOVESPA — jan/2023 a jul/2026

**Baseline:** carteira set–dez/2022 (coluna 30/11/2022 do xlsx) — **92 papéis / 89 empresas**.

**Semântica:** a lista de cada janela é a composição no **início da vigência** da carteira quadrimestral.
Eventos extraordinários (deslistagens, fusões, RJ/RE, trocas de ticker, bonificações) alteram a carteira
**durante** a janela e estão na linha do tempo abaixo.

**Validação:** a contagem derivada por roll-forward bate com a contagem oficial da B3 em **todas** as 11
janelas, e a lista completa oficial de mai/2026 (79 papéis) coincide **100%** com a lista derivada — drift
acumulado zero desde nov/2022.

## Janelas quadrimestrais

| Vigência | Papéis | Entradas (rebalanceamento) | Saídas (rebalanceamento) | Confiança |
|---|---|---|---|---|
| jan–abr/2023 | 89 | — | POSI3, IRBR3, SULA11 | média |
| mai–ago/2023 | 86 | IRBR3 (retorna) | BPAN4, ECOR3, QUAL3 | média |
| set–dez/2023 | 86 | RECV3, VAMO3 | CASH3 | média |
| jan–abr/2024 | 87 | TRPL4 | — | média |
| mai–ago/2024 | 86 | VIVA3 | BHIA3 | média |
| set–dez/2024 | 86 | AURE3, STBP3, CXSE3 | DXCO3 | média |
| jan–abr/2025 | 87 | POMO4, PSSA3 (+AMOB3 por cisão) | ALPA4, EZTC3 | média |
| mai–ago/2025 | 87 | DIRR3, SMFT3 | AMOB3, LWSA3 | média |
| set–dez/2025 | 84 | CEAB3, CURY3 | PETZ3, SMTO3 | média |
| jan–abr/2026 | 85 | CSMG3 | CVCB3 | média |
| mai–set/2026 | 79 | — | IRBR3, AXIA7, CYRE4, RENT4 | **alta (lista completa oficial)** |

Contagens oficiais (papéis/empresas): 89/86 · 86/83 · 86/83 · 87/84 · 86/83 · 86/83 · 87/84 · 87/84 · 84/81 · 85/79 · 79/76.

## Linha do tempo de eventos extraordinários

**2023**
- **jan/2023 — AMER3** removida (recuperação judicial da Americanas, pedido em 19/01).
- **jan/2023 — BRML3 → ALSO3** (BR Malls incorporada pela Aliansce Sonae; ALSO3 assume a vaga).
- **fev/2023 — SULA11** deixa de negociar (incorporada pela Rede D'Or); já retirada na carteira jan/2023.
- **jun/2023 — ENBR3** removida (OPA da EDP, fechamento de capital).
- **set/2023 — VIIA3 → BHIA3** (Via vira Grupo Casas Bahia — troca de ticker).
- **nov–dez/2023 — ALSO3 → ALOS3** (Allos — troca de ticker).

**2024**
- **01/02/2024 — GOLL4** removida (Chapter 11 da Gol em 25/01; excluída do IBOV e de mais 9 índices).
- **mai/2024 — BHIA3** sai (recuperação extrajudicial de 28/04; VIVA3 entra como "nova titular").
- **ago/2024 — SOMA3 + ARZZ3 → AZZA3** (fusão Soma+Arezzo = Azzas 2154; 2 papéis viram 1).
- **ago/2024 — CIEL3** removida (OPA de fechamento de capital da Cielo).
- **ago–set/2024 — RRRP3 → BRAV3** (3R Petroleum + Enauta = Brava Energia).
- **nov/2024 — TRPL4 → ISAE4** (ISA CTEEP vira ISA Energia Brasil).
- **dez/2024 — cisão Automob/Vamos**: AMOB3 estreia e entra nos índices da VAMO3 (aparece na carteira jan/2025; sai em mai/2025).

**2025**
- **02/05/2025 — CCRO3 → MOTV3** (CCR vira Motiva).
- **28/05/2025 — AZUL4** removida de todos os índices (Chapter 11 da Azul).
- **jun/2025 — JBSS3** deixa a B3 (dupla listagem NYSE; BDR JBSS32 estreia 09/06 e **não** entra no índice).
- **jul/2025 — NTCO3 → NATU3** (Natura incorpora a Natura &Co).
- **jul/2025 — CRFB3** removida (fechamento de capital do Carrefour Brasil).
- **23/09/2025 — MRFG3 + BRFS3 → MBRF3** (fusão Marfrig+BRF; 2 papéis viram 1).
- **out/2025 — STBP3** removida (OPA da CMA CGM concluída; Santos Brasil deixa a B3).
- **03/11/2025 — EMBR3 → EMBJ3** (Embraer).
- **10/11/2025 — ELET3/ELET6 → AXIA3/AXIA6** (Eletrobras vira Axia Energia).
- **dez/2025 — CPLE6 → CPLE3** (Copel: unificação de classes / Novo Mercado; data aproximada).
- **dez/2025 — AXIA7 e RENT4** (novas PNs de bonificação — capitalização de reservas p/ driblar tributação
  de dividendos) entram no índice; **CYRE4** idem na virada dez/2025–jan/2026. As três são classes
  temporárias e saem no rebalanceamento de mai/2026.

**2026**
- **16/03/2026 — PCAR3 e RAIZ4** removidas de todos os índices (recuperações extrajudiciais do GPA, 11/03,
  e da Raízen, 12/03).
- **04/05/2026 — carteira mai/2026**: sem entradas; saem IRBR3, AXIA7, CYRE4, RENT4 → 79 papéis / 76 empresas.

## Ressalvas
1. Datas com mês correto mas dia incerto: AMER3, BRML3→ALSO3, ENBR3, BHIA3 (troca), ALOS3, AZZA3/CIEL3/BRAV3, ISAE4, AMOB3, CPLE3, CYRE4.
2. IRBR3 tem **gap**: fora do índice em jan–abr/2023, retorna em mai/2023, sai de vez em mai/2026.
3. Carteira jan/2023: fonte oficial (borainvestir/B3) tem título "88 papéis" mas URL/conteúdo "89"; 89 é o valor que fecha com mai/2023 = 86.
4. Portal Tela (04/05/2026) cita "entradas" de Marcopolo ON e Pine PN em mai/2026 — inconsistente com B3/CNN e com a lista completa; descartado.
5. Para preços em backtest, emendar séries nas trocas de ticker (mesma posição econômica): VIIA3→BHIA3, ALSO3→ALOS3, RRRP3→BRAV3, TRPL4→ISAE4, CCRO3→MOTV3, NTCO3→NATU3, EMBR3→EMBJ3, ELET3/6→AXIA3/6, CPLE6→CPLE3; fusões SOMA3+ARZZ3→AZZA3 e MRFG3+BRFS3→MBRF3 exigem tratamento de relação de troca.
