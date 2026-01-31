# Inventário do projeto — nova-selachiia

## 1) Motivação científica e pessoal
Estudar como mudanças em drivers ambientais e antrópicos alteram a dinâmica, persistência e risco de colapso de espécies de Selachiia na costa brasileira.

Recorte pessoal: foco local (Brasil), dados relativamente melhores e relevância aplicada.

## 2) Pergunta central de pesquisa
Se variáveis ambientais e/ou antrópicas mudarem em Δ% (intervenção contrafactual), como isso altera:
- dinâmica temporal,
- probabilidade de sobrevivência,
- padrões de colapso

de espécies-chave de tubarões no Atlântico Sudoeste?

Pergunta de regime: o sistema se comporta como determinístico, estocástico ou híbrido?

## 3) Hipóteses implícitas
- **H1**: dinâmica populacional depende de variáveis ambientais (ex.: SST).
- **H2**: pressão antrópica altera média e regime (aumenta estocasticidade).
- **H3**: respostas a perturbações ±Δ% são assimétricas (não-lineares).
- **H4**: sistema híbrido = componente determinístico médio + ruído ecológico relevante.

## 4) Restrições reais
- Hardware: notebook pessoal (CPU/RAM limitadas; GPU opcional).
- Tempo: semanas a poucos meses.
- Escopo: costa brasileira; 3–6 espécies; 5–10 presas/guildas; grid ~1°×1°; mensal.
- Evitar overengineering (sem GNNs pesadas/transformers/HPC).

## 5) O que o projeto NÃO é
- Não é estimativa absoluta de biomassa sem esforço amostral justificado.
- Não é identificação causal perfeita.
- Não é multimodal com visão.
- Não é global nem HPC.

## 6) Formulação do problema

### Entradas candidatas
- **Ambientais**: SST (principal), anomalia (ΔSST), tendência temporal local.
- **Presas/biodiversidade**: proxies via OBIS (contagem normalizada por célula/tempo; ajustar por esforço quando possível).
- **Antrópicas**: fishing_pressure (GFW ideal; FAO como alternativa coarse).
- **Ruído**: variabilidade intrínseca tratada como feature.

### Saídas esperadas
- Séries por espécie × célula: índice de ocorrência normalizado (proxy de abundância relativa).
- Derivadas: $P_{surv}(\delta X,T)$, tempo até colapso, entropia/variância do ensemble, sensitividades marginais $\partial Y/\partial X$.
- Diagnósticos: regime (determinístico / estocástico / híbrido).

### Natureza do sistema (tese de trabalho)
**Híbrido**: há uma dinâmica média estrutural (capturada por um modelo determinístico interpretável — NSSM) e variabilidade residual/bifurcações (capturadas por um DMM condicional).

## 7) Contrafactual (estilo Aphelion)

### Definição
Para intervenção $\delta X$ aplicada a um driver $X$ (ex.: SST, fishing_pressure):
- gerar trajetórias $Y'(t)$ condicionadas ao mesmo estado inicial $z_{t0}$ e ao driver modificado
  $$X'(t) = X(t)\cdot(1+\delta).$$
- comparar $Y'(t)$ com o factual $Y(t)$.

### Plausível vs inválido
- **Plausível**: dentro de limites físicos/ecológicos; preserva invariantes simples (não-negatividade, taxa máxima de variação).
- **Inválido**: extrapola para fora de limites observacionais; viola invariantes (ex.: abundância negativa, colapso instantâneo sem mecanismo).

### Métricas A/B
- $P_{surv}(\delta X,T)$
- $E[T_{collapse}|\delta X]$
- ΔAUC (diferença de área sob curva)
- entropia/variância das trajetórias
- sensitividades $\partial Y/\partial X$

### Incerteza e ruído
- **DMM**: ruído explícito no estado latente ($\sigma_z$) e/ou observação ($\sigma_y$), amostrado via Monte Carlo.
- **NSSM**: dinâmica média determinística; DMM explora trajetórias plausíveis. Ruído = feature.

## 8) Escopo (sweet spot)

### Protótipo rápido (fim de semana estendido)
- Definir grid 1°×1° na costa brasileira (lat 5°N→35°S, lon 50°W→30°W) e máscara por profundidade (<1000 m).
- Baixar/agregar SST mensal (NOAA OISST/ERSST).
- Extrair ocorrências OBIS para 3–4 espécies de tubarões + 5 presas/guildas e agregar por célula/mês.
- Fishing pressure coarse (FAO) ou subset GFW.
- Pipeline mínimo: pré-processamento → NSSM (LSTM pequeno) → DMM condicional → 100–500 trajetórias MC para 1–2 cenários Δ.
- Calcular outputs N1–N5.

### Fora do escopo
- Identificação causal completa.
- Modelos pesados (HPC).
- Biomassa absoluta sem esforço amostral robusto.
- Escala global.

## 9) Checklist antes de escolher arquitetura
- Região (bbox) e máscara de profundidade.
- Grid/resolução e nº de células válidas.
- Lista final de tubarões (3–6) e presas/guildas (5–10).
- Horizonte temporal e resolução (mensal).
- Fontes de dados confirmadas (NOAA/OBIS/GFW/FAO) e formatos.
- Proxy operacional de abundância (normalização e esforço).
- Definição de limiar de colapso/persistência.
- Capacidade computacional real (CPU/RAM/GPU) e tempo aceitável.
- Nº de trajetórias MC por cenário e seeds.
- Métricas prioritárias.
- Estratégia de validação (holdout temporal, ablations).
- Regularização/constraints (não-negatividade, taxa máxima de variação).
