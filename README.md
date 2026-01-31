# nova-selachiia

Projeto de pesquisa (enxuto e reproduzível) para estudar como mudanças em drivers ambientais e antrópicos alteram a dinâmica temporal, persistência e risco de colapso de espécies de Selachiia na costa brasileira.

## Pergunta central
Se variáveis ambientais e/ou antrópicas mudarem em Δ% (intervenção contrafactual), como isso altera a dinâmica temporal, probabilidade de sobrevivência e padrões de colapso de espécies-chave de tubarões no Atlântico Sudoeste? O sistema é determinístico, estocástico ou híbrido?

## Ideia do sistema (híbrido)
- **NSSM (determinístico)**: aprende a dinâmica média (baseline interpretável) por célula espacial e tempo.
- **DMM condicional (estocástico)**: adiciona variabilidade ecológica explícita para gerar ensembles Monte Carlo, incluindo trajetórias contrafactuais sob intervenções $X'(t)=X(t)\cdot(1+\delta)$.

## Dados (alvo)
- **Ambiental**: SST (principal), anomalia/ΔSST (opcional), tendência local.
- **Biodiversidade/presas**: proxies via ocorrências OBIS agregadas (célula × mês) para 5–10 presas/guildas.
- **Antrópico**: fishing_pressure (GFW ideal; FAO como alternativa coarse).

## Outputs
- Séries por espécie × célula: índice de ocorrência normalizado (proxy de abundância relativa).
- Métricas: $P_{surv}(\delta X,T)$, $E[T_{collapse}|\delta X]$, ΔAUC, variância/entropia de trajetórias, sensitividades $\partial Y/\partial X$.

## Como rodar (ambiente)
Este projeto usa **Poetry**.

- Instalar deps base: `poetry install`
- Instalar com extras (dados/plot/ML):
  - Dados: `poetry install -E data`
  - Plot: `poetry install -E plot`
  - ML (Torch): `poetry install -E ml`
  - Combinar extras: `poetry install -E data -E plot -E ml`
- Ferramentas de dev (notebooks/lint): `poetry install --with dev`

Dica (opcional): para criar o venv dentro do repo, use `poetry config virtualenvs.in-project true`.

## Artefatos principais
- Documento do inventário: veja [docs/project_inventory.md](docs/project_inventory.md)
- Notebook inicial: [notebooks/01_project_inventory.ipynb](notebooks/01_project_inventory.ipynb)
