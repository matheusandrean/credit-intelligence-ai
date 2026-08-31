# Ficha de Dados — Carteira Sintética do Credit Intelligence AI

Seguindo o espírito de [Datasheets for Datasets](https://arxiv.org/abs/1803.09010),
adaptado para um dataset sintético demonstrativo de risco de crédito.

## Motivação

**Por que este dataset foi criado?** Para viabilizar um projeto de
portfólio que demonstra uma plataforma completa de Credit Intelligence
(engenharia de dados, modelagem de risco de crédito, IA explicável, IA
generativa) sem usar nenhum dado pessoal ou financeiro real.

**Quem o criou?** Gerado programaticamente por
`src/data/generate_synthetic_credit_data.py`, escrito por Matheus Marcondes
para este projeto.

## Composição

- **100.000 clientes sintéticos**, uma linha cada, em
  `data/raw/customer_portfolio.parquet` (31 colunas — veja
  [`FEATURE_DICTIONARY.md`](FEATURE_DICTIONARY.md)).
- **1.542.441 linhas de cliente-mês** em
  `data/raw/monthly_performance.parquet`, capturando a trajetória de faixa
  de inadimplência por conta (limitada a 18 meses de histórico) para as
  análises de safra/MOB e roll rate.
- Todo cliente é inteiramente fictício. Os IDs são sequenciais
  (`CUST_000001`...`CUST_100000`) e não carregam nenhum significado do
  mundo real.
- Nenhuma característica protegida/sensível existe em nenhum ponto do
  schema (veja `src/data/schema.py::PROHIBITED_PROTECTED_ATTRIBUTES`), e
  isso é garantido por uma verificação automatizada de qualidade de dados
  (`src/data/validation.py`), não apenas uma promessa no momento da
  geração.

## Processo de Coleta

Não existe "coleta" — cada linha é gerada por `generate_customer_snapshot()`
e `generate_monthly_performance_panel()` usando o gerador de números
pseudoaleatórios do `numpy` com uma seed fixa (`42`), de forma que o
dataset é exatamente reproduzível. As relações estatísticas entre os
fatores de risco e o target são construídas deliberadamente (veja abaixo);
elas não são copiadas ou derivadas de nenhum dataset, instituição ou
indivíduo real.

## Escolhas de Design para Realismo

- **Taxa de default**: calibrada para ≈7,5% no nível da carteira (intercepto
  ajustado automaticamente), alinhada com carteiras típicas de crédito ao
  consumidor não garantido.
- **Relações de risco monotônicas**: a taxa de default aumenta
  monotonicamente com o quartil de DTI (observado: 0,96% → 3,48% → 7,23% →
  18,3% do quartil mais baixo para o mais alto), e da mesma forma com
  utilização de crédito, atrasos de pagamento e sinalizador de default
  anterior — garantido por verificações automatizadas de monotonicidade em
  `tests/unit/test_generate_synthetic_data.py`.
- **Ruído e imperfeição**: valores ausentes (2-4% em vários campos),
  outliers (0,4-0,6% em renda/saldo/utilização) e ruído latente irredutível,
  de forma que o problema de classificação resultante seja realista
  (ROC-AUC out-of-time ≈0,83-0,85 para o modelo campeão, não um implausível
  ~0,95+).
- **Estrutura temporal**: `observation_date` cobre de 2023-01 a 2025-06;
  `origination_date` é derivada de um tempo de conta com distribuição gama,
  de forma que a carteira inclua tanto contas novas quanto contas maduras.
- **Narrativa de deterioração da carteira**: uma deriva lenta e deliberada
  para cima no fator de risco latente subjacente ao longo do tempo
  calendário, de forma que períodos de observação mais recentes mostrem, em
  média, comportamento levemente pior — sustentando a narrativa de
  monitoramento/drift da demonstração (veja
  [`docs/PORTFOLIO_STORY.md`](docs/PORTFOLIO_STORY.md)).

## Usos

**Uso pretendido**: demonstração de portfólio/educacional de uma plataforma
de risco de crédito + IA generativa. **Não pretendido para**, e nunca deve
ser usado para: decisões de crédito reais, treinar um modelo implantado
sobre clientes reais sem uma revalidação completa em dados reais, ou
qualquer inferência sobre indivíduos reais.

## Distribuição

Os arquivos de dados brutos e processados **não são versionados no
repositório git** (veja `.gitignore`) — eles são regenerados localmente via
`make data` / `make features`. Isso mantém o repositório pequeno e deixa
explícito que cada recrutador/revisor executa exatamente o mesmo código de
geração, em vez de confiar em um arquivo estático.

## Manutenção

Regenerar com um `random_seed` ou `n_customers` diferente em
`configs/project_config.yaml` produz um novo dataset sintético, igualmente
válido, com as mesmas propriedades estatísticas.
