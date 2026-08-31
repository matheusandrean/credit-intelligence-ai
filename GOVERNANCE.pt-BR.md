# Governança

Práticas de gestão de risco de modelo, governança de dados e gestão de
mudanças para o Credit Intelligence AI.

## 1. Ciclo de Vida do Modelo

```
gerar dados -> validar -> engenharia de features -> treinar (baseline + desafiante)
    -> avaliar (conjunto completo de métricas) -> calibrar -> selecionar campeão
    -> construir carteira pontuada -> monitorar drift
```

Implementado de ponta a ponta via `make data validate features train` e
rastreado no MLflow (backend local SQLite, `mlflow.db`) — cada execução
registra parâmetros, métricas para cada split, e o artefato do modelo
ajustado.

## 2. Processo de Campeão/Desafiante

- Tanto a Regressão Logística (baseline) quanto o LightGBM (desafiante) são
  sempre treinados e avaliados juntos (`src/models/train.py`).
- Regra de seleção do campeão: maior ROC-AUC no teste OOT, registrado em
  `models/model_metadata.json` e `reports/model_comparison.json` — nunca
  assumido silenciosamente.
- O método de calibração (Platt vs isotônica) também é selecionado por uma
  regra explícita e documentada, favorecendo suavidade a menos que a
  isotônica vença por uma margem material (veja [`MODEL_CARD.pt-BR.md`](MODEL_CARD.pt-BR.md)).

## 3. Cadência de Monitoramento e Limites

Population Stability Index (PSI), calculado tanto para as features de
entrada quanto para a distribuição da PD prevista pelo modelo ao longo do
tempo (`src/monitoring/drift.py`, ferramenta `detect_drift`):

| PSI | Status | Ação |
|---|---|---|
| < 0.10 | Estável | Nenhuma ação |
| 0.10 - 0.25 | Monitorar | Aumentar frequência de revisão |
| > 0.25 | Possível drift significativo | Escalar para revisão do modelo |

**Esses limites são convenções demonstrativas**, não validadas contra o
apetite de risco de nenhuma instituição real — veja
`configs/project_config.yaml` e ajuste conforme o framework de governança
de uma implantação real.

O monitoramento de carteira também acompanha curvas de taxa de
inadimplência por safra/MOB e migração de roll rate
(`src/analytics/vintage.py`, `src/analytics/roll_rate.py`) para distinguir
deterioração genuína de efeitos de crescimento/amadurecimento da carteira.

## 4. Gatilhos de Retreinamento

Conceitualmente, nesta demonstração, o retreinamento é acionado por:

1. `detect_drift` reportando "Monitorar" ou pior em uma feature-chave ou na
   distribuição de PD prevista.
2. Um novo ciclo completo de geração de dados sintéticos (seed/volume
   diferente).
3. Uma mudança documentada no pipeline de engenharia de features.

Qualquer um desses casos exige reexecutar todo o conjunto de avaliação
(`make train` + calibração) antes que um novo modelo seja considerado para
o status de campeão — nenhuma mudança é implantada sem uma nova comparação
lado a lado.

## 5. Limitações Conhecidas (Reject Inference)

Esta plataforma treina apenas com contas originadas — não existe uma
população de "solicitantes recusados", o clássico problema de reject
inference em credit scoring. **Esta demonstração não implementa nem alega
implementar reject inference.** Isso é divulgado aqui por transparência,
conforme [`MODEL_CARD.pt-BR.md`](MODEL_CARD.pt-BR.md) e
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## 6. Validação Independente

Nenhum componente desta plataforma — modelo, calibração ou limites — passou
por validação independente, revisão jurídica ou aprovação regulatória. Veja
o aviso de isenção de responsabilidade no [`LICENSE`](LICENSE) do
repositório.

## 7. Governança de Dados

- Todos os dados são sintéticos, gerados por código versionado e com seed
  fixa (`src/data/generate_synthetic_credit_data.py`) — veja
  [`DATA_CARD.pt-BR.md`](DATA_CARD.pt-BR.md).
- Verificações automatizadas de qualidade de dados
  (`src/data/validation.py`) rodam antes de cada ciclo de treinamento e
  bloqueiam em caso de: IDs duplicados, presença de qualquer atributo
  protegido, e falhas de consistência entre campos.
- Nenhum dataset — bruto, processado, ou um binário de modelo treinado — é
  versionado no git; todo artefato é regenerado localmente a partir do
  código (veja [`SECURITY.md`](SECURITY.md)).

## 8. Disciplina de Change Log

Os commits neste repositório são delimitados por etapa do pipeline (`feat:
add synthetic credit portfolio generator`, `feat: add credit risk modeling
and probability calibration`, ...) de forma que a evolução de cada
componente relevante para governança (modelo, calibração, limites) seja
revisável de forma independente no `git log`.
