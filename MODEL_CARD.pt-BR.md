# Ficha do Modelo — Modelo de Risco de Crédito do Credit Intelligence AI

Seguindo o espírito de [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
(Mitchell et al., 2019).

## Detalhes do Modelo

- **Desenvolvido por**: Matheus Marcondes, como projeto de portfólio.
- **Tipo de modelo**: Classificação binária (previsão de default em 90 dias),
  dois candidatos treinados e comparados:
  1. **Regressão Logística** (`sklearn.linear_model.LogisticRegression`,
     `class_weight="balanced"`, `C=0.5`) — a baseline interpretável.
  2. **LightGBM** (`lightgbm.LGBMClassifier`, 400 estimadores, early
     stopping) — o desafiante não-linear.
- **Seleção do campeão**: pelo ROC-AUC no teste out-of-time (OOT), registrado
  em `models/model_metadata.json`. Na última execução completa de
  treinamento, a **Regressão Logística** foi selecionada como campeã (AUC
  OOT 0.836 vs 0.834 do LightGBM — um empate técnico, o que por si só é um
  achado relevante: veja "Campeão vs Desafiante" abaixo).
- **Calibração**: Platt scaling (`sigmoid`) ou regressão isotônica, ajustada
  apenas no **split de validação**; a plataforma usa sigmoid por padrão, a
  menos que a isotônica melhore o Brier score de validação em mais de 5%
  relativo, porque a natureza de função-degrau da regressão isotônica pode
  absorver silenciosamente pequenas variações de entrada nas ferramentas de
  simulação/stress-testing (uma escolha deliberada e documentada — veja
  `src/models/calibration.py`).
- **Versão**: atrelada a `models/model_metadata.json` (regenerado por
  `make train` + a etapa de calibração).

## Uso Pretendido

**Uso principal pretendido**: apoio à decisão para analistas de crédito e
gestores de risco investigando uma carteira sintética demonstrativa — nunca
decisão de crédito autônoma. Veja [`RESPONSIBLE_AI.pt-BR.md`](RESPONSIBLE_AI.pt-BR.md).

**Fora de escopo**: qualquer decisão de crédito real, qualquer implantação
sobre dados reais de clientes sem revalidação independente completa, e
qualquer uso do score/faixa como política comercial de precificação (ambos
são explicitamente demonstrativos).

## Dados de Treinamento

100.000 clientes sintéticos, divididos temporalmente (não aleatoriamente)
para evitar look-ahead bias — veja [`docs/TEMPORAL_VALIDATION.md`](docs/TEMPORAL_VALIDATION.md)
e [`DATA_CARD.pt-BR.md`](DATA_CARD.pt-BR.md):

| Split | Janela | Linhas |
|---|---|---|
| Treino | 2023-01 a 2024-06 | 63.234 |
| Validação | 2024-07 a 2024-12 | 20.180 |
| Teste (OOT) | 2025-01 a 2025-06 | 16.586 |

## Features

38 colunas no dataset de modelagem: 24 features brutas + 6 features de
engenharia + 2 categóricas (faixa etária, histórico de pagamento) +
identificadores/target/split — veja [`FEATURE_DICTIONARY.md`](FEATURE_DICTIONARY.md)
para a lista completa e as fórmulas. Nenhuma característica
protegida/sensível é usada, direta ou indiretamente como proxy (garantido
por `src/data/validation.py`).

## Métricas (reais, medidas — veja `reports/model_comparison.json`)

Avaliado com um conjunto completo de métricas, nunca apenas acurácia (a
acurácia é pouco informativa nesse problema desbalanceado com taxa-base de
≈8%):

| Métrica (teste OOT) | Regressão Logística (campeã) | LightGBM (desafiante) |
|---|---:|---:|
| ROC-AUC | **0.8357** | 0.8339 |
| Gini | 0.6715 | 0.6679 |
| Estatística KS | 0.5259 | 0.5243 |
| PR-AUC | 0.3308 | 0.3133 |
| Brier score (bruto, não calibrado) | 0.1816 | 0.1812 |
| Lift no decil superior | 3.98x | 3.97x |
| Recall no decil superior | 39.8% | 39.7% |
| Recall nos 2 decis superiores | 63.4% | 63.5% |

**Calibração** (modelo campeão, teste OOT): o Brier score melhorou de
**0.1816 (bruto)** para **0.0655 (com Platt scaling)**.

**Nota Campeão vs Desafiante**: os dois modelos ficam a ~0.002 de AUC um do
outro nos dados OOT. Este é um resultado realista e honesto para esse
conjunto de features — não indica um bug, e é um motivo legítimo para
preferir a Regressão Logística, mais simples e totalmente interpretável,
como campeã em um contexto sensível à governança, quando os demais fatores
estão praticamente equivalentes.

## Considerações Éticas

- Treinado exclusivamente com dados sintéticos; veja [`DATA_CARD.pt-BR.md`](DATA_CARD.pt-BR.md)
  para como o risco de default foi construído e por que o AUC foi
  deliberadamente reduzido de um valor inicial de ~0.95 (implausivelmente
  bom) para um valor realista de ~0.83-0.85.
- Nenhum atributo protegido/sensível está presente ou é utilizável, direta
  ou via proxy — veja [`RESPONSIBLE_AI.pt-BR.md`](RESPONSIBLE_AI.pt-BR.md).
- A faixa etária é usada como um segmento comportamental grosseiro; em uma
  implantação real, a pontuação baseada em idade pode estar sujeita a uma
  revisão regulatória específica de fair-lending que esta demonstração não
  realiza.

## Ressalvas e Recomendações

- **Reject inference**: o modelo é treinado apenas com contas originadas —
  não existe uma população de "solicitantes recusados", então o modelo pode
  não generalizar para uma verdadeira população through-the-door. Esta é
  uma limitação conhecida e não endereçada da demonstração (veja
  [`GOVERNANCE.pt-BR.md`](GOVERNANCE.pt-BR.md)).
- **Sem validação independente**: este modelo não passou por validação
  independente de modelo, revisão jurídica ou aprovação regulatória de
  qualquer tipo.
- **Score/faixa ilustrativos**: o score de 300-900 e as faixas A-E são uma
  transformação padrão de pontos-para-odds-duplos da PD, calibrada para
  pontos de referência escolhidos para esta demonstração — não um scorecard
  comercial real.
- **O SHAP explica o modelo bruto**: os valores SHAP são calculados sobre a
  função de decisão pré-calibração; a calibração é monotônica, então a
  direção/ranking dos fatores permanece válido, mas a magnitude exata do
  SHAP não mapeia 1:1 para a sensibilidade marginal da PD calibrada.
- Retreine e execute novamente todo o conjunto de avaliação (`make train`)
  sempre que o pipeline de features mudar, ou quando `detect_drift`
  reportar um PSI acima do limite "Monitorar" — veja
  [`GOVERNANCE.pt-BR.md`](GOVERNANCE.pt-BR.md).
