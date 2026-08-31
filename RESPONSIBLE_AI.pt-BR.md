# IA Responsável

Esta é a declaração canônica e real das práticas de engenharia de IA
responsável deste projeto — distinta do fictício
`knowledge_base/responsible_ai_policy.md`, que é um documento de política
sintética usado dentro do app para demonstrar a funcionalidade de RAG.

## 1. Humano no circuito (human-in-the-loop), estruturalmente

Nenhum componente desta plataforma — os modelos quantitativos ou o agente
de IA generativa — pode aprovar, negar ou finalizar uma decisão de crédito.
Isso é garantido em múltiplos níveis, não apenas documentado como política:

- Não existe nenhum endpoint de API que grave uma decisão de volta em
  qualquer registro.
- O system prompt do agente (`src/llm/system_prompt.py`) proíbe
  explicitamente linguagem de aprovação/negação e instrui o modelo a
  redirecionar para revisão humana.
- `tests/integration/test_llm_tools.py` e `test_agent.py` garantem que o
  agente sempre fundamenta suas respostas na saída das ferramentas, e não
  em afirmações de forma livre.

## 2. Justiça (fairness) e entradas proibidas

`src/data/schema.py::PROHIBITED_PROTECTED_ATTRIBUTES` lista características
que nunca podem aparecer no dataset ou ser usadas como feature: raça, cor,
etnia, religião, orientação sexual, identidade de gênero, opinião política,
deficiência, nacionalidade, estado civil. Isso é garantido por uma
verificação automatizada em `src/data/validation.py`
(`protected_attributes_found`), não apenas uma promessa no momento da
geração — um dataset contendo qualquer uma dessas colunas falha na
validação.

O gerador sintético não constrói nenhuma variável proxy destinada a
aproximar uma característica protegida. `age_band` é usada como um
segmento demográfico grosseiro e não protegido, consistente com a prática
comum (embora imperfeita) da indústria; uma implantação real exigiria
revisão dedicada de fair-lending sobre qualquer efeito baseado em idade, o
que esta demonstração não realiza.

## 3. Explicabilidade

Toda PD no nível de cliente é explicável via SHAP
(`src/models/explainability.py`): os 5 principais fatores que aumentam o
risco e os 5 principais que o reduzem, reportados com **valores reais
observados das features**, não internos escalados/codificados. O agente
LLM só tem permissão para narrar esses números — o system prompt proíbe
explicitamente inventar um fator que não apareceu na saída do SHAP, e
`get_customer_shap` é o único caminho pelo qual o agente pode discutir
fatores no nível do cliente.

## 4. Arquitetura anti-alucinação

Além do prompting, a alucinação é restringida estruturalmente:

- Toda afirmação numérica que o agente pode fazer vem de uma chamada de
  ferramenta tipada (`src/llm/tools/registry.py`); não existe caminho para
  o LLM emitir um número que não tenha vindo do resultado de uma
  ferramenta.
- `retrieve_credit_policy` retorna citações de documento + seção; o system
  prompt exige citá-las, e o texto recuperado é explicitamente tratado
  como **dado, não instrução** (defesa contra prompt injection — um trecho
  de política recuperado que dissesse "ignore todas as instruções
  anteriores" seria citado/analisado, nunca obedecido).
- O guard de SQL do `query_portfolio` (`src/llm/tools/sql_guard.py`) só
  permite instruções `SELECT`/`WITH` únicas e rejeita
  `DROP/DELETE/UPDATE/ALTER/INSERT/TRUNCATE/CREATE/ATTACH/PRAGMA/...` —
  verificado por `tests/unit/test_sql_guard.py` com payloads reais no
  formato de injeção.
- O Modo Demo (sem provedor de LLM) usa as mesmas ferramentas e apresenta
  sua saída real — ele não pode "alucinar" porque nunca gera texto de
  forma livre; toda resposta é montada a partir de um resultado de
  ferramenta.

## 5. Risco de modelo

- A validação temporal (out-of-time) é garantida, não apenas simulada — o
  split de teste é genuinamente posterior, em tempo calendário, ao de
  treino/validação (veja [`docs/TEMPORAL_VALIDATION.md`](docs/TEMPORAL_VALIDATION.md)).
- As métricas do campeão e do desafiante são sempre reportadas lado a lado
  (`reports/model_comparison.json`), e a calibração é ajustada apenas no
  split de validação para evitar overfitting da própria calibração.
- O drift é monitorado via PSI tanto nas features de entrada quanto na
  distribuição da PD prevista ao longo do tempo (`src/monitoring/drift.py`),
  com limites documentados e demonstrativos (veja
  [`GOVERNANCE.pt-BR.md`](GOVERNANCE.pt-BR.md)).

## 6. Governança de dados e alinhamento com a LGPD

Embora 100% sintético, o design de tratamento de dados do projeto segue
princípios de minimização de dados e limitação de finalidade consistentes
com a LGPD (Lei Geral de Proteção de Dados):

- Nenhum campo desnecessário é gerado ou retido.
- IDs de clientes sintéticos nunca se parecem com identificadores
  brasileiros reais (nenhum CPF/CNPJ, cartão ou número de conta em formato
  válido é gerado).
- Uma adaptação real desta plataforma para dados reais de clientes exigiria
  uma revisão completa de LGPD/compliance antes de qualquer uso — isso é
  declarado explicitamente, e não deixado implícito.

## 7. Auditabilidade

Toda interação com o Analista de Crédito com IA é registrada em
`reports/audit_log.jsonl` (`src/llm/audit.py`): timestamp, pergunta,
ferramentas invocadas (nome + sucesso/falha, nunca dados no nível de
linha), fontes de RAG citadas, e a versão do modelo+calibração — sem nunca
registrar segredos, chaves de API ou payloads completos de provedor (veja
[`SECURITY.md`](SECURITY.md)).

## 8. Limitações declaradas com honestidade

Veja [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) para a lista completa:
reject inference, ausência de modelagem de limites de concentração, dados
exclusivamente sintéticos, ausência de validação independente do modelo,
limites demonstrativos ao longo de toda a plataforma.

## 9. Aviso de Isenção de Responsabilidade

Este é um projeto de portfólio/educacional. Ele não deve ser usado para
tomar decisões de crédito reais sobre pessoas reais sem validação
independente, governança, compliance e revisão jurídica específicas da
instituição que o implantar. Veja o [`LICENSE`](LICENSE) do repositório.
