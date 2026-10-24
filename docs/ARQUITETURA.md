# Arquitetura do Banco Ágil

## Conversa e responsabilidades

O cliente vê um assistente. Internamente, o orquestrador mantém a sessão e
coordena quatro agentes:

| Agente | Escopo |
| --- | --- |
| Triagem | Saudação, identificação e direcionamento após autenticação. |
| Crédito | Consulta, proposta, aumento de limite e reanálise. |
| Entrevista | Coleta conversacional, confirmação dos dados e cálculo de score. |
| Câmbio | Identificação da moeda, consulta à API e apresentação da cotação. |

Prompts e contexto orientam a IA. O catálogo de ferramentas é filtrado pelo
agente, autenticação e etapa. O executor valida novamente as ações antes de
executar qualquer efeito. Uma instrução do modelo não concede uma permissão.

O estado da sessão guarda agente ativo, etapa, campos coletados e confirmados,
propostas, consentimentos e resultados de ferramentas. A troca de agente reduz
o histórico enviado ao LLM, preservando os dados relevantes no estado. Esse
estado vive na memória do processo; não há retomada integral da conversa após
reiniciar o servidor.

## Autenticação e consentimento

CPF e nascimento são normalizados e comparados com `clientes.csv`. A coleta e
a validação de credenciais ficam no código, enquanto a IA conduz o diálogo e
interpreta o assunto do cliente. O modelo não recebe uma ferramenta que lhe
permita autenticar arbitrariamente outra identidade. Após três tentativas
completas inválidas consecutivas, o atendimento é encerrado.

O consentimento fica vinculado aos dados e à operação apresentados. Um pedido
literal e inequívoco pode autorizar a solicitação; valores ou modalidades
ambíguos exigem esclarecimento. A confirmação de renda ou emprego pertence à
entrevista e não autoriza um aumento de limite. Correções invalidam a confirmação
anterior do dado alterado.

## Crédito e score

O Crédito consulta a faixa do score em `score_limite.csv`. O valor solicitado
precisa ser maior que o limite atual e caber na faixa para ser aprovado.
Pedidos reprovados também são registrados. Depois da entrevista confirmada, o
score é atualizado e o pedido anterior pode voltar para reanálise.

A fórmula adotada pelo projeto é:

```text
score = renda / (despesas + 1) * 30
        + peso_emprego + peso_dependentes + peso_dividas
```

| Componente | Pesos |
| --- | --- |
| Emprego | Formal: 300; autônomo: 200; desempregado: 0. |
| Dependentes | Zero: 100; um: 80; dois: 60; três ou mais: 30. |
| Dívidas | Possui: −100; não possui: 100. |

O resultado é arredondado e limitado a 0–1000. Entradas financeiras inválidas
são recusadas antes da gravação. Essa fórmula simples pode saturar o score com
renda alta em relação às despesas.

## Persistência e concorrência

| Arquivo | Conteúdo |
| --- | --- |
| `clientes.csv` | CPF, nome, nascimento, limite de crédito e score. |
| `score_limite.csv` | Score mínimo, máximo e limite de cada faixa. |
| `solicitacoes_aumento_limite.csv` | CPF, instante ISO 8601, limite anterior, valor solicitado e status. |

Os status canônicos são `pendente`, `aprovado` e `reprovado`. A posição da linha
identifica uma solicitação; a ordem precisa ser preservada para a retomada.
A gravação usa arquivo temporário no mesmo diretório e `os.replace`.

O serviço registra a solicitação, grava sua decisão e aplica o limite quando
aprovado. Recibos identificam tentativas interrompidas, e a reconciliação trata
solicitações pendentes e efeitos ainda não aplicados. Repetir uma geração da IA
não é autorização para repetir a operação financeira.

Um `RLock` protege as threads do processo. Não há lock entre processos, transação
entre CSVs ou sincronização com `fsync`. Execute uma única instância por
diretório de dados; gravações manuais e outros processos ficam fora dessa
proteção. Sessões e recibos em memória não substituem um registro transacional.

## Protocolos e falhas do LLM

`LLM_OUTPUT_MODE=native` usa ferramentas nativas e é o padrão.
`LLM_OUTPUT_MODE=structured` usa um envelope JSON Schema, com mensagens e ações
mutuamente exclusivas, `strict=true` e decoder local. O transporte estruturado
também tem recuperação limitada por função estrita, usando o mesmo contrato.
Os dois modos mantêm permissões e validações financeiras locais.

Todas as chamadas usam `LLM_MODEL`. O SDK mantém `max_retries=0`; as recuperações
pertencem à aplicação. Falhas recuperáveis de geração compartilham uma tentativa
por turno. Requisições ou schemas inválidos e respostas truncadas não recebem
repetição idêntica para tentar contornar a validação.

HTTP 429 temporário permite uma recuperação, com espera mínima de 20 segundos,
respeitando `Retry-After` maior dentro do teto configurado. A contabilização por organização/modelo é compartilhada
entre agentes e sessões do processo, mas não cria espera preventiva por estimativa
ou cabeçalho de sucesso.

O guard de respostas confronta alguns valores e afirmações com o estado e o
resultado das ferramentas. Ele reduz inconsistências conhecidas, mas não valida
todo o significado de um texto livre. Os testes simulados não comprovam a
compreensão ou a disponibilidade de um modelo real.

## Câmbio e interface

O serviço consulta AwesomeAPI e usa Frankfurter como alternativa. Valores
inválidos e falhas das fontes geram uma resposta controlada. A cotação tem
precisão de quatro casas e inclui fonte e atualização disponíveis; a data de
referência da fonte pode ser anterior ao momento da consulta.

A interface Streamlit processa cada turno em segundo plano. Enquanto o trabalho
está ativo, mostra os três pontos animados, permite digitar um rascunho e bloqueia
envio e **Novo atendimento**. A trava do servidor impede turnos simultâneos na
mesma sessão. Uma pausa fixa de cinco segundos antecede a exibição da resposta,
independentemente do cooldown de recuperação.

## Organização do código

- `agents`, `agent_context`, `orchestrator`, `pipeline` e `flow`: agentes e fluxo.
- `interceptors`, `session`, `interview` e `intents`: entrada e estado da conversa.
- `tools`, `tool_contracts` e `tool_executor`: contratos e execução autorizada.
- `credit_service`, `repository`, `scoring` e `exchange`: regras e integrações.
- `llm`, `structured_output` e `rate_limit`: protocolo, contabilização e recuperação.
- `response_guard`, `reply_recorder`, `messages` e `presentation`: respostas.
- `privacy`, `observability` e `provider_diagnostics`: privacidade e diagnóstico.
- `processing`, `composer`, `bootstrap` e `app.py`: execução e interface.
