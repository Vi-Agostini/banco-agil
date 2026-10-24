# Testes locais

## Preparação e execução

Na raiz do projeto, com o ambiente virtual ativo:

```bash
python -m pip install -r requirements-dev.txt
python scripts/test_local.py
```

`requirements-dev.txt` inclui as dependências da aplicação e declara os pacotes
importados diretamente pelos testes: pytest, httpx2 e jsonschema. Não é necessário
configurar chave de API para a suíte.

O runner bloqueia resolução de nomes externos, conexões TCP e envio UDP pelas
operações de socket Python que ele protege. Loopback continua permitido para os
servidores HTTP locais usados em testes. Isso é uma proteção contra chamadas
acidentais da suíte, não uma sandbox de rede para código arbitrário ou processos
filhos. Não acrescente testes reais ou pagos a esse comando.

Os testes usam relógio, respostas HTTP e LLM simulados. As fixtures escrevem
clientes e solicitações em diretórios temporários. Os três CSVs da base servem
como entrada e não devem ser alterados pela suíte.

## Cobertura de comportamento

| Área | O que verificar |
| --- | --- |
| Triagem | Credenciais juntas ou separadas, três falhas, privacidade e encaminhamento após autenticação. |
| Crédito | Consulta, interpretação do valor, consentimento, faixas de score, reprovação, recibos e reconciliação. |
| Entrevista | Dados fora de ordem, confirmação, correção, próxima pergunta e retorno à reanálise. |
| Câmbio | Identificação da moeda, fontes simuladas, data de atualização, precisão e indisponibilidade. |
| LLM | Protocolos nativo e estruturado, schema/decoder, 400, 429, timeout, truncamento e recuperações limitadas. |
| Concorrência | Proteção da sessão, reservas compartilhadas no processo e ausência de dupla execução financeira. |
| Interface | Indicador temporário, rascunho editável, bloqueio de envio e retorno após conclusão ou falha. |

Exemplos de recortes:

```bash
python scripts/test_local.py tests/test_credit.py tests/test_financial_concurrency.py
python scripts/test_local.py tests/test_interview_protocol.py tests/test_agent_continuity.py
python scripts/test_local.py tests/test_generation_recovery.py tests/test_rate_limit.py
python scripts/test_local.py tests/test_ui_smoke.py tests/test_editable_composer.py
```

Uma execução aprovada termina com código de saída zero e sem testes com falha.
O total é informado pelo pytest para a versão executada; relatórios de campanhas
antigas não representam automaticamente a versão atual.

## Limites da validação

Os testes com SDK e HTTP simulados verificam requisições, respostas, estado e
efeitos financeiros. Eles não demonstram que o modelo real compreenderá todas
as mensagens ou respeitará o protocolo em toda geração. Também não validam cotas,
latência ou disponibilidade do Groq e das fontes reais de câmbio.

Os testes de Streamlit inspecionam widgets, estado e código da interface. Foco,
teclado, animação e aparência precisam de conferência em um navegador real.

Windows, macOS, múltiplos processos e sistemas de arquivos de rede não estão
homologados por uma execução local em Linux. A persistência continua limitada
ao modelo de CSV e ao lock de threads do processo.

## Conferência manual

Essas etapas usam serviços reais e podem consumir a cota do provedor. Elas não
são executadas pelo runner. Prepare uma cópia dos dados com `DATA_DIR`, conforme
o guia de operação, e confira nela os valores atuais antes de começar.

1. Autentique um cliente fictício com CPF e nascimento, juntos e em mensagens
   separadas. Em outra sessão, confira o encerramento após três pares inválidos.
2. Consulte o limite e compare a resposta com a cópia de `clientes.csv`.
3. Teste pedidos acima e abaixo do teto da faixa do cliente, com valores maiores
   que o limite atual. Confira a decisão e as gravações na cópia. Se o cliente já
   estiver no teto, escolha outro cenário adequado para o caso aprovado.
4. Após uma reprovação, aceite a entrevista, informe dados fora de ordem e faça
   uma correção. Confira o resumo, as confirmações, o score e a reanálise.
5. Consulte uma moeda e confira valor, fonte e atualização apresentados. Mude de
   assunto e encerre o atendimento explicitamente para verificar as transições.
6. Enquanto a resposta é preparada, digite um rascunho, tente enviá-lo e observe
   o indicador. Confira que o texto continua editável, o envio fica bloqueado e
   o rascunho não é enviado sozinho ao concluir.

