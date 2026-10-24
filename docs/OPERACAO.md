# Operação local

## Configuração

Instale `requirements.txt`, copie `.env.example` para `.env` uma única vez e
preencha `LLM_API_KEY`. O arquivo privado não deve ser compartilhado. Variáveis
já exportadas no terminal prevalecem sobre `.env`. Reinicie o Streamlit após
alterar configurações, pois o orquestrador fica em cache.

| Variável | Uso |
| --- | --- |
| `LLM_API_KEY` | Chave da conta do provedor. Necessária para iniciar o atendimento real. |
| `LLM_BASE_URL` | Endpoint; padrão `https://api.groq.com/openai/v1`. |
| `LLM_MODEL` | Único modelo usado; padrão `openai/gpt-oss-20b`. |
| `LLM_OUTPUT_MODE` | `native` ou `structured`; padrão `native`. |
| `LLM_TIMEOUT` | Timeout por chamada, em segundos; padrão 30. |
| `LLM_MAX_RETRIES` | Recuperação genérica da aplicação; padrão 0, máximo 1. O SDK sempre usa 0. |
| `LLM_MAX_RATE_LIMIT_WAIT_SECONDS` | Teto da recuperação de 429; padrão 60, ajustado ao intervalo 20–120 segundos. |
| `DATA_DIR` | Diretório dos três CSVs. Caminhos relativos partem da raiz do projeto. |
| `GROQ_ORGANIZATION` | Identificador local para compartilhar contabilização entre chaves da mesma organização; padrão `default`. |
| `GROQ_RPM` / `GROQ_TPM` | Limites de referência; padrões 30 requisições e 8.000 tokens por minuto. Não criam fila preventiva. |
| `GROQ_ADMISSION_TIMEOUT_SECONDS` | Prazo para aguardar um cooldown já observado; padrão 60, entre 0 e 120 segundos. |
| `GROQ_MAX_COMPLETION_TOKENS` | Reserva e teto de saída; padrão 2.048, entre 256 e 8.192 tokens. |
| `GROQ_REASONING_EFFORT` | Para GPT-OSS no Groq: `low`, `medium` ou `high`; padrão `low`. Vazio usa o padrão do provedor. |
| `GROQ_RATE_LIMITS_JSON` | Ajustes locais de RPM, TPM e `output_tokens` por organização/modelo. |

Exemplo de ajuste específico, sem credenciais:

```dotenv
GROQ_RATE_LIMITS_JSON={"default":{"openai/gpt-oss-20b":{"rpm":30,"tpm":8000,"output_tokens":2048}}}
```

Os limites efetivos são os da conta do provedor. A configuração local não aumenta
cotas. A aplicação não muda automaticamente para outro modelo quando a cota acaba.

## Testar sem alterar a base de demonstração

No Linux/macOS, copie somente os três arquivos utilizados:

```bash
BANCO_DEMO_DIR=$(mktemp -d)
cp data/clientes.csv data/score_limite.csv data/solicitacoes_aumento_limite.csv "$BANCO_DEMO_DIR/"
DATA_DIR="$BANCO_DEMO_DIR" python -m streamlit run app.py
```

No PowerShell:

```powershell
$bancoDemoDir = Join-Path $env:TEMP ("banco-agil-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $bancoDemoDir | Out-Null
Copy-Item .\data\clientes.csv, .\data\score_limite.csv, .\data\solicitacoes_aumento_limite.csv -Destination $bancoDemoDir
$env:DATA_DIR = $bancoDemoDir
python -m streamlit run app.py
```

Depois de encerrar o servidor, `Remove-Item Env:DATA_DIR` restaura a configuração
padrão no PowerShell. No Linux/macOS, o prefixo `DATA_DIR=...` vale só para aquele
comando. A cópia começa no estado atual dos arquivos; não é uma restauração de
valores iniciais. **Novo atendimento** também não restaura os CSVs.

Para dados consistentes, não execute vários processos sobre o mesmo diretório
nem edite CSVs durante o atendimento. A reconciliação de solicitações acontece
na inicialização e nos fluxos de crédito.

## Interface e diagnóstico

O indicador “Aguarde um momento, estou trabalhando nisso.” aparece enquanto o
turno é processado, inclusive durante a pausa de cinco segundos para apresentar
a resposta. É possível escrever um rascunho; ele não é enviado automaticamente
quando o processamento termina. O envio e **Novo atendimento** ficam bloqueados
até a conclusão.

O diagnóstico técnico na barra lateral só fica disponível com
`BANCO_AGIL_ENV=development` e `BANCO_AGIL_ENABLE_DEBUG=1`. Os logs em `logs/`
são locais e não entram no pacote de entrega. CPF, nascimento, histórico e
mensagens do provedor exigem cuidado mesmo quando há sanitização; não publique
logs ou capturas sem revisar o conteúdo.

## Falhas comuns

| Situação | Orientação |
| --- | --- |
| Chave ausente | Confira o nome `LLM_API_KEY` no arquivo privado, sem imprimir seu valor. |
| Configuração alterada sem efeito | Pare e reinicie o Streamlit; confira variáveis exportadas no terminal. |
| Dependência não encontrada | Use o Python da `.venv` e instale o manifesto correspondente à aplicação ou aos testes. |
| HTTP 429 temporário | A aplicação faz no máximo uma recuperação dentro do prazo seguro. Se não recuperar, mantém a sessão em memória e informa indisponibilidade. |
| Cota diária esgotada | Aguarde a recuperação da cota da conta; não há troca automática de modelo. |
| Falha de geração ou timeout | Não suponha que uma operação falhou só porque faltou a mensagem final. Confira o resultado registrado antes de repetir um pedido. |
| Câmbio indisponível | A consulta depende das fontes externas e pode não ter cotação para a moeda pedida. |

## Gerar uma entrega revisável

```bash
python scripts/make_zip.py --list
python scripts/make_zip.py /tmp/banco-agil.zip
```

O empacotador usa uma lista positiva: aplicação, módulos Python, testes, três
CSVs, configuração de exemplo, dependências, scripts e documentação pública.
Ele recusa links simbólicos e só substitui um pacote anterior depois de validar
o novo. Arquivos locais fora da lista permanecem intactos.

Revise o estado e a origem dos CSVs antes de compartilhar o arquivo. O empacotador
inclui os dados atuais, sem limpá-los nem certificar que são adequados à publicação.
Gerar o ZIP não faz commit, push ou publicação.
