# 🏦 Banco Ágil — Atendimento bancário com agentes de IA

## 📌 Visão geral do projeto

Assistente bancário desenvolvido como desafio técnico, com Python, Streamlit e
Groq. Reúne autenticação, crédito, entrevista financeira e câmbio em uma conversa.
O cliente vê um único assistente; Triagem, Crédito, Entrevista e Câmbio dividem
o atendimento internamente, conforme o assunto e a etapa.

A ideia foi usar a IA para entender o que o cliente precisa e conduzir o diálogo.
Autenticação, validação de consentimento, cálculos, aprovação de crédito e gravação
dos dados ficam sob controle do código. O projeto é demonstrativo, usa cadastros
fictícios e não se destina à operação bancária real.

## 💬 Funcionalidades implementadas

| Agente | Responsabilidade |
| --- | --- |
| Triagem | Coletar CPF e nascimento, autenticar no CSV e encaminhar o assunto. Três tentativas completas inválidas consecutivas encerram a sessão. |
| Crédito | Consultar o limite, esclarecer o valor solicitado e avaliar aumentos pelas faixas de score. |
| Entrevista | Coletar e confirmar renda, emprego, despesas, dependentes e dívidas, calcular o score e retornar ao Crédito para reanálise. |
| Câmbio | Consultar a moeda por API externa e informar valor, fonte e atualização disponíveis. |

Solicitações de aumento ficam registradas mesmo quando reprovadas. Uma aprovação
atualiza o limite. Após uma reprovação, o cliente pode aceitar a entrevista para
reavaliar o score e retornar ao Crédito para nova análise. A entrevista confirmada
atualiza o score; não autoriza, por si só, uma operação financeira.

O cliente pode encerrar a conversa a qualquer momento.

## 🚀 Tutorial de execução e testes

Você precisa do **Python 3.11 ou superior**, de uma chave de API do **Groq** e de acesso à internet para conversar com os agentes. O projeto foi validado localmente com Python 3.12 em Linux; os comandos de Windows e macOS foram conferidos, mas não executados nesses sistemas.

Abra um terminal **na pasta do projeto** (a mesma que contém `app.py` e `requirements.txt`) e siga os passos do seu sistema operacional.

### 1. Preparar o ambiente

**Linux e macOS (Terminal):**

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
if [ ! -f .env ]; then cp .env.example .env; fi
```

Se o sistema disponibilizar apenas `python`, use esse comando no lugar de `python3`, desde que seja a versão 3.11 ou superior.

**Windows (PowerShell):**

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Se o comando `py` não estiver disponível, confira `python --version` e use `python -m venv .venv`. Se a política do PowerShell bloquear a ativação, não é necessário alterá-la: utilize `.\.venv\Scripts\python.exe` no lugar de `python` nos comandos de instalação e execução, sem ativar o ambiente virtual. Por exemplo: `.\.venv\Scripts\python.exe -m streamlit run .\app.py`.

Os comandos criam o ambiente virtual, instalam as dependências e copiam `.env.example` para `.env` **somente se o `.env` ainda não existir**.

### 2. Configurar a chave do Groq

1. Acesse o [console do Groq](https://console.groq.com/keys), entre na sua conta e crie uma chave de API, caso ainda não tenha uma.
2. No VS Code, abra o arquivo **`.env` na raiz do projeto**. Se estiver oculto no explorador, você também pode abri-lo pelo terminal com `code .env` (ou com seu editor preferido).
3. Encontre a linha `LLM_API_KEY` e **substitua somente o valor** pela sua chave:

   ```dotenv
   LLM_API_KEY=sua_chave_do_groq_aqui
   LLM_BASE_URL=https://api.groq.com/openai/v1
   LLM_MODEL=openai/gpt-oss-20b
   LLM_OUTPUT_MODE=structured
   ```

4. Salve o `.env`. O modo `structured` foi utilizado na homologação e já está definido no `.env.example` da entrega.

**Para trocar a chave depois:** abra novamente o `.env`, substitua o valor de `LLM_API_KEY`, salve e reinicie o Streamlit. **Não é necessário recriar o ambiente virtual ou reinstalar as dependências.** Alterar o `.env.example` não atualiza um `.env` que já existe.


### 3. Iniciar o atendimento

Com o ambiente virtual ativo, execute:

**Linux e macOS:**

```bash
python -m streamlit run app.py
```

**Windows (PowerShell):**

```powershell
python -m streamlit run .\app.py
```

O Streamlit mostrará um endereço local no terminal (normalmente `http://localhost:8501`). Abra esse endereço no navegador e inicie a conversa. Para parar o aplicativo, volte ao terminal e pressione **Ctrl + C**.

Os atendimentos utilizam a cota da conta Groq configurada e dependem de conexão com o modelo e com as APIs de câmbio. **As operações podem modificar os CSVs**; a seção de dados de demonstração explica como trabalhar com uma cópia.

### Configurações principais

| Variável | Para que serve |
| --- | --- |
| `LLM_API_KEY` | Chave privada do Groq, configurada no `.env`. |
| `LLM_BASE_URL` | Endereço da API do Groq. |
| `LLM_MODEL` | Modelo utilizado; o exemplo usa `openai/gpt-oss-20b`. |
| `LLM_OUTPUT_MODE` | Protocolo de resposta; use `structured` para reproduzir a homologação. |
| `DATA_DIR` | Diretório opcional dos três CSVs; sem ajuste, usa `data/`. |
| `LLM_TIMEOUT` | Tempo limite de cada chamada ao modelo, em segundos; exemplo: `30`. |

No modo `structured`, as respostas seguem um JSON Schema com validação local de mensagens e ações. O modo `native` utiliza as ferramentas nativas do provedor. Ambos respeitam os controles de autenticação, escopo e operações financeiras.

Para outras opções e diagnóstico de problemas, consulte o [guia de operação](docs/OPERACAO.md). Os comandos de execução dos testes estão na seção **Testes locais**.

## 👤 Dados de demonstração

O pacote contém três cadastros fictícios e nenhuma solicitação de aumento.
Após iniciar o sistema, informe o CPF e a data de nascimento de um dos clientes
abaixo para autenticar e experimentar o atendimento. Não utilize dados reais.

| Cliente fictício | CPF | Nascimento | Score inicial | Limite atual | Máximo pela faixa atual |
| --- | --- | --- | --- | --- | --- |
| Ana Demonstração | 11144477735 | 12/05/1990 | 450 | R$ 2.000,00 | R$ 3.000,00 |
| Bruno Demonstração | 52998224725 | 30/11/1985 | 650 | R$ 5.000,00 | R$ 8.000,00 |
| Carla Demonstração | 39053344705 | 23/07/1992 | 780 | R$ 10.000,00 | R$ 15.000,00 |

Os valores são iniciais: operações aprovadas alteram o limite e uma entrevista
concluída pode alterar o score. A aprovação exige consentimento vinculado ao valor
solicitado e respeita o máximo da faixa correspondente ao score no momento da
análise. Propostas ambíguas precisam ser esclarecidas e confirmadas.

Faixas definidas em `score_limite.csv`:

| Score | Limite máximo |
| --- | --- |
| 0 a 299 | R$ 1.000,00 |
| 300 a 499 | R$ 3.000,00 |
| 500 a 699 | R$ 8.000,00 |
| 700 a 849 | R$ 15.000,00 |
| 850 a 1000 | R$ 30.000,00 |

`solicitacoes_aumento_limite.csv` começa apenas com o cabeçalho das cinco colunas
esperadas pela aplicação. Os arquivos usam UTF-8, vírgula como separador, datas
armazenadas em `AAAA-MM-DD` e valores monetários com ponto decimal.

Para preservar esses valores ao experimentar, copie `data/` para outro diretório
antes do primeiro atendimento e configure `DATA_DIR=./data-demo` no `.env`.
Exemplos para criar essa cópia, na raiz do projeto:

Linux e macOS:

```bash
cp -R data data-demo
```

PowerShell:

```powershell
Copy-Item -Path .\data -Destination .\data-demo -Recurse
```

Use uma pasta `data-demo` que ainda não exista. As operações passam a gravar nela,
mantendo `data/` como referência inicial. **Reinicie o Streamlit após configurar `DATA_DIR`**,
pois uma sessão já em execução pode continuar usando a configuração anterior. Para restaurar
os dados, encerre o aplicativo e substitua os CSVs de `data-demo/` pelas cópias iniciais de `data/`.
Se a base inicial também foi alterada, extraia novamente os três CSVs do ZIP original.
**Novo atendimento** reinicia apenas a conversa, não os arquivos.

### Demonstração rápida

Com a base inicial da Ana, envie uma mensagem por vez e aguarde a resposta:

1. `Meu CPF é 11144477735 e nasci em 12/05/1990.`
2. `Qual é meu limite atual?`
3. `Quero aumentar meu limite total para R$ 2.500,00.`
4. Se houver pedido de confirmação, envie **confirmo** após conferir o valor.
5. `Qual é a cotação do dólar americano em reais?`
6. `Quero encerrar o atendimento.`

Na base inicial, o limite consultado é R$ 2.000,00 e R$ 2.500,00 cabe na faixa
do score 450. O pedido de aumento é registrado no CSV e, se aprovado, atualiza
o limite. Para experimentar a entrevista, solicite um limite acima de R$ 3.000,00
e aceite a reavaliação oferecida após a reprovação; responda às perguntas e confira
os dados antes de confirmá-los.

## 🧩 Arquitetura do sistema

O orquestrador coordena os quatro agentes e oferece apenas as ferramentas
permitidas para a etapa atual. O estado da sessão mantém agente ativo, dados
coletados, propostas e consentimentos nas transições. O executor confere os
contratos e as permissões antes de executar uma ferramenta.

`clientes.csv` guarda cadastro, limite e score; `score_limite.csv` define as
faixas de crédito; `solicitacoes_aumento_limite.csv` registra os pedidos e suas
decisões. As gravações usam arquivo temporário e `os.replace`, com um `RLock`
compartilhado pelas threads do processo. Isso não fornece transações entre CSVs nem
coordenação entre processos.

O contrato estruturado separa mensagens de ações e valida os argumentos. O texto
gerado pela IA não substitui o consentimento do cliente nem determina a aprovação.

Durante o processamento, o indicador de três pontos fica na conversa. É possível
digitar a próxima mensagem, mas o envio e **Novo atendimento** ficam bloqueados.
Uma proteção no servidor também impede processamento simultâneo da mesma sessão.
A interface mantém uma pausa fixa de cinco segundos antes de exibir cada resposta
processada, independente do cooldown após erro do provedor.

Veja os contratos, a fórmula do score e os limites em
[Arquitetura](docs/ARQUITETURA.md).

## 🛠️ Desafios enfrentados e como foram resolvidos

- **HTTP 400 e formato das respostas:** adotei JSON estruturado com `strict=true`
  e decoder local para reduzir respostas incompatíveis com o contrato. Mensagens
  e ações são mutuamente exclusivas. Falhas recuperáveis de geração compartilham
  uma nova tentativa por turno; requisições ou schemas inválidos não são repetidos
  de forma idêntica. Isso reduz algumas falhas de formato, mas não elimina todas as causas de HTTP 400.
- **HTTP 429 e limites do provedor:** a aplicação permite uma recuperação, com
  espera mínima de 20 segundos e respeito a `Retry-After` dentro do teto configurado.
  O SDK usa `max_retries=0` para evitar tentativas sobrepostas. Não há fila preventiva
  por estimativa de tokens; o cooldown depende de erro observado. Se a cota continuar
  indisponível, o cliente recebe uma mensagem controlada.
- **Continuidade entre agentes e falhas de IA:** agente ativo, etapa, dados,
  propostas e consentimentos ficam no estado da sessão. As transições preservam
  essas informações, e uma falha de geração não apaga resultados já confirmados
  das ferramentas. A continuidade depende da sessão em memória.
- **Segurança financeira e idempotência:** autenticação, escopo e consentimento
  vinculado à operação são conferidos antes dos efeitos financeiros. Recibos e
  reconciliação reconhecem operações já aplicadas ou interrompidas, para que uma
  nova geração de texto não repita a operação. Esses controles mantêm os limites
  do armazenamento em CSV e da coordenação por processo descritos abaixo.

## ⚙️ Escolhas técnicas e justificativas

| Escolha | Justificativa |
| --- | --- |
| Python e Streamlit | Manter regras, integrações e uma interface local para testar o atendimento no mesmo projeto. |
| Orquestração própria com quatro agentes | Deixar escopos, transições e estado explícitos, sem acrescentar um framework de agentes ao fluxo. |
| Groq com cliente compatível com a API da OpenAI | Concentrar a integração em um cliente configurável, com modelo, timeout e protocolo definidos na configuração. |
| CSV | Atender ao contrato do desafio e facilitar a inspeção de cadastros e decisões, preservando suas limitações de persistência e concorrência. |

## 🧪 Testes locais

Na raiz do projeto, com o ambiente virtual ativo, use no Linux e no macOS:

```bash
python -m pip install -r requirements-dev.txt
python scripts/test_local.py
```

No PowerShell:

```powershell
python -m pip install -r .\requirements-dev.txt
python .\scripts\test_local.py
```

Sem ativação no Windows, substitua `python` por
`.\.venv\Scripts\python.exe`. O manifesto de desenvolvimento também inclui as
dependências da aplicação. Não é necessário configurar uma chave para a suíte.

O comando usa relógio, HTTP e LLM simulados e bloqueia conexões externas feitas
pelos sockets Python comuns. Os testes que precisam de HTTP local usam loopback.
As fixtures financeiras gravam somente em diretórios temporários.

O relatório de validação local registra **1.664 testes aprovados**, tanto no
projeto quanto no pacote extraído. A suíte cobre autenticação, escopos,
consentimento, crédito, entrevista, câmbio, transições, recuperação de falhas,
concorrência e interface.

Esse resultado vem de simulações, distintas da homologação manual com o Groq
real em modo `structured`. Não comprova a qualidade de toda conversa nem elimina
falhas do provedor. A execução local registrada foi em Linux; os comandos de
Windows e macOS não representam homologação nessas plataformas. Veja os recortes
e os limites em [Testes](docs/TESTES.md).

## ⚠️ Limitações

- Sessões, recibos em memória, locks e contabilização de uso pertencem ao processo.
  Use uma única instância da aplicação sobre o mesmo diretório de dados.
- CSV não substitui um banco transacional. Uma queda pode exigir reconciliação,
  e não há garantia de durabilidade contra queda de energia.
- O guard de respostas detecta alguns valores e afirmações inconsistentes, mas
  não oferece validação semântica completa do texto gerado pela IA.
- A cotação é a última disponibilizada pela fonte, com data e horário quando
  fornecidos; não é garantia de preço de execução ou atualização instantânea.
- Erros, cotas e indisponibilidade do provedor podem impedir a continuação de
  um atendimento. O contexto da sessão é preservado enquanto o processo existir.


## 📚 Documentação complementar

- [Arquitetura](docs/ARQUITETURA.md): agentes, contratos, score, persistência e recuperação.
- [Operação](docs/OPERACAO.md): configuração, cópias dos dados e falhas comuns.
- [Testes](docs/TESTES.md): execução, cenários cobertos e limites das simulações.
