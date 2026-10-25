# TCC — Quantificação do vazamento por imagem e por paciente em avaliação few-shot de tecidos histológicos H&E

> Documento-base do projeto. Reúne o que foi alinhado com o orientador, o que foi
> analisado no artigo e no código de referência, e todas as decisões tomadas até
> 05/10/2026. Status de cada decisão: **[DECIDIDO]**, **[PADRÃO]** (valor
> recomendado, mudar só com motivo) ou **[PENDENTE]**.

- **Aluno:** João Gabriel Santos Nunes
- **Orientador:** Prof. Marcelo Zanchetta do Nascimento
- **Alinhamento 1:** 10/09/2026 (transcrição em `TCC2 - Joao Gabriel Nunes - Alinhamento 1.docx`)

---

## 1. Resumo em uma frase

Medir **quanto o desempenho de um método few-shot em histologia (BreakHis) fica
artificialmente alto quando os dados são separados por imagem em vez de por
paciente**, e como esse efeito varia com a **magnificação** e com o **número de
exemplos por classe (k-shot)**.

O trabalho **não propõe método novo**: o método do artigo base é mantido fixo
(variável de controle) e o que se investiga é o protocolo de separação dos dados.

---

## 2. Contexto e origem da ideia

### 2.1 O que o orientador pediu (alinhamento de 10/09)

- Artigo base: MetaChest (few-shot em raio-X de tórax). Os autores disponibilizam o código.
- Ideia: aplicar a mesma abordagem few-shot em **histologia H&E**, usando a base
  **BreakHis**, que tem duas propriedades úteis:
  1. as imagens podem ser associadas ao **paciente** (o ID está no nome do arquivo);
  2. as imagens estão em **4 magnificações** (40×, 100×, 200×, 400×).
- Avaliar **por imagem** e depois **por paciente** e verificar se há **vazamento de
  dados**: "quando a gente avalia por imagem, um desempenho; quando vai para
  paciente, cai".
- Avaliar também a **influência da magnificação**.
- **Um modelo só**: "Só primeiro só um. Vamos focar."
- Manter a abordagem do artigo, sem investigar outras técnicas.
- Primeiro passo pedido: **ler o artigo, entender e rodar a implementação original**,
  depois passar para o domínio de histologia.
- Escrita: a metodologia é escrita quando a implementação estabilizar, depois os
  resultados; a introdução e a conclusão são ajustadas no final; a fundamentação
  teórica é escrita em paralelo, enquanto os experimentos rodam.
- Uso de IA: permitido e esperado, **com consciência e contraprova** (validar o
  que a IA produz). O código vai para um repositório Git disponível à banca.

### 2.2 Artigo base

> Montalvo-Lezama, B.; Fuentes-Pineda, G. **MetaChest: generalized few-shot
> learning of pathologies from chest X-rays.** *Visual Computing for Industry,
> Biomedicine, and Art*, 9:4, 2026. https://doi.org/10.1186/s42492-026-00214-4

- Código do experimento: https://github.com/bereml/meta-cxr
- Código de geração da base: https://github.com/bereml/metachest

**O que fazem:** classificação multirrótulo de patologias em raio-X com poucos
exemplos por classe, na formulação *generalized few-shot* (episódios com classes
já vistas no treino e classes novas). Comparam o **BatchBased** (transfer learning
convencional e avaliação por episódios) com o **ProtoNet-ML**.

**Principais achados:** o BatchBased supera o ProtoNet-ML na maioria das
configurações; mais shots e mais classes por episódio melhoram o resultado; o
pré-treino no ImageNet ajuda; uma resolução maior ajuda pouco e custa caro; e
arquiteturas eficientes ficam próximas das grandes.

### 2.3 Lacuna identificada (justificativa do TCC)

Conferido no código oficial (`meta-cxr` e `metachest`):

1. A divisão em meta-treino, meta-validação e meta-teste é feita **pelos rótulos de
   cada imagem** (`_filter_mset` em `data.py`), sem considerar o paciente.
2. Várias imagens por paciente são mantidas (`UNIQUE_STUDIES = False` nos notebooks
   das quatro bases).
3. O CSV final não tem coluna de paciente (as colunas são `dataset, name, age, sex, view`).
4. Dentro de cada episódio, o treino (support) e o teste (query) são sorteados da
   **mesma pilha de imagens** (`sample_at_least_k_shot`), sem restrição de paciente.

Ou seja, **o trabalho base não controla o paciente em nenhum nível**. Isso **não
prova** que os resultados deles estão inflados; mostra que a questão não foi
controlada. Escrever no texto sem tom de acusação.

**Por que pesa mais em histologia:** no BreakHis, as imagens de um paciente são
recortes da **mesma lâmina** (mesma coloração H&E, mesmo preparo, às vezes regiões
vizinhas). O modelo pode acertar por reconhecer **a lâmina ou a cor** em vez do
**tipo de tumor** (*shortcut learning*).

---

## 3. Glossário (com equivalentes em ML tradicional)

| Termo | Significado | Equivalente em ML tradicional |
|---|---|---|
| **Backbone** (f_φ) | Rede que transforma a imagem num vetor de características (*embedding*) | Extrator de features, só que aprendido |
| **Cabeça** (g_ϕ) | Uma camada linear `embedding → n classes`, seguida de sigmoide | Regressão logística *one-vs-rest* |
| **Pré-treino ImageNet** | O backbone começa com pesos aprendidos em fotos comuns | Partir de um extrator pronto |
| **Episódio** | Mini-tarefa de classificação sorteada (n classes, k exemplos por classe) | Um mini-dataset de treino e teste |
| **n-way** | Número de classes no episódio | Número de classes do problema |
| **k-shot** | Exemplos por classe para adaptar | Tamanho do treino por classe |
| **Support / D_trn** | Imagens de treino do episódio | Conjunto de treino |
| **Query / D_tst** | Imagens de teste do episódio | Conjunto de teste |
| **Seen / unseen** | Classes vistas no meta-treino / classes totalmente novas | — |
| **Meta-treino** | Etapa em que o backbone é treinado | Treino |
| **Meta-validação** | Episódios para early stopping e escolha do checkpoint | Validação |
| **Meta-teste** | Episódios para a nota final (usados só no fim) | Teste |
| **HM** | Média harmônica entre a AUC seen e a AUC unseen | — |
| **Vazamento por paciente** | Mesmo paciente dos dois lados de uma separação treino/teste | Usar `KFold` quando deveria ser `GroupKFold` |

**Dois níveis de divisão no few-shot:**

```
NÍVEL EXTERNO ("meta"): divide CLASSES e PACIENTES/IMAGENS em três mundos
  meta-treino            meta-validação             meta-teste
  (treina o backbone)    (episódios: early stop)    (episódios: nota final)

NÍVEL INTERNO: cada episódio tem seu próprio treino e teste
  episódio = n classes sorteadas
    ├── support: k imagens por classe   → adapta a cabeça
    └── query:   outras imagens         → mede AUC / acurácia
```

---

## 4. A base BreakHis

- 7.909 imagens RGB, PNG, 700×460, de **82 pacientes** (24 benignos e 58 malignos),
  em 4 magnificações.
- Imagens por magnificação (página oficial): 40× = 1.995; 100× = 2.081; 200× = 2.013;
  400× = 1.820.
- Download: Kaggle (BreaKHis_v1). Costuma vir com um `Folds.csv` com partições
  oficiais por paciente, útil como referência.
- **Nome do arquivo:** `SOB_M_DC-14-2523-40-010.png` → procedimento `SOB`, classe
  `M` (maligno; `B` = benigno), subtipo `DC`, paciente `14-2523`, magnificação `40`,
  sequência `010`.

### 4.1 Subtipos e pacientes (contagem real, Etapa 2)

Base bruta: 7.909 imagens, igual à página oficial (40× = 1.995, 100× = 2.081,
200× = 2.013, 400× = 1.820); todas as pastas têm as 4 magnificações. Depois da
limpeza (seção 4.2): **7.458 imagens de 70 pacientes** (40× = 1.875, 100× = 1.969,
200× = 1.897, 400× = 1.717).

- **Pasta:** o ID oficial (nome da pasta, ex. `14-22549AB` = `id_paciente` + `secao`);
  são os 82 "pacientes" do artigo da base.
- **Paciente (`id_paciente`):** o número de registro sem as letras de seção (ex. `14-22549`), a
  unidade adotada (D21).

| Subtipo | Sigla | Classe | Papel (D2) | Pastas (bruta) | Pacientes (bruta) | Imagens (bruta) | Pastas (usada) | **Pacientes (usada)** | Imagens (usada) |
|---|---|---|---|---|---|---|---|---|---|
| Carcinoma ductal | DC | M | seen | 38 | 38 | 3.451 | 37 | **37** | 3.328 |
| Fibroadenoma | F | B | seen | 10 | 7 | 1.014 | 7 | **5** | 688 |
| Carcinoma mucinoso | MC | M | seen | 9 | 7 | 792 | 9 | **7** | 792 |
| Adenoma tubular | TA | B | seen | 7 | 6 | 569 | 7 | **6** | 569 |
| Adenose | A | B | unseen mval | 4 | 2 | 444 | 4 | **2** | 444 |
| Carcinoma lobular | LC | M | unseen mval | 5 | 4 | 626 | 5 | **4** | 624 |
| Tumor filoide | PT | B | unseen mtst | 3 | 3 | 453 | 3 | **3** | 453 |
| Carcinoma papilífero | PC | M | unseen mtst | 6 | 6 | 560 | 6 | **6** | 560 |

Pipeline de dados em três passos, cada um sobre a saída do anterior:

1. `scripts/build_csv.py` → `data/breakhis_raw.csv` (7.909 imagens, sem tratamento):
   `path`, `malignidade` (benign/malignant), `id_subtipo`, `des_subtipo`, `id_paciente`,
   `secao`, `magnificacao`, `seq`, `rgb_hash` (MD5 dos pixels RGB).
2. `notebooks/01_eda.ipynb`: exploração da base bruta, onde as anomalias aparecem.
3. `notebooks/02_clear_csv.ipynb`: aplica D19 a D22 → `data/breakhis.csv` (7.458
   imagens, mesmas colunas). É o CSV usado daqui em diante.

### 4.2 Anomalias de identificação de paciente e decisões

**A. Imagens repetidas.** Comparando os arquivos byte a byte não aparece nenhuma
repetição (os PNGs foram regravados); comparando o **conteúdo em pixels** (hash MD5 da
matriz RGB), aparecem 125 grupos:

- **Nenhuma imagem se repete entre benigno e maligno.** A regra, caso aparecesse,
  seria excluir a imagem das duas classes (regra mantida no `02_clear_csv.ipynb`).
- **14-13412:** as 123 imagens da pasta DC são idênticas às 123 da pasta LC (às vezes
  com outro número de sequência). Mesmos dados, dois rótulos malignos; uma hipótese
  plausível é um carcinoma misto (ductal e lobular) catalogado duas vezes.
  **D19:** as imagens ficam só em LC. Motivo: LC é a classe unseen da meta-validação,
  com poucos pacientes (4), enquanto DC tem 38 e perder 1 quase não pesa.
- **LC 14-12204:** 2 pares repetidos a 40× (seq 16 = 34 e 17 = 35). **D20:** fica a
  primeira imagem de cada par (menor `seq`). Sem isso, a mesma imagem poderia cair no
  support e no query de um episódio.

**B. Pastas com o mesmo número de registro → D21: paciente = registro.**

O README da base define o ID como `ano-número` seguido de **letras de seção**
(`<SLIDE_ID>::=<NUMBER><SECTION>`), sem explicar as letras. Nove números se repartem
em várias pastas: 14134, 15570, 16184, 18842, 19979, 22549 e 23060 (mesmo subtipo);
**21998** (F, F e **PT**) e **29960** (**A** e **F**), com subtipos diferentes.

Justificativa da D21 (para a metodologia e para a banca):

1. **Como funciona um laboratório de patologia** (convenção usual, não documentada
   pela base): o número (`14-22549`) é o **registro do exame**, ou seja, um pedido,
   uma peça cirúrgica, um paciente. Na macroscopia a peça é cortada em fragmentos
   (**blocos**, identificados por letras), e cada lâmina leva um corte de ~3 µm de um
   ou mais blocos. Pastas com o mesmo número seriam, então, fragmentos diferentes da
   **mesma peça do mesmo paciente**.
2. **Direção segura:** tratar como pacientes distintas duas pastas que são da mesma
   pessoa deixaria **vazamento escondido dentro do P-PAC**, o protocolo que deveria
   ser o "sem vazamento". Isso subestimaria o Δ que o TCC mede. Agrupar a mais custa
   alguns pacientes; separar a menos invalida a comparação. É o mesmo raciocínio de
   escolher a coluna de grupo no `GroupKFold`.
3. **Indício nos dados:** a cor média H&E das pastas do mesmo registro é mais
   parecida do que a de pastas de registros diferentes (distância mediana 18,4 contra
   35,1). Alguns pares ficam quase idênticos (22549AB × CD no percentil 0,1 de todos
   os pares; 21998CD × EF no 0,8). Outros não (16184 × 16184CD no percentil 80), o
   que **não contradiz** a D21: blocos de regiões distintas da peça podem ter tecido
   e coloração diferentes.
4. **Limitações a declarar:** a interpretação das letras é uma inferência, não um
   fato documentado pela base; e registros diferentes **não** garantem pacientes
   diferentes (a mesma pessoa pode ter dois exames), o que a base não permite
   verificar. A literatura costuma usar a pasta (82 pacientes).
5. **Reversível:** se o orientador preferir a pasta, basta trocar a coluna de
   agrupamento (`id_paciente` → `id_paciente` + `secao`) nas divisões. Também é
   possível rodar as duas como análise de sensibilidade.

**D22: pacientes com dois subtipos.** Com a D21, 14-21998 (fibroadenoma e tumor
filoide) e 14-29960 (adenose e fibroadenoma) passam a ser um paciente com lesões de
dois subtipos, o que é biologicamente possível. Como F é seen (vai para mtrn, mval e
mtst) e PT/A são unseen (etapa fixa), o mesmo paciente acabaria em duas etapas
(vazamento V2). **Decisão:** descartar as imagens de F desses dois pacientes (326
imagens). PT e A, as classes mais escassas, ficam intactas. Alternativa descartada:
forçar o F desses pacientes na etapa da classe unseen, o que complicaria as divisões e
tiraria essas imagens do sorteio.

**Resumo da limpeza** (`02_clear_csv.ipynb`): D22 = 326 imagens, D19 = 123,
D20 = 2; total 451. O notebook confere com `assert` que, no CSV limpo, não
sobra hash repetido e cada paciente tem um só subtipo.

### 4.3 Viabilidade dos k-shots no P-PAC

Simulação do sorteio da seção 6.3 com k_tst = 15, sobre as imagens usadas (classes
seen com 20% dos pacientes por etapa, mínimo de 2):

- Classes unseen (exceto A) e DC: sempre viáveis com k ≤ 20.
- MC (até 15%) e TA (17% a 35%) falham **mesmo com k = 1**. O gargalo é o query:
  às vezes o paciente restante tem menos de 15 imagens.
- k = 20 é crítico para TA (10% a 25% de sucesso), MC (48% a 72%) e F (80%).
- Adenose (2 pacientes): a 100× só 51% dos sorteios funcionam com k ≤ 10, e nenhum
  com k = 20.
- Um paciente de DC tem só 1 imagem a 40×.
- **Com k_tst = 10** (simulação sobre as divisões reais, 5 sementes × 4
  magnificações × mval/mtst, testando **todas** as ordens de pacientes): k = 1, 5 e
  10 são viáveis em 100% das 40 combinações; k = 15 só em 18 de 40 e é impossível
  em uma (TA, mtst, 100×, semente 2026). Daí as decisões D27 e D28 (seção 6.4).
- **Consequência da D22 para a Etapa 3:** o F ficou com 5 pacientes. Com no mínimo 2
  na meta-validação e 2 no meta-teste, sobra **1 paciente de F no meta-treino** (ver P2).

---

## 5. Decisões tomadas

| # | Tema | Decisão | Status |
|---|---|---|---|
| D1 | Base | BreakHis, 8 subtipos, 4 magnificações tratadas **separadamente** | [DECIDIDO] |
| D2 | Papéis das classes | Seen = **DC, F, MC, TA**; unseen na meta-validação = **A, LC**; unseen no meta-teste = **PT, PC** | [DECIDIDO] |
| D3 | Eixo principal | Separação **por imagem × por paciente**, aplicada nos **dois níveis** (entre etapas e dentro do episódio) | [DECIDIDO] |
| D4 | Métrica principal | AUC-ROC seen, AUC-ROC unseen e HM (como no artigo), com IC de 95% | [DECIDIDO] |
| D5 | Métrica complementar | **Acurácia balanceada por paciente** (seção 8.2) | [DECIDIDO] |
| D6 | Backbone | **MobileNetV3-Small-0.75**, pré-treino ImageNet-1K (`timm`: `tf_mobilenetv3_small_075.in1k`, o mesmo do `meta-cxr`) | [DECIDIDO] |
| D7 | Método | **BatchBased** (padrão e melhor resultado do artigo). ProtoNet-ML fora do escopo (trabalho futuro) | [DECIDIDO] |
| D8 | Saída da cabeça | Manter a **sigmoide por classe** (fiel ao artigo). Classe prevista = *argmax* das probabilidades (para a acurácia) | [DECIDIDO] |
| D9 | Código | Novo repositório próprio, **copiando o que serve** do `meta-cxr` e simplificando o resto (ver `REUSO_CODIGO.md`). Sem fork formal | [DECIDIDO] |
| D10 | lr da cabeça na adaptação | **0,005** (valor do código e da Tabela 9; a Tabela 3 diz 0,05, provável erro de digitação) | [DECIDIDO] |
| D11 | Formato dos episódios | 3-way com 1 unseen (padrão do artigo): 2 seen + 1 unseen | [PADRÃO] |
| D12 | Episódios | 100 na meta-validação (por época) e 1.000 no meta-teste | [PADRÃO] |
| D13 | Sementes | 3 a 5 **divisões de pacientes** diferentes (a semente controla split, pré-treino e episódios) | [PADRÃO] |
| D14 | Protocolo P-MIX | Incluir (custo baixo, decompõe o vazamento) | [PADRÃO] |
| D15 | Resolução | **Resolução nativa 700×456**, sem redimensionar (D35); normalização do `pretrained_cfg` do modelo (média 0,5 e desvio 0,5 no `tf_mobilenetv3_small_075`) | [PADRÃO] |
| D16 | Data augmentation | **Desligada** (padrão do código original) | [PADRÃO] |
| D17 | Magnificações misturadas | Só como extra opcional (expõe o vazamento V3) | [PADRÃO] |
| D18 | Hardware e precisão | Sem GPU NVIDIA: macOS (M1, MPS) e Windows (AMD RX 7600, ROCm ou DirectML). Dispositivo escolhido em `src/device.py` (cuda/ROCm → mps → directml → cpu), precisão **fp32** (bf16 só como opção em CUDA) | [DECIDIDO] |
| D19 | Paciente 14-13412 (mesmas imagens em DC e LC) | Imagens mantidas **só em LC** (seção 4.2-A) | [DECIDIDO] |
| D20 | Imagens repetidas | Deduplicar, mantendo a de menor `seq`; repetidas entre benigno e maligno sairiam das duas classes (nenhuma encontrada) | [DECIDIDO] |
| D21 | Unidade de "paciente" | **Número de registro sem as letras de seção** (pastas com o mesmo número = mesmo paciente). Justificativa na seção 4.2-B | [DECIDIDO] |
| D22 | Pacientes com dois subtipos, 14-21998 (F/PT) e 14-29960 (A/F) | Descartar as imagens de **F** desses pacientes (seção 4.2-B) | [DECIDIDO] |
| D23 | Pacientes por etapa nas classes seen (P2) | mval = mtst = máx(2, arred(0,2·n)) pacientes; resto no mtrn. **F fica 1/2/2**: o mínimo de 2 é obrigatório para o P-PAC separar support e query; 1 paciente de F no mtrn é limitação declarada | [DECIDIDO] |
| D24 | Divisão entre magnificações | O sorteio de pacientes depende só da semente: **o mesmo nas 4 magnificações**, para não confundir o efeito da magnificação com o da divisão | [DECIDIDO] |
| D25 | Arquivo das divisões | `data/splits/seed{S}.csv`, uma linha por imagem: `path, id_paciente, id_subtipo, magnificacao, etapa_pac, etapa_img` | [DECIDIDO] |
| D26 | Nome do grupo benigno/maligno | Coluna `malignidade`; no few-shot, "classe" é sempre o **subtipo** | [DECIDIDO] |
| D27 | k-shots avaliados (P3) | **{1, 5, 10}**. k = 15 e k = 20 excluídos por inviabilidade no P-PAC (justificativa na seção 6.4) | [DECIDIDO] |
| D28 | Tamanho do query (P4) | **k_tst = 10** imagens por classe (artigo: 30) | [DECIDIDO] |
| D29 | Episódios pareados entre protocolos | O episódio nº i usa as **mesmas classes** em P-IMG, P-MIX e P-PAC; muda só a origem das imagens | [DECIDIDO] |
| D30 | Episódios da meta-validação | As **mesmas 100 tarefas** em todas as épocas (o original sorteia novas a cada época) | [DECIDIDO] |
| D31 | Sementes dos episódios | Gerados na hora, com gerador derivado da semente da rodada: `default_rng([semente, magnificação, etapa, k, episódio])`; sem o protocolo (pareamento) e sem a época (mval fixa). IDs dos pacientes gravados nos resultados | [DECIDIDO] |
| D32 | k da meta-validação | **k = 5, k_tst = 10** nos episódios do early stopping (o original usa 30/30) | [DECIDIDO] |
| D33 | Formato dos resultados | CSVs em `results/` (sem MLflow): por época no treino, por episódio no mtst e auditoria de pacientes (seção 8.4) | [PADRÃO] |
| D34 | Episódios da mval de cada backbone | Backbone da divisão por paciente: early stopping com episódios **por paciente**; da divisão por imagem: **por imagem** (como o original). Cada protocolo é seguido de ponta a ponta. Ver seção 6.4 | [DECIDIDO] |
| D35 | Formato da entrada | **Resolução nativa 700×456** (largura × altura), sem redimensionar; as imagens de 460 perdem 2 linhas em cima e 2 embaixo para igualar às 74 de 456. Sem recorte lateral, sem bordas, sem deformação. Custo e riscos na seção 9.6 | [DECIDIDO] |
| D36 | Cache de imagens | Gerado **uma vez** em disco: `data/cache/{mag}.npz` (uint8, com a lista de caminhos), por `src/cache.py` | [DECIDIDO] |
| D37 | Lote da adaptação | Fiel ao original: `max(1, int(0,5 · n_support))` por passo. Com k = 1 (3 imagens no support), cada passo usa 1 imagem | [DECIDIDO] |
| D38 | Época 0 | Avaliar a mval com o backbone só do ImageNet, antes do meta-treino: linha de base de quanto o meta-treino ajuda | [DECIDIDO] |
| D39 | Desequilíbrio no meta-treino | Mantido (o DC é ~60% do mtrn), sem pesos por classe, como no original (método fixo) | [DECIDIDO] |
| D40 | Arquivos da Etapa 5 | `cache.py` (gera e carrega o cache), `model.py` (backbone e embeddings), `adapt.py` (um episódio: cabeça nova + métricas), `train.py` (meta-treino + mval + early stopping; nunca lê o mtst), `evaluate.py` (nota final no mtst → CSVs) | [DECIDIDO] |
| D41 | Máquina de execução | Qualquer uma (`get_device`); o dispositivo vai para o `config.json`. Os dois backbones de uma rodada (semente, magnificação) são treinados na mesma máquina. Sem comparação de hardware | [DECIDIDO] |
| D42 | Limite de memória | `get_device(max_memory_usage=0.75, allow_swap=False)`: fração máxima da memória da GPU (MPS: do *working set* recomendado; CUDA/ROCm: da VRAM). Passar do limite gera erro de falta de memória em vez de deixar a máquina lenta. `allow_swap=True` remove o limite no MPS (o macOS pode usar o disco). DirectML e CPU: sem controle | [DECIDIDO] |
| D43 | Micro-lotes no meta-treino | Lote **efetivo** de 64 (método fixo), formado por micro-lotes com acumulação de gradientes. **Configurável:** parâmetro `micro_batch_size` do `train.py` (padrão 16; precisa dividir 64). `micro_batch_size=64` = lote único, sem acumulação. Ressalva da BatchNorm na seção 9.6 | [DECIDIDO] |
| D44 | Onde roda a adaptação | **CPU**: a cabeça é pequena demais para compensar a GPU, e a CPU é determinística. Os embeddings (pesados) são calculados na GPU. ~20 ms por episódio | [DECIDIDO] |
| D45 | Aleatoriedade da cabeça | Inicialização e sorteios de 50% por passo vêm de um gerador derivado da chave do episódio (D31), sem o protocolo: a cabeça começa com os **mesmos pesos** nos três protocolos (o pareamento da D29 vale também para a inicialização) | [DECIDIDO] |
| D46 | Acurácia por imagem | Só `acc_img`: com o query balanceado (10 por classe), a acurácia balanceada por imagem é igual à acurácia comum | [DECIDIDO] |
| D47 | Embeddings | Em memória durante o treino; os do mtst são gravados pelo `evaluate.py` em `results/embeddings/{mag}_{backbone}_{divisao}_seed{S}.npz` (~220 MB no total, com os do ImageNet), para a contraprova (Etapa 6) e a auditoria | [DECIDIDO] |
| D48 | Época 0 no early stopping | **Concorre**: se nenhuma época do meta-treino superar a HM da mval do backbone só ImageNet, o `best_model.pt` é o próprio ImageNet (e isso é um achado) | [DECIDIDO] |
| D49 | Lote final do meta-treino | O lote incompleto do fim de cada época é usado (como o `DataLoader` padrão); a perda é ponderada pelo tamanho real | [DECIDIDO] |
| D50 | Treino interrompido | Sem retomada: a rodada roda de novo. O `config.json` só ganha o campo `fim` ao terminar, e o `run_grid` pula só rodadas completas | [DECIDIDO] |
| D51 | Registro da mval por época | Todas as médias das 100 tarefas: HM, AUC seen/unseen, acc_img, acc_bal_pac, frac_V1/V2 (mostra como o vazamento guia o early stopping no P-IMG) | [DECIDIDO] |
| D52 | Testes lentos | Testes que treinam a rede levam `@pytest.mark.slow` e ficam fora do `pytest` padrão; rodam com `pytest -m slow` (ou `-m ""` para todos) antes de commitar mudanças em `cache`, `model`, `adapt` ou `train` e antes de toda PR | [DECIDIDO] |
| D53 | Ambiente do Windows (RX 7600) | PyTorch-ROCm do índice estável da AMD: `torch 2.13.0+rocm10.1.0` + `torchvision 0.28.0` (`device-gfx1102`), rodado com `UV_NO_SYNC=1`. O MIOpen dessa versão falha ao compilar a BatchNorm, e o `get_device()` o desliga quando o PyTorch é ROCm. Lote 16 a 700×456: 0,21 s por passo (M1: 0,56 s); lote 64 não cabe no limite de 75% | [DECIDIDO] |
| D54 | Backbone do ImageNet no mtst | Os três protocolos são avaliados também com o backbone só do ImageNet (coluna `backbone` = `meta` ou `imagenet`), com os mesmos episódios e a mesma cabeça inicial. Sem meta-treino não há V2 (`frac_V2` = 0), então a diferença entre protocolos com esse backbone isola o efeito do V1 | [DECIDIDO] |
| D55 | Auditoria dos episódios | Pacientes de cada papel (support/query) em cada episódio, por classe e com o número de imagens, em `.csv.gz` (uma vez por episódio: não depende do backbone) | [DECIDIDO] |
| D56 | Configuração da avaliação | `config.json` próprio da avaliação (commit, dispositivo, constantes da adaptação e melhor época/HM de cada treino) | [DECIDIDO] |
| D57 | Execuções pesadas | Testes lentos, treinos e avaliações rodam no Windows (RX 7600) via `ssh win`, a partir da sessão do Mac, **só com autorização explícita do aluno** a cada vez | [DECIDIDO] |

---

## 6. Fontes de vazamento e protocolos

### 6.1 Fontes de vazamento

| Código | Onde | Exemplo |
|---|---|---|
| **V1** | Dentro do episódio: support × query | A imagem 001 do paciente X está no support e a 002 está no query |
| **V2** | Entre etapas: meta-treino × meta-val/teste | O backbone treinou com o paciente X (classe seen) e o paciente X reaparece nos episódios de teste |
| **V3** | Entre magnificações | Mesma região do paciente X a 40× no treino e a 100× no teste (só se as magnificações forem misturadas) |

Cuidado extra (não é de paciente): **nunca** escolher hiperparâmetros ou fazer early
stopping com o meta-teste.

### 6.2 Protocolos comparados

| Protocolo | Entre etapas (V2) | Dentro do episódio (V1) | Papel |
|---|---|---|---|
| **P-IMG** | Por imagem | Por imagem | Como o artigo base faz (com vazamento) |
| **P-MIX** | Por paciente | Por imagem | Isola o efeito de V1 |
| **P-PAC** | Por paciente | Por paciente | Cenário realista, sem vazamento |

**Quantificação:**

- Vazamento total: `Δ_total = M(P-IMG) − M(P-PAC)`
- Parcela de V2: `Δ_V2 ≈ M(P-IMG) − M(P-MIX)`
- Parcela de V1: `Δ_V1 ≈ M(P-MIX) − M(P-PAC)`
- Também reportar o Δ relativo: `Δ / M(P-PAC) × 100%`.

> **Correção após o piloto (P7, pendente):** a decomposição `Δ_V2 ≈ P-IMG − P-MIX` e
> `Δ_V1 ≈ P-MIX − P-PAC` supõe que P-IMG e P-MIX tenham o mesmo V1, o que **não**
> acontece: a divisão por paciente concentra poucos pacientes por classe na etapa, e o
> P-MIX tem **mais** V1 que o P-IMG (frac_V1 de 0,42 a 0,93 contra 0,22 a 0,76 no
> piloto). Com o backbone só ImageNet, o "Δ_V2" chega a sair negativo (−6 a −9). Proposta
> para a Etapa 9: medir o **V2** pela comparação entre backbones no mesmo protocolo
> (meta × ImageNet, D54) e o **V1** controlando o frac_V1 episódio a episódio (por
> exemplo, regressão da HM sobre frac_V1 e protocolo). O Δ_total continua válido.

### 6.3 Como cada protocolo monta os dados

**Classes seen (DC, F, MC, TA), que aparecem nas três etapas:**

- **P-PAC / P-MIX:** para cada classe seen, sortear os **pacientes** em
  meta-treino, meta-validação e meta-teste (alvo ~60/20/20), com **no mínimo 2
  pacientes** na validação e no teste. Com os pacientes da seção 4.1:
  DC ≈ 23/7/7; MC ≈ 3/2/2; TA ≈ 2/2/2; **F = 1/2/2** (ver P2). Todas as imagens de um paciente vão para a mesma etapa.
- **P-IMG:** juntar as imagens da classe e sortear **imagens** para cada etapa,
  com **o mesmo número de imagens por etapa e por classe** que o P-PAC gerou na
  mesma semente. Assim, só a regra de agrupamento muda.

**Classes unseen (A, LC na validação; PT, PC no teste):** aparecem numa única etapa,
então não há divisão entre etapas. A diferença entre os protocolos está só dentro
do episódio (V1).

**Dentro do episódio:**

- **Por imagem (P-IMG, P-MIX):** sortear k imagens para o support e k_tst para o
  query, sem olhar o paciente (como o código original).
- **Por paciente (P-PAC):** para cada classe do episódio, embaralhar os pacientes
  daquela classe na etapa; alocar pacientes ao support até ter k imagens; sortear
  as k imagens do support entre eles; sortear as k_tst do query **apenas entre os
  pacientes restantes**. Com os k da D27 isso é sempre possível; se algum dia não
  for, é erro (o código para com `assert`), nunca se viola a regra.

### 6.4 Episódios: regras e justificativas (texto-base para a metodologia)

**Formato.** Cada episódio tem 3 classes (2 seen e 1 unseen da etapa, D11), k imagens
por classe no support e k_tst = 10 no query (D28). O support treina uma cabeça linear
nova sobre o backbone congelado; o query mede AUC seen, AUC unseen e HM.

**k-shots avaliados: {1, 5, 10} (D27).** No protocolo por paciente, cada classe do
episódio precisa de pacientes que somem k imagens no support e de **outros** pacientes
que somem k_tst = 10 no query. As classes TA, MC e adenose têm apenas 2 pacientes em
mval ou mtst (D23), com 12 a 20 imagens por paciente em algumas magnificações. A
viabilidade foi verificada sobre as divisões reais (5 sementes × 4 magnificações ×
mval/mtst = 40 combinações), testando todas as ordens possíveis de sorteio dos pacientes:

| k | Combinações em que todas as classes são sempre viáveis |
|---|---|
| 1, 5, 10 | 40 de 40 |
| 15 | 18 de 40 (e uma impossível: TA, mtst, 100×, semente 2026) |
| 20 | ainda menos (já crítico com k_tst = 15, seção 4.3) |

Avaliar k = 15 exigiria uma de duas saídas, ambas ruins para a comparação:
(a) **ressortear** os episódios inviáveis, o que, numa classe com 2 pacientes, força
sempre o paciente maior no support e o menor no query, de modo que a AUC dessa classe
passa a medir um único paciente, além de a classe aparecer menos vezes; ou
(b) **descartar** combinações, deixando a grade desigual entre sementes e magnificações.
Por isso k foi limitado a {1, 5, 10}, valores com os quais o sorteio por paciente é
sempre possível, sem ressorteio e sem descarte. A exclusão é também um **achado**: com
poucos pacientes por classe, o número de exemplos por classe que se pode usar sem
vazamento é limitado pela própria base, algo que a avaliação por imagem esconde.

**Episódios pareados (D29).** O episódio nº i sorteia as mesmas classes nos três
protocolos; muda apenas de onde vêm as imagens (divisão por imagem ou por paciente,
sorteio dentro do episódio por imagem ou por paciente). Assim, a diferença entre
protocolos não se mistura com a variação de "quais classes caíram" (combinações como
MC × TA tendem a ser mais difíceis que DC × F), e a diferença pode ser testada por
episódio (teste pareado, ex.: Wilcoxon). É o equivalente a comparar dois modelos nos
mesmos folds de uma validação cruzada.

**Meta-validação com tarefas fixas (D30).** O early stopping compara a HM média da
meta-validação entre épocas. Com tarefas sorteadas de novo a cada época (como no
código original), parte dessa diferença viria da sorte do sorteio, e não da mudança
do backbone; com classes de 2 pacientes por etapa, esse ruído é grande e pode
antecipar a parada ou escolher uma época "de sorte". Por isso as mesmas 100 tarefas
são usadas em todas as épocas. O custo é que o checkpoint fica escolhido para essas
100 tarefas; o meta-teste, separado, não é afetado.

**k da meta-validação: 5 (D32).** O backbone é treinado uma vez e avaliado em todos
os k (seção 7), então o early stopping usa um único k. O original usa 30, inviável no
BreakHis. k = 5 é o valor central da grade e é sempre viável:

| k na mval | Efeito |
|---|---|
| 1 | Cada cabeça aprende com 1 imagem por classe: a HM de cada episódio varia muito, e a média de 100 episódios oscila entre épocas por acaso. O early stopping fica errático (pode parar cedo ou escolher uma época de sorte), e o checkpoint é escolhido para o regime mais difícil |
| **5** | Ruído intermediário; checkpoint escolhido para o meio da grade {1, 5, 10} |
| 10 | HM mais estável e early stopping mais confiável, mas o checkpoint é escolhido para o regime mais fácil e pode não ser o melhor para k = 1 |

O custo computacional é praticamente o mesmo nos três casos (a cabeça treina sobre
embeddings pré-calculados). O impacto se limita à **escolha da época**: o treino do
backbone em si não depende do k.

**Sementes (D31).** Cada rodada tem uma semente (a mesma da divisão, D13), e dela se
derivam geradores independentes por finalidade, por exemplo
`np.random.default_rng([semente, magnificação, etapa, k, episódio])`. Isso torna cada
episódio reprodutível e independente dos demais (mudar k ou gerar mais episódios não
altera os outros); como o protocolo não entra no gerador, as classes coincidem entre
protocolos (D29), e como a época não entra, a mval se repete (D30).

**Episódios da mval de cada backbone (D34).** O backbone da divisão por
paciente (usado por P-PAC e P-MIX) faz early stopping com episódios por paciente; o
da divisão por imagem (P-IMG), com episódios por imagem, como o artigo. Assim cada
protocolo é seguido de ponta a ponta, incluindo uma validação que, no P-IMG, também
tem vazamento e pode favorecer épocas mais tardias.

---

## 7. Desenho experimental

**Fatores (variáveis independentes):**

- Protocolo: P-IMG, P-MIX, P-PAC.
- Magnificação: 40×, 100×, 200×, 400× (cada uma é um experimento independente).
- k-shot: {1, 5, 10} (D27; k = 15 e 20 excluídos por inviabilidade, seção 6.4).

**Fixos:** BatchBased, MobileNetV3-Small-0.75, 3-way com 1 unseen, resolução nativa 700×456 (seção 9.6), sem
augmentation e hiperparâmetros da seção 9.

**Custo:** o pré-treino do BatchBased **não depende do k-shot**. Treina-se o
backbone uma vez por (magnificação, divisão de etapas, semente) e avaliam-se
todos os k-shots com ele. O P-MIX compartilha o backbone do P-PAC (mesma divisão
entre etapas). Portanto:

- pré-treinos = 4 magnificações × 2 divisões (por imagem, por paciente) × S sementes
  → com S = 5, são **40 pré-treinos** (cerca de 650 a 700 imagens de meta-treino cada a 40×: ~50% das ~1.350 imagens seen, porque o mínimo de 2 pacientes em mval/mtst pesa nas classes pequenas;
  devem levar minutos com a MobileNet);
- avaliações = 4 magnificações × 3 protocolos × 3 k-shots × S sementes × 1.000 episódios
  (com S = 5, 180.000 episódios, cada um uma cabeça linear sobre embeddings).

**Atalho de implementação (exato):** na adaptação, o backbone fica congelado e em
modo `eval`, e não há augmentation. Então o embedding de cada imagem é sempre o
mesmo, e dá para **pré-calcular os embeddings** de todas as imagens de
meta-validação e meta-teste uma vez por checkpoint. Cada episódio passa a ser só
treinar a cabeça linear sobre vetores em memória. Isso é ordens de grandeza mais
rápido e matematicamente equivalente ao código original (que passa as imagens pelo
backbone congelado a cada passo). Na meta-validação durante o treino, os embeddings
precisam ser recalculados a cada época, porque o backbone muda.

---

## 8. Métricas

### 8.1 Métricas do artigo (por imagem, por episódio)

- **AUC seen:** AUC-ROC de todos os rótulos das classes seen do episódio, tratados
  como uma única tarefa binária (média *micro*, como no código: `multilabel_auroc`,
  `average='micro'`).
- **AUC unseen:** a mesma coisa para as classes unseen.
- **HM** = 2 · seen · unseen / (seen + unseen).
- Nota final = média sobre os episódios, com **IC de 95% = 1,96 · desvio / √N**
  (`compute_mean_ci` do código original). Entre sementes: média das sementes e IC
  por bootstrap (`agregate_run_df_aux` do original) ou desvio padrão.

### 8.2 Acurácia balanceada por paciente (complementar)

Calculada em cada episódio, sobre as imagens do query:

1. Classe prevista de cada imagem = `argmax` das probabilidades.
2. Para cada **paciente** p no query: `acc_p` = fração das imagens dele classificadas corretamente.
3. Para cada **classe** c: `acc_c` = média de `acc_p` entre os pacientes da classe c.
4. Episódio: `acc_bal_pac` = média de `acc_c` entre as classes do episódio.
5. Nota final: média entre episódios (com IC).

Cada paciente pesa igual dentro da sua classe, e cada classe pesa igual no total.
Corrige o desequilíbrio de imagens por paciente **e** de pacientes por classe.
**Não remove vazamento** (isso é papel da separação).

Também registrar a acurácia **por imagem** (`acc_img`), para comparar a unidade de
contagem. Como o query tem exatamente k_tst = 10 imagens por classe, a acurácia
balanceada por imagem é igual à acurácia comum e não é registrada à parte (D46).

### 8.3 Exposição ao vazamento (descritiva)

Por episódio, registrar:

- `frac_V1` = fração das imagens do query cujo paciente também está no support;
- `frac_V2` = fração das imagens do query cujo paciente esteve no meta-treino.

No P-PAC, as duas devem ser **0** (teste automático). No P-IMG, mostram *quanto*
vazamento existiu, e permitem relacionar a AUC do episódio com a exposição.

### 8.4 Formato dos resultados

Arquivos em `results/` (D33), sem MLflow: são poucos treinos e o pandas basta.

| Nível | Arquivo | Uma linha por |
|---|---|---|
| Treino | `results/train/{mag}_{divisao}_seed{S}/epocas.csv` (+ `best_model.pt` e `config.json` com constantes, semente, dispositivo, versões e commit) | época: perda do mtrn, HM/AUCs da mval, se foi a melhor, tempo |
| Avaliação | `results/mtst/{mag}_seed{S}.csv` | episódio × protocolo × k × backbone (colunas abaixo, mais `backbone`) |
| Auditoria | `results/mtst/{mag}_seed{S}_pacientes.csv.gz` | paciente × papel (support/query) em cada episódio (D55) |
| Configuração da avaliação | `results/mtst/{mag}_seed{S}_config.json` | commit, dispositivo, constantes e melhor época/HM de cada backbone; `fim` marca a rodada completa (D56) |
| Embeddings | `results/embeddings/{mag}_{backbone}_{divisao}_seed{S}.npz` | imagem do mtst (vetor de 1.024 + caminho), para a contraprova e a auditoria (D47) |

A mval é guardada só agregada por época (diagnóstico do treino); o mtst, por episódio,
porque dele saem a nota final, os ICs e o teste pareado. Colunas da avaliação:
`magnificacao, protocolo, semente, k_shot, episodio, classes_seen, classes_unseen,
auc_seen, auc_unseen, hm, acc_img, acc_bal_pac, frac_V1, frac_V2`,
mais a AUC por classe. Guardar também os IDs dos pacientes do support e do query
de cada episódio (arquivo à parte), para auditoria.

---

## 9. Modelo: BatchBased + MobileNetV3-Small-0.75

### 9.1 Estrutura

```
imagem 384×384×3 → backbone f_φ (timm tf_mobilenetv3_small_075.in1k, num_classes=0)
                 → embedding de 1024 dimensões (`backbone.head_hidden_size`; cuidado:
                   no timm, `num_features` = 432 é a saída antes da camada conv_head)
                 → cabeça g_ϕ = nn.Linear(D, n_classes)
                 → sigmoide → p(y[c]=1 | x) = σ(g_ϕ(f_φ(x)))
perda: BCEWithLogitsLoss (rótulos one-hot)
```

Camada linear + sigmoide + log-loss é matematicamente **n regressões logísticas
one-vs-rest** sobre o embedding.

### 9.2 Fase 1: meta-treino (uma vez por backbone)

- Cabeça com 4 saídas (as classes seen DC, F, MC, TA).
- **Backbone e cabeça treinados juntos**, em lotes convencionais das imagens de
  meta-treino.
- Ao fim de cada época: avaliar as **mesmas** 100 tarefas de meta-validação (D30),
  com k = 5 e k_tst = 10 (D32), e calcular a HM média.
- Early stopping pela HM da meta-validação (paciência de 10) e guardar o melhor
  checkpoint.
- A cabeça de 4 saídas é **descartada**; segue só o backbone.

### 9.3 Fase 2: adaptação (a cada episódio de validação/teste)

1. Cópia do modelo (ou os embeddings pré-calculados, ver seção 7).
2. Cabeça **nova**, `nn.Linear(D, 3)`, com pesos aleatórios.
3. Backbone **congelado** (`requires_grad=False` e `eval()`).
4. 100 passos de AdamW (lr 0,005); em cada passo, um sorteio de 50% das imagens do support.
5. Prever o query → sigmoide → métricas.
6. Descartar a cabeça.

### 9.4 Hiperparâmetros

| Parâmetro | Valor | Origem |
|---|---|---|
| Resolução | 700×456, nativa (artigo: 384×384) | D35 |
| Normalização | `backbone.pretrained_cfg` (média 0,5, desvio 0,5 nos pesos `tf_`; não os valores clássicos do ImageNet) | `base.py` |
| Augmentation | desligada | `args.py` (`data_aug=False`) |
| Lote do meta-treino | 64 efetivo; micro-lotes de `micro_batch_size` (padrão 16) com acumulação de gradientes | Tabela 3 / D43 |
| Otimizador do meta-treino | AdamW, lr 1e-4 | Tabela 3 / `batchbased.py` |
| Épocas máximas | 150 | Tabela 3 |
| Early stopping | HM da meta-validação, paciência 10 | Tabela 3 |
| Episódios de meta-validação | 100 por época, sempre as mesmas tarefas | Tabela 3 / D30 |
| k / k_tst da meta-validação | 5 / 10 (artigo: 30 / 30) | D32 |
| k / k_tst do meta-teste | {1, 5, 10} / 10 (artigo: 30) | D27, D28 |
| Episódios de meta-teste | 1.000 (artigo: 10.000) | D12 |
| Adaptação: passos / fração / lr | 100 / 0,5 / **0,005** | `batchbased.py` (D10) |
| Perda | `BCEWithLogitsLoss` | `base.py` |
| Precisão | **fp32** (o artigo usa bf16; aqui bf16 fica só como opção em CUDA, ver D18) | Tabela 3 / D18 |
| Weight decay | padrão do AdamW no PyTorch | não informado no artigo |

### 9.5 Justificativa (para a metodologia)

1. **Fidelidade:** é a configuração padrão do trabalho base, e o BatchBased superou
   o ProtoNet-ML na maioria das configurações.
2. **Controle experimental:** fixar o modelo isola o efeito da separação dos dados
   e da magnificação.
3. **Custo:** é a arquitetura mais leve avaliada no artigo (~1,0 M de parâmetros,
   0,11 GMACs; Tabela 8), o que viabiliza dezenas de treinos com várias sementes.
   O ConvNeXt-Tiny ganhou só cerca de 2 pontos de HM a um custo muito maior.
4. **Literatura:** Chen et al. (2019, *A Closer Look at Few-shot Classification*)
   mostraram que um baseline com pré-treino convencional e classificador linear por
   episódio é competitivo com o meta-aprendizado.

Observações para o texto: citar a MobileNetV3 por **Howard et al., 2019
(*Searching for MobileNetV3*)** (o artigo base cita a MobileNet v1); e informar o
lr da cabeça realmente usado (0,005), mencionando a divergência com a Tabela 3 numa
frase neutra.

---

### 9.6 Formato da entrada: resolução nativa (700×456) e o que isso pode prejudicar

As imagens do BreakHis têm 700×460, exceto 74 de 700×456 (paciente 14-12465, PC). O
artigo usa 384×384, mas a MetaChest já era quadrada. As alternativas avaliadas:

| Opção | Escala do tecido | Perda | Deformação | Pixels (vs. 384×384) |
|---|---|---|---|---|
| Esticar para 384×384 | 55% × 83% | nenhuma | sim (núcleos achatados) | 1× |
| Recorte central 460×460 → 384×384 | 83% | 34% da área (laterais) | não | 1× |
| Bordas até 700×700 → 384×384 | 55% | nenhuma (34% da entrada é borda) | não | 1× |
| Proporção mantida → 584×384 | 83% | nenhuma | não | 1,5× |
| **Resolução nativa 700×456** (escolhida) | **100%** | 4 linhas de pixel nas imagens de 460 | não | **2,2×** |

A MobileNet aceita entradas não quadradas porque termina num *global average pooling*,
que faz a média das features sobre qualquer área. Assim a imagem entra inteira, sem
interpolação, com o tecido na escala original da captura. Em histologia isso importa:
forma e tamanho de núcleos e arranjo das células são informação diagnóstica. Para todas
terem o mesmo tamanho sem interpolar nenhum pixel, as imagens de 460 perdem 2 linhas
em cima e 2 embaixo (`src/cache.py`).

**Custo medido no M1 (MPS), lote de 16:** 0,56 s por passo de treino (0,28 s em
384×384) e 0,22 s para os embeddings de 16 imagens, com 3,5 GB de memória da GPU.
**Lote de 64 não cabe** na memória da GPU do M1: o macOS passa a usar o disco (swap) e o
passo sobe para 78 s. Daí a D42 (limite de memória) e a D43 (micro-lotes).

**O que pode prejudicar (a declarar no texto):**
1. **Desvio do artigo:** a entrada deixa de ser 384×384, tanto no formato quanto na
   escala; os números não são diretamente comparáveis aos da MetaChest (que, de todo
   modo, é outra base). O próprio artigo mostra que resoluções maiores ajudam pouco e
   custam muito; aqui a escolha é pela fidelidade ao tecido, não pelo ganho esperado.
2. **Custo:** 2,2× mais pixels; tempo por época cerca de 2× e cache de ~1,9 GB por
   magnificação. Estimativa: ~45 s a 1 min por época, ~1 a 1,5 dia para os 40 treinos.
3. **Escala diferente do pré-treino:** os pesos do ImageNet foram aprendidos em fotos de
   224×224, com objetos numa escala muito maior que a de células a 700×456. As
   convoluções funcionam em qualquer tamanho, mas as features pré-treinadas podem
   aproveitar menos essa escala, e a estatística do pooling final muda com a área. O
   efeito é igual para todos os protocolos e não afeta as comparações.
4. **Periferia do campo:** bordas da imagem, às vezes menos focadas ou com vinheta da
   ótica do microscópio, entram na média final; num recorte central seriam descartadas.
   Também é igual para todos os protocolos.
5. **BatchNorm com micro-lotes (D43):** se o lote de 64 for formado por micro-lotes, a
   BatchNorm da MobileNet normaliza pelas estatísticas de cada micro-lote, não das 64
   imagens; a diferença é pequena e diminui com micro-lotes maiores.

## 10. Pendências

| # | Pendência | Como resolver | Padrão sugerido |
|---|---|---|---|
| P1 | ~~Contagem real de pacientes por subtipo~~ | Resolvida na Etapa 2 (seção 4.1) | — |
| P2 | ~~Pacientes por etapa nas classes seen~~ | Resolvida: D23 | — |
| P3 | ~~k-shots viáveis~~ | Resolvida: D27 | — |
| P4 | ~~Tamanho do query~~ | Resolvida: D28 | — |
| P5 | ~~Anomalias nos IDs de paciente~~ | Resolvida: D19 a D22 (seção 4.2) | — |
| P6 | Agregação entre sementes | Escolher a forma de IC entre sementes | Média ± desvio padrão das sementes, mais o IC dos episódios |
| P7 | Decomposição V1/V2 | A subtração P-IMG − P-MIX não isola o V2 (seção 6.2, correção após o piloto). Decidir na Etapa 9 | V2 por meta × ImageNet; V1 controlando frac_V1 |

---

## 11. Etapas de desenvolvimento

Cada etapa tem um **entregável** e um **critério de pronto**. Não avançar sem cumprir o critério.

### Etapa 0: Ambiente e repositório
- Criar o repositório Git do TCC (separado das referências) com `src/`, `scripts/`,
  `notebooks/`, `data/` (fora do Git), `results/` (fora do Git), `docs/`
  e `tests/`.
- Documentação na branch órfã `docs`, montada em `docs/` via `git worktree`
  (ver `CLAUDE.md`).
- Ambiente Python 3.12 gerenciado com **uv**, com `torch`, `torchvision`, `timm`,
  `torchmetrics`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`,
  `pytest`.
- Criar `docs/DIARIO.md` (caderno de laboratório: data, o que rodou, resultado, decisão).
- **Pronto quando:** `src/device.py` encontra o acelerador da máquina (ou a CPU) e
  um passo de treino da MobileNet roda nele (`tests/test_device.py`), no macOS e no
  Windows.

### Etapa 1: Entender o código de referência (sem BreakHis)
- Ler o `meta-cxr` na ordem `config.toml` → `args.py` → `data.py` →
  `method/base.py` → `method/batchbased.py` → `network/factory.py` →
  `pretrain.py` → `adapt.py`.
- **Pronto quando:** você consegue explicar, sem olhar, o caminho de uma imagem do
  disco até a AUC de um episódio.

### Etapa 2: Dados (sem deep learning)
- `scripts/build_csv.py`: varre o BreakHis e gera `data/breakhis_raw.csv` (seção 4.1).
- `notebooks/01_eda.ipynb`, sobre o CSV bruto: anomalias de identificação; pacientes por subtipo; imagens por subtipo ×
  magnificação; **imagens por paciente × magnificação** (mínimo, mediana, máximo);
  pacientes presentes em todas as magnificações; IDs em mais de um subtipo;
  visualização de imagens do mesmo paciente lado a lado.
- `notebooks/02_clear_csv.ipynb`: aplica as decisões da EDA e gera `data/breakhis.csv`;
  viabilidade dos k-shots sobre o CSV limpo.
- Atualizar a seção 4.1 e resolver P1, P3, P4 e P5.
- **Pronto quando:** o CSV tem 7.909 linhas (ou a diferença está explicada) e as
  tabelas estão salvas em `docs/`.

### Etapa 3: Divisões (splits)
- `src/splits.py`: dado (CSV limpo, semente), gera
  - a divisão **por paciente** entre etapas (classes seen; unseen fixas por etapa), a
    mesma nas 4 magnificações (D24);
  - a divisão **por imagem** com os **mesmos tamanhos** por classe, magnificação e etapa.
- Salvar em `data/splits/seed{S}.csv` (D25); sementes (0, 10, 42, 109, 2026) e frações como constantes no próprio `splits.py`.
- **Pronto quando:** os testes `tests/test_splits.py` passam (seção 12).

### Etapa 4: Amostrador de episódios
- `src/episodes.py`: sorteia um episódio (classes, support e query, com os IDs dos
  pacientes) a partir de uma divisão (`etapa_pac` ou `etapa_img`), com sorteio por
  imagem ou por paciente; regras da seção 6.4 (D27 a D32).
- **Pronto quando:** os testes `tests/test_episodes.py` passam e 1.000 episódios de
  cada configuração são gerados sem erro.

### Etapa 5: Modelo e treino
Em quatro passos, com revisão entre eles (D40):
- **5a** `src/cache.py`: gera `data/cache/{mag}.npz` uma vez (D35, D36) e carrega.
- **5b** `src/model.py` (backbone, normalização, embeddings) e `src/adapt.py` (cabeça
  nova por episódio, D37, e todas as métricas da seção 8).
- **5c** `src/train.py`: `treina(seed, mag, divisao, micro_batch_size=16, max_memory_usage=0.75, allow_swap=False)`;
  meta-treino em micro-lotes configuráveis (D43) + época 0, que concorre no early stopping (D38, D48) + mval com as 100 tarefas fixas +
  early stopping → `best_model.pt`, `epocas.csv`, `config.json` (D41). Nunca lê o mtst.
- **5d** `src/evaluate.py`: embeddings do mtst com os dois backbones da rodada e
  3 protocolos × 3 k × 1.000 episódios → CSVs da seção 8.4.
- **Pronto quando:** os testes de sanidade da seção 12 passam numa magnificação e
  numa semente.
- **Concluída em 21/10/2026:** 23 testes rápidos e 3 lentos passam (lentos na RX 7600);
  rodada completa da semente 0 a 40× em 39 min (resultados na seção 16).

### Etapa 6: Contraprova com scikit-learn
- `notebooks/03_contraprova.ipynb`: com os mesmos embeddings e episódios, treinar
  `LogisticRegression` (one-vs-rest) do scikit-learn e comparar a AUC com a cabeça
  em PyTorch.
- **Pronto quando:** as AUCs médias ficam próximas (diferença de poucos décimos) e
  isso está registrado no diário.
- **Concluída em 21/10/2026** (semente 0, 40×, os 18.000 episódios, `C = 1`): HM média
  das duas cabeças a no máximo 0,64 ponto (em geral, poucos décimos), correlação por
  episódio de 0,93 a 0,99 e Δ_total igual a menos de 1 ponto.

### Etapa 7: Experimento piloto
- Uma magnificação (ex.: 40×), uma semente, os três protocolos, k = 5, 200 episódios.
- Conferir se os números fazem sentido e quanto tempo levou.
- **Levar ao orientador** antes da grade completa (marco "está rodando").
- **Pronto quando:** o orientador valida o piloto.
- A rodada completa da semente 0 a 40× (Etapa 5) cobre e amplia o piloto: três
  protocolos, k = 1, 5 e 10, 1.000 episódios e os dois backbones (seção 16).

### Etapa 8: Grade completa
- `scripts/run_grid.py` define a grade em constantes e roda todas as combinações, pulando
  as que já têm resultado (permite retomar).
- **Pronto quando:** todos os CSVs existem e nenhum teste de vazamento falhou.

### Etapa 9: Análise
- `notebooks/04_resultados.ipynb`: tabela central (linhas: magnificação × k; colunas:
  HM em P-IMG, P-MIX e P-PAC, Δ_total, Δ_V1, Δ_V2, Δ%); gráfico HM × k-shot (uma
  linha por protocolo, um painel por magnificação, no estilo da Fig. 6 do artigo);
  a mesma análise com a acurácia balanceada por paciente; e a exposição × AUC no P-IMG.

### Etapa 10: Escrita (em paralelo a partir da Etapa 5)
- Metodologia (quando a Etapa 5 estabilizar) → resultados (Etapa 9) → introdução e
  conclusão. Fundamentação teórica em paralelo.
- Declaração de uso de IA conforme a norma da instituição.

---

## 12. Testes de sanidade obrigatórios

1. **Anti-vazamento entre etapas (P-PAC/P-MIX):** a interseção de pacientes entre
   meta-treino, meta-validação e meta-teste é vazia.
2. **Anti-vazamento no episódio (P-PAC):** a interseção de pacientes entre support e
   query é vazia em 100% dos episódios.
3. **Tamanhos pareados:** P-IMG e P-PAC têm o mesmo número de imagens por classe e
   por etapa, na mesma semente.
4. **Unicidade:** nenhuma imagem aparece em duas etapas, e nenhuma imagem está no
   support e no query do mesmo episódio.
5. **Magnificação:** todas as imagens de um experimento são da magnificação dele.
6. **Rótulos embaralhados:** com rótulos aleatórios, a AUC fica perto de 50.
7. **Overfit proposital:** com ~20 imagens, a perda do meta-treino vai a quase zero.
8. **Reprodutibilidade:** a mesma semente gera exatamente as mesmas divisões e os mesmos
   episódios. O treino em MPS/DirectML não é determinístico bit a bit: entre execuções,
   exigir só diferenças pequenas nas métricas.
9. **Contraprova** com scikit-learn (Etapa 6).

---

## 13. Riscos e armadilhas

- Escolher hiperparâmetros olhando o meta-teste.
- Calcular estatísticas (normalização, etc.) com dados de teste.
- Comparar protocolos com quantidades diferentes de dados.
- Usar uma única semente e concluir sobre diferenças pequenas.
- Misturar magnificações sem perceber (V3).
- Classes com poucos pacientes: o resultado depende de quais pacientes caíram em
  cada etapa (por isso as várias sementes).
- O k-shot alto pode ser inviável no P-PAC por falta de imagens por paciente.
  Isso é **um resultado**, não um erro: documentar.
- Código gerado por IA sem conferência: toda peça nova precisa de teste ou contraprova.

---

## 14. Perguntas para levar ao orientador

1. Validação do piloto (Etapa 7) antes de rodar a grade completa.

---

## 15. Referências principais

- Montalvo-Lezama, B.; Fuentes-Pineda, G. MetaChest: generalized few-shot learning
  of pathologies from chest X-rays. *Vis. Comput. Ind. Biomed. Art* 9:4 (2026).
- Spanhol, F. A. et al. A dataset for breast cancer histopathological image
  classification (BreakHis). *IEEE Trans. Biomed. Eng.* 63(7), 2016.
- Snell, J.; Swersky, K.; Zemel, R. Prototypical networks for few-shot learning. NeurIPS 2017.
- Chen, W.-Y. et al. A closer look at few-shot classification. ICLR 2019.
- Shakeri, F. et al. FHIST: a benchmark for few-shot classification of histological
  images. arXiv:2206.00092 (2022).
- Howard, A. et al. Searching for MobileNetV3. ICCV 2019.
- Xian, Y.; Schiele, B.; Akata, Z. Zero-shot learning — the good, the bad and the
  ugly. CVPR 2017 (média harmônica em GFSL).

---

## 16. Resultados preliminares: piloto (semente 0, 40×)

Rodada completa no Windows (RX 7600) em 21/10/2026, 39 min: treino `pac` com 35 épocas
(melhor 24, HM da mval 58,9; época 0: 57,7), treino `img` com 45 épocas (melhor 35, HM
da mval 95,1; época 0: 84,3) e avaliação com 18.000 episódios (~17 min). Imagens a 40×:
mtrn 651, mval 619, mtst 605; os 1.000 episódios cobrem 100% do mtst. Auditoria: nenhum
paciente no support e no query no P-PAC.

**HM no mtst** (média de 1.000 episódios; IC95 de ±0,3 a ±1,3):

| Backbone | Protocolo | k = 1 | k = 5 | k = 10 | frac_V1 (k = 1 → 10) | frac_V2 |
|---|---|---|---|---|---|---|
| meta-treinado | P-IMG | 80,9 | 91,5 | 94,8 | 0,22 → 0,76 | 0,67 |
| | P-MIX | 75,5 | 91,0 | 95,8 | 0,42 → 0,93 | 0 |
| | P-PAC | 55,6 | 56,7 | 56,1 | 0 | 0 |
| só ImageNet | P-IMG | 68,8 | 84,8 | 90,4 | 0,22 → 0,76 | 0 |
| | P-MIX | 77,4 | 92,6 | 96,2 | 0,42 → 0,93 | 0 |
| | P-PAC | 56,9 | 58,1 | 58,0 | 0 | 0 |

**Indícios** (uma semente e uma magnificação; a confirmar na grade):

1. **Vazamento total (meta):** Δ = P-IMG − P-PAC = 25,3 / 34,7 / 38,7 pontos de HM
   (+45% / +61% / +69% sobre o P-PAC) para k = 1 / 5 / 10.
2. **O ganho com mais exemplos é quase todo vazamento:** no P-IMG a HM sobe 14 pontos de
   k = 1 para k = 10; no P-PAC fica estável. O frac_V1 do P-IMG cresce com k.
3. **O meta-treino só ajuda com V2:** diferença pareada meta − ImageNet de +12,1 / +6,7 /
   +4,4 no P-IMG (AUC seen +21 com k = 1), de −2,0 a −0,4 no P-MIX e de −1,3 a −1,9 no
   P-PAC (só k = 10 significativo, Wilcoxon p = 0,008). No P-PAC o meta-treino sobe a AUC
   seen (+3 a +4) e derruba a unseen (−6 a −7): especializa nas classes seen e nos
   pacientes do mtrn (perda do treino ~0,002).
4. **O V1 domina:** mesmo sem meta-treino (ImageNet, sem V2), o P-IMG fica 12 a 32
   pontos acima do P-PAC; no P-IMG, a correlação entre frac_V1 e HM é ~0,25 em todo k.
5. **A HM por episódio varia muito:** desvio-padrão de ~18 a 21 no P-PAC; ~20% dos
   episódios abaixo de 40 e ~22% acima de 75 nos dois backbones; a correlação da HM
   entre backbones, episódio a episódio, é de 0,58 a 0,70 (o sorteio dos pacientes pesa
   mais que o backbone).
6. **Sinal de atalho pela lâmina:** no P-PAC, a AUC do TA fica **abaixo de 50** (41,8
   ImageNet; 34,8 meta), com apenas 2 pacientes de TA no mtst (um no support, o outro no
   query); a do F fica em ~98. Classes com 1 ou 2 pacientes levam a extremos, o que
   reforça a necessidade das várias sementes.
7. **Decomposição:** ver a correção na seção 6.2 (P7).
8. **Reprodutibilidade:** dados, divisões e episódios idênticos entre Mac e Windows; nas
   2 épocas do teste de ponta a ponta, perda e HM da mval coincidiram entre M1 e RX 7600.

