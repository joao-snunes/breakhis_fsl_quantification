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

### 4.1 Subtipos e pacientes

Valores de pacientes **comumente citados, ainda a confirmar no CSV** (Etapa 2):

| Subtipo | Sigla | Classe | Pacientes (aprox.) | Imagens (aprox., 4 mags) |
|---|---|---|---|---|
| Carcinoma ductal | DC | M | ~38 | ~3.450 |
| Fibroadenoma | F | B | ~10 | ~1.010 |
| Carcinoma mucinoso | MC | M | ~9 | ~790 |
| Adenoma tubular | TA | B | ~7 | ~570 |
| Carcinoma papilífero | PC | M | ~6 | ~560 |
| Carcinoma lobular | LC | M | ~5 | ~630 |
| Adenose | A | B | ~4 | ~440 |
| Tumor filoide | PT | B | ~3 | ~450 |

**A verificar na análise exploratória:** a contagem real de pacientes por subtipo;
as imagens por paciente em cada magnificação (o gargalo do k-shot); se todo
paciente tem imagens nas 4 magnificações; e se algum ID de paciente aparece em mais
de um subtipo (há relatos disso no BreakHis).

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
| D15 | Resolução | Redimensionar 700×460 → **384×384**; normalização do `pretrained_cfg` do modelo (média 0,5 e desvio 0,5 no `tf_mobilenetv3_small_075`) | [PADRÃO] |
| D16 | Data augmentation | **Desligada** (padrão do código original) | [PADRÃO] |
| D17 | Magnificações misturadas | Só como extra opcional (expõe o vazamento V3) | [PADRÃO] |
| D18 | Hardware e precisão | Sem GPU NVIDIA: macOS (M1, MPS) e Windows (AMD RX 7600, ROCm ou DirectML). Dispositivo escolhido em `src/device.py` (cuda/ROCm → mps → directml → cpu), precisão **fp32** (bf16 só como opção em CUDA) | [DECIDIDO] |

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

### 6.3 Como cada protocolo monta os dados

**Classes seen (DC, F, MC, TA), que aparecem nas três etapas:**

- **P-PAC / P-MIX:** para cada classe seen, sortear os **pacientes** em
  meta-treino, meta-validação e meta-teste (alvo ~60/20/20), com **no mínimo 2
  pacientes** na validação e no teste. Exemplo: DC ≈ 24/7/7; F ≈ 6/2/2;
  MC ≈ 5/2/2; TA ≈ 3/2/2. Todas as imagens de um paciente vão para a mesma etapa.
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
  pacientes restantes**. Se não houver imagens suficientes, registrar e tratar
  (ressortear ou reduzir k_tst), nunca violar a regra.

---

## 7. Desenho experimental

**Fatores (variáveis independentes):**

- Protocolo: P-IMG, P-MIX, P-PAC.
- Magnificação: 40×, 100×, 200×, 400× (cada uma é um experimento independente).
- k-shot: {1, 5, 10, 20}. **[PENDENTE]**: confirmar a viabilidade na análise
  exploratória (seção 10).

**Fixos:** BatchBased, MobileNetV3-Small-0.75, 3-way com 1 unseen, 384×384, sem
augmentation e hiperparâmetros da seção 9.

**Custo:** o pré-treino do BatchBased **não depende do k-shot**. Treina-se o
backbone uma vez por (magnificação, divisão de etapas, semente) e avaliam-se
todos os k-shots com ele. O P-MIX compartilha o backbone do P-PAC (mesma divisão
entre etapas). Portanto:

- pré-treinos = 4 magnificações × 2 divisões (por imagem, por paciente) × S sementes
  → com S = 5, são **40 pré-treinos** (cerca de 900 imagens de meta-treino cada: ~60% das ~1.450 imagens seen por magnificação;
  devem levar minutos com a MobileNet);
- avaliações = 4 magnificações × 3 protocolos × 4 k-shots × S sementes × 1.000 episódios.

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

Também registrar a acurácia e a acurácia balanceada **por imagem**, para comparar a
unidade de contagem.

### 8.3 Exposição ao vazamento (descritiva)

Por episódio, registrar:

- `frac_V1` = fração das imagens do query cujo paciente também está no support;
- `frac_V2` = fração das imagens do query cujo paciente esteve no meta-treino.

No P-PAC, as duas devem ser **0** (teste automático). No P-IMG, mostram *quanto*
vazamento existiu, e permitem relacionar a AUC do episódio com a exposição.

### 8.4 Formato dos resultados

Um CSV por avaliação, **uma linha por episódio**, com as colunas:
`magnificacao, protocolo, semente, k_shot, episodio, classes_seen, classes_unseen,
auc_seen, auc_unseen, hm, acc_img, acc_bal_img, acc_bal_pac, frac_V1, frac_V2`,
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
- Ao fim de cada época: avaliar 100 episódios de meta-validação (fase 2) e calcular
  a HM média.
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
| Resolução | 384×384 | Tabela 3 / `args.py` |
| Normalização | `backbone.pretrained_cfg` (média 0,5, desvio 0,5 nos pesos `tf_`; não os valores clássicos do ImageNet) | `base.py` |
| Augmentation | desligada | `args.py` (`data_aug=False`) |
| Lote do meta-treino | 64 | Tabela 3 |
| Otimizador do meta-treino | AdamW, lr 1e-4 | Tabela 3 / `batchbased.py` |
| Épocas máximas | 150 | Tabela 3 |
| Early stopping | HM da meta-validação, paciência 10 | Tabela 3 |
| Episódios de meta-validação | 100 por época | Tabela 3 |
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

## 10. Pendências

| # | Pendência | Como resolver | Padrão sugerido |
|---|---|---|---|
| P1 | Contagem real de pacientes por subtipo | Análise exploratória (Etapa 2) | — |
| P2 | Pacientes por etapa nas classes seen | Depois de P1 | ~60/20/20, mínimo de 2 na validação e no teste |
| P3 | k-shots viáveis por magnificação | Tabela de imagens por paciente × magnificação | {1, 5, 10, 20}; tirar o 20 se não couber |
| P4 | Tamanho do query (k_tst) | Depois de P3 | 15 por classe (artigo: 30) |
| P5 | Anomalias nos IDs de paciente | Análise exploratória | Tratar e documentar |
| P6 | Agregação entre sementes | Escolher a forma de IC entre sementes | Média ± desvio padrão das sementes, mais o IC dos episódios |

---

## 11. Etapas de desenvolvimento

Cada etapa tem um **entregável** e um **critério de pronto**. Não avançar sem cumprir o critério.

### Etapa 0: Ambiente e repositório
- Criar o repositório Git do TCC (separado das referências) com `src/`, `scripts/`,
  `notebooks/`, `configs/`, `data/` (fora do Git), `results/` (fora do Git), `docs/`
  e `tests/`.
- Documentação na branch órfã `docs`, montada em `docs/` via `git worktree`
  (ver `CLAUDE.md`).
- Ambiente Python 3.12 gerenciado com **uv**, com `torch`, `torchvision`, `timm`,
  `torchmetrics`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `pyyaml`, `tqdm`,
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
- `scripts/build_csv.py`: varre o BreakHis e gera `data/breakhis.csv` com
  `path, classe, subtipo, paciente, magnificacao, seq`.
- `notebooks/01_eda.ipynb`: pacientes por subtipo; imagens por subtipo ×
  magnificação; **imagens por paciente × magnificação** (mínimo, mediana, máximo);
  pacientes presentes em todas as magnificações; IDs em mais de um subtipo;
  visualização de imagens do mesmo paciente lado a lado.
- Atualizar a seção 4.1 e resolver P1, P3, P4 e P5.
- **Pronto quando:** o CSV tem 7.909 linhas (ou a diferença está explicada) e as
  tabelas estão salvas em `docs/`.

### Etapa 3: Divisões (splits)
- `src/splits.py`: dado (CSV, magnificação, semente), gera
  - a divisão **por paciente** entre etapas (classes seen; unseen fixas por etapa);
  - a divisão **por imagem** com os **mesmos tamanhos** por classe e etapa.
- Salvar cada divisão em `data/splits/{mag}_{split}_{seed}.csv`
  (`path, paciente, subtipo, etapa`).
- **Pronto quando:** os testes `tests/test_splits.py` passam (seção 12).

### Etapa 4: Amostrador de episódios
- `src/episodes.py`: `sample_episode(df_etapa, seen, unseen, n_way, n_unseen,
  k, k_tst, group_by_patient, rng)` → support e query com os IDs dos pacientes.
- **Pronto quando:** os testes `tests/test_episodes.py` passam e 1.000 episódios de
  cada configuração são gerados sem erro.

### Etapa 5: Modelo e treino
- `src/model.py`: backbone `timm` + cabeça linear + `freeze_backbone()`.
- `src/train.py`: meta-treino (lotes) + meta-validação por episódios + early stopping
  + checkpoint do melhor backbone.
- `src/adapt.py`: embeddings pré-calculados + adaptação da cabeça + métricas por
  episódio → CSV.
- **Pronto quando:** os testes de sanidade da seção 12 passam numa magnificação e
  numa semente.

### Etapa 6: Contraprova com scikit-learn
- `notebooks/02_contraprova.ipynb`: com os mesmos embeddings e episódios, treinar
  `LogisticRegression` (one-vs-rest) do scikit-learn e comparar a AUC com a cabeça
  em PyTorch.
- **Pronto quando:** as AUCs médias ficam próximas (diferença de poucos décimos) e
  isso está registrado no diário.

### Etapa 7: Experimento piloto
- Uma magnificação (ex.: 40×), uma semente, os três protocolos, k = 5, 200 episódios.
- Conferir se os números fazem sentido e quanto tempo levou.
- **Levar ao orientador** antes da grade completa (marco "está rodando").
- **Pronto quando:** o orientador valida o piloto.

### Etapa 8: Grade completa
- `scripts/run_grid.py` lê `configs/grid.yaml` e roda todas as combinações, pulando
  as que já têm resultado (permite retomar).
- **Pronto quando:** todos os CSVs existem e nenhum teste de vazamento falhou.

### Etapa 9: Análise
- `notebooks/03_resultados.ipynb`: tabela central (linhas: magnificação × k; colunas:
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
