# Método e resultados, explicados

Este documento descreve o que foi feito no TCC, do dado bruto ao número final, com os valores
reais usados no código. Serve de base para a metodologia e os resultados do texto científico.
As decisões têm o número delas no `PROJETO.md` (D1 a D57) entre parênteses, para quem quiser
a justificativa completa.

## 1. A pergunta

O artigo base (MetaChest, Montalvo-Lezama e Fuentes-Pineda, 2026) avalia um método few-shot
separando os dados por imagem, sem olhar de qual paciente cada imagem veio. Em histologia isso
é arriscado: as imagens de um paciente são recortes da mesma lâmina, com a mesma coloração e o
mesmo preparo. Um modelo pode acertar porque reconhece a lâmina, e a nota sai mais alta do que
seria com pacientes novos.

O TCC mede o tamanho desse efeito na base BreakHis. O método do artigo fica fixo (mesmo
modelo, mesmo treino, mesmos hiperparâmetros, com as adaptações da seção 9) e muda só a regra
de separação dos dados. A diferença de nota entre a separação por imagem e a separação por
paciente é o vazamento.

Em ML tradicional, é a diferença entre avaliar com `KFold` e com `GroupKFold` usando o paciente
como grupo.

## 2. A base BreakHis e a limpeza

A BreakHis tem 7.909 imagens H&E de tumores de mama, em PNG de 700×460 pixels, em quatro
magnificações (40×, 100×, 200× e 400×). Há 8 subtipos, 4 benignos e 4 malignos. O nome de cada
arquivo traz o subtipo, o paciente e a magnificação, por exemplo `SOB_M_DC-14-2523-40-010.png`
(maligno, carcinoma ductal, paciente 14-2523, 40×, imagem 10).

### 2.1 O que conta como paciente (D21)

O identificador oficial de cada pasta tem um número de registro seguido de letras, como
`14-22549AB`. Nove números aparecem em mais de uma pasta (por exemplo `14-22549AB` e
`14-22549CD`). Aqui, paciente é o número de registro sem as letras, então essas pastas viram um
paciente só. Assim a base passa de 82 "pacientes" (contagem da literatura, por pasta) para 70.

O motivo é a direção do erro. Se duas pastas forem da mesma pessoa e forem tratadas como
pacientes diferentes, sobra vazamento escondido justamente no protocolo que deveria ser limpo,
e o vazamento medido fica menor do que o real. Juntar pastas a mais só custa alguns pacientes.
Há também um indício nos dados: pastas com o mesmo registro têm cor média mais parecida entre si
(distância mediana 18,4) do que pastas de registros diferentes (35,1). A leitura das letras como
blocos da mesma peça cirúrgica é uma inferência; a base não documenta isso.

### 2.2 Limpeza (D19, D20, D22)

Comparando os pixels das imagens (hash MD5 da matriz RGB), apareceram imagens repetidas, e
dois pacientes ficaram com dois subtipos depois da D21. A limpeza removeu 451 imagens:

| Problema | Decisão | Imagens removidas |
|---|---|---|
| Paciente 14-13412: as mesmas 123 imagens estão em DC e em LC | Ficam só em LC, classe com poucos pacientes (D19) | 123 |
| Paciente 14-12204 (LC): 2 pares de imagens idênticas a 40× | Fica a primeira de cada par (D20) | 2 |
| Pacientes 14-21998 (F e PT) e 14-29960 (A e F) | Saem as imagens de F, para o paciente não cair em duas etapas (D22) | 326 |

Nenhuma imagem se repetia entre benigno e maligno. A base usada tem 7.458 imagens de 70
pacientes: 1.875 a 40×, 1.969 a 100×, 1.897 a 200× e 1.717 a 400×.

### 2.3 Subtipos, pacientes e papéis

| Subtipo | Sigla | Classe | Papel no few-shot | Pacientes | Imagens |
|---|---|---|---|---|---|
| Carcinoma ductal | DC | maligno | seen | 37 | 3.328 |
| Fibroadenoma | F | benigno | seen | 5 | 688 |
| Carcinoma mucinoso | MC | maligno | seen | 7 | 792 |
| Adenoma tubular | TA | benigno | seen | 6 | 569 |
| Adenose | A | benigno | unseen da meta-validação | 2 | 444 |
| Carcinoma lobular | LC | maligno | unseen da meta-validação | 4 | 624 |
| Tumor filoide | PT | benigno | unseen do meta-teste | 3 | 453 |
| Carcinoma papilífero | PC | maligno | unseen do meta-teste | 6 | 560 |

"Seen" são as classes que o modelo vê no treino. "Unseen" são classes que ele nunca viu e só
aparecem na avaliação (D2). Cada paciente tem um único subtipo.

Cada magnificação é um experimento separado (D1): nada mistura imagens de magnificações
diferentes, o que evita uma terceira fonte de vazamento (a mesma região do tecido em duas
magnificações).

## 3. Como funciona o few-shot generalizado

O objetivo é classificar com poucos exemplos por classe, incluindo classes que o modelo nunca
viu. Os dados se dividem em três etapas, como treino, validação e teste:

| Etapa | Sigla | Para que serve | Classes |
|---|---|---|---|
| Meta-treino | mtrn | Treinar a rede que extrai as características (backbone) | DC, F, MC, TA |
| Meta-validação | mval | Escolher a melhor época (early stopping) | DC, F, MC, TA + A, LC |
| Meta-teste | mtst | Nota final, usada uma única vez | DC, F, MC, TA + PT, PC |

A avaliação é feita em episódios. Um episódio é uma mini-tarefa de classificação:

- 3 classes sorteadas: 2 seen e 1 unseen da etapa (D11);
- support: k imagens de cada classe, usadas para treinar um classificador novo;
- query: 10 outras imagens de cada classe (k_tst = 10, D28), usadas para medir o acerto.

Em ML tradicional, cada episódio é um pequeno `train_test_split` com 3k imagens de treino e 30
de teste. O k avaliado é 1, 5 ou 10.

## 4. As duas fontes de vazamento

| Fonte | Onde acontece | Exemplo |
|---|---|---|
| V1 | Dentro do episódio, entre support e query | A imagem 001 do paciente X está no support e a 002 do mesmo paciente está no query |
| V2 | Entre etapas, entre meta-treino e meta-teste | O backbone foi treinado com imagens do paciente X, e o paciente X aparece nos episódios do meta-teste |

V1 afeta o classificador do episódio, que é treinado e testado no mesmo paciente. V2 afeta o
backbone, que pode ter decorado características daquele paciente durante o treino.

## 5. A separação dos dados

São duas divisões em etapas, geradas por `src/splits.py` para cada semente e gravadas em
`data/splits/seed{S}.csv` (uma linha por imagem, colunas `etapa_pac` e `etapa_img`).

### 5.1 Divisão por paciente (usada pelo P-PAC e pelo P-MIX)

Para cada classe seen, os pacientes são embaralhados e distribuídos:

- meta-validação: n pacientes;
- meta-teste: n pacientes;
- meta-treino: os que sobram;

com (D23)

$$
n_c = \max\big(2,\ \mathrm{round}(0{,}2 \cdot N_c)\big),
$$

onde $N_c$ é o total de pacientes da classe $c$ (por exemplo, DC: $N = 37$, $n = \max(2, \mathrm{round}(7{,}4)) = 7$;
TA: $N = 6$, $n = \max(2, 1) = 2$). O mínimo de 2 existe
porque o P-PAC precisa de pelo menos um paciente para o support e outro para o query. As classes
unseen não são sorteadas: vão inteiras para a etapa delas.

Todas as imagens de um paciente ficam na mesma etapa, nas quatro magnificações (D24). O sorteio
depende só da semente, então é o mesmo nas quatro magnificações, e a comparação entre
magnificações não se confunde com a sorte da divisão.

Resultado (igual nas 5 sementes; mudam os pacientes, não as quantidades):

| Classe | Pacientes no mtrn | na mval | no mtst |
|---|---|---|---|
| DC | 23 | 7 | 7 |
| F | 1 | 2 | 2 |
| MC | 3 | 2 | 2 |
| TA | 2 | 2 | 2 |
| A, LC | 0 | 2 e 4 | 0 |
| PT, PC | 0 | 0 | 3 e 6 |

O F fica com um único paciente no meta-treino. É uma limitação da base (5 pacientes de F depois
da limpeza), declarada no texto.

### 5.2 Divisão por imagem (usada pelo P-IMG)

Dentro de cada classe seen e magnificação, os rótulos de etapa da divisão por paciente são
embaralhados entre as imagens. Na prática, as imagens são sorteadas soltas, sem olhar o
paciente, mas cada etapa fica com exatamente o mesmo número de imagens por classe que na divisão
por paciente da mesma semente. Assim, a única diferença entre as duas divisões é a regra de
agrupamento (regra 4 do `CLAUDE.md`). As classes unseen ficam iguais nas duas.

Exemplo, semente 0, 40× (igual nas duas divisões):

| Classe | mtrn | mval | mtst |
|---|---|---|---|
| DC | 509 | 191 | 132 |
| F | 22 | 91 | 54 |
| MC | 89 | 41 | 75 |
| TA | 31 | 28 | 90 |
| A, LC | 0 | 114 e 154 | 0 |
| PT, PC | 0 | 0 | 109 e 145 |

Entre todas as sementes e magnificações, o meta-treino tem de 570 a 727 imagens, a meta-validação
de 530 a 672 e o meta-teste de 530 a 661. Os tamanhos variam porque os pacientes têm números de
imagens diferentes.

O código confere com `assert` que nenhum paciente aparece em duas etapas na divisão por paciente
e que as duas divisões têm os mesmos tamanhos.

## 6. Os três protocolos

Cada protocolo combina uma divisão entre etapas com uma forma de montar os episódios:

| Protocolo | Divisão entre etapas | Montagem do episódio | V2 | V1 | Papel |
|---|---|---|---|---|---|
| P-IMG | por imagem | por imagem | sim | sim | Como o artigo base faz |
| P-MIX | por paciente | por imagem | não | sim | Isola o V1 |
| P-PAC | por paciente | por paciente | não | não | Cenário realista, sem vazamento |

### 6.1 Montagem do episódio (`src/episodes.py`)

Primeiro sorteiam-se as 3 classes. Depois, para cada classe, dentro das imagens da etapa:

Por imagem (P-IMG e P-MIX): as imagens da classe são embaralhadas; as k primeiras vão para o
support e as 10 seguintes para o query. O mesmo paciente pode aparecer dos dois lados.

Por paciente (P-PAC):

1. os pacientes da classe são embaralhados;
2. entram no support, na ordem, até somarem pelo menos k imagens;
3. sorteiam-se k imagens entre os pacientes do support;
4. sorteiam-se 10 imagens para o query, só entre os pacientes que sobraram.

Nenhuma imagem aparece dos dois lados em nenhum protocolo. No P-PAC, o código confere com
`assert` que nenhum paciente aparece no support e no query do mesmo episódio. Se um dia faltassem
pacientes para o query, o código pararia com erro; com os k usados isso nunca acontece.

### 6.2 O P-MIX comparado com os outros dois

Um episódio é montado em duas decisões, e cada protocolo faz uma escolha em cada uma:

1. Divisão entre etapas: quais imagens ficam no meta-teste, e com quais imagens o backbone foi
   treinado. Decide se existe V2.
2. Montagem do episódio: como as imagens do meta-teste são repartidas entre support e query.
   Decide se existe V1.

| | 1. Divisão entre etapas | 2. Montagem do episódio | Backbone usado |
|---|---|---|---|
| P-IMG | por imagem | por imagem | treinado na divisão por imagem |
| P-MIX | por paciente | por imagem | treinado na divisão por paciente |
| P-PAC | por paciente | por paciente | treinado na divisão por paciente |

O P-MIX fica no meio: copia o P-PAC na primeira decisão e o P-IMG na segunda.

- P-MIX × P-PAC: o mesmo backbone e o mesmo conjunto de imagens no meta-teste. Os pacientes do
  meta-teste nunca foram vistos no treino, então nenhum dos dois tem V2. A única diferença é a
  montagem do episódio: o P-MIX sorteia imagens soltas, e o mesmo paciente pode cair no support e
  no query (V1); o P-PAC proíbe isso. Logo, P-MIX − P-PAC mede o efeito do V1.
- P-MIX × P-IMG: a mesma montagem do episódio (imagens soltas, com V1). Muda a divisão entre etapas
  e, com ela, o backbone. No P-IMG, os pacientes do meta-teste também têm imagens no meta-treino
  (V2); no P-MIX, não.

Em resumo, o P-MIX permite que meta-teste e meta-avaliação possuam os mesmos pacientes (com imagens diferentes), mas não permite o mesmo paciente no support e query no meta-teste.

#### Exemplo real: classe TA, 40×, semente 0

A classe TA tem 6 pacientes. Na divisão por paciente, cada um vai inteiro para uma etapa; na
divisão por imagem, as mesmas quantidades de imagens por etapa são sorteadas soltas:

| Paciente | Divisão por paciente (P-PAC e P-MIX) | Divisão por imagem (P-IMG): mtrn / mval / mtst |
|---|---|---|
| 14-13200 | mtrn (15 imagens) | 4 / 1 / 10 |
| 14-21978 | mtrn (16) | 5 / 3 / 8 |
| 14-15275 | mval (12) | 4 / 0 / 8 |
| 14-3411 | mval (16) | 3 / 5 / 8 |
| 14-16184 | mtst (74) | 12 / 14 / 48 |
| 14-19854 | mtst (16) | 3 / 5 / 8 |

Nas duas divisões o meta-teste tem 90 imagens de TA. O que muda é de quem elas são. Num episódio
do meta-teste com TA:

- P-PAC: as 90 imagens são de 2 pacientes, 14-16184 e 14-19854. Um deles vai para o support e o
  outro fornece as 10 imagens do query. O classificador sempre é testado num paciente que não viu,
  e o backbone também nunca viu nenhum dos dois.
- P-MIX: as mesmas 90 imagens dos mesmos 2 pacientes, mas sorteadas soltas. Como 74 das 90 são do
  14-16184, o support e o query quase sempre têm imagens dele: há V1. O backbone continua sem ter
  visto esses pacientes: não há V2.
- P-IMG: as 90 imagens vêm dos 6 pacientes, e todos eles também têm imagens no meta-treino (V2).
  Sorteadas soltas, o support e o query podem dividir paciente (V1), mas com 6 pacientes isso
  acontece menos do que no P-MIX, que tem só 2.

Esse último ponto explica um resultado que parece estranho à primeira vista: o P-MIX tem mais V1
que o P-IMG (`frac_V1` de 40% a 93%, contra 21% a 76%, seção 13.5). A divisão por paciente
concentra cada classe em poucos pacientes por etapa, e sorteando imagens soltas desses poucos
pacientes, o mesmo paciente cai dos dois lados com mais frequência. Por isso a subtração
P-IMG − P-MIX não isola o V2 (seção 10.9).

### 6.3 Por que k = 1, 5 e 10 (D27)

Algumas classes têm só 2 pacientes na etapa (TA, MC, F e A), alguns com 12 a 20 imagens numa
magnificação. No P-PAC, um paciente precisa cobrir o support e o outro precisa ter pelo menos 10
imagens para o query. Testando todas as ordens possíveis de sorteio nas 40 combinações reais
(5 sementes × 4 magnificações × mval e mtst), k = 1, 5 e 10 funcionam sempre. k = 15 funciona em
só 18 das 40 e é impossível em uma (TA, mtst, 100×, semente 2026).

Usar k = 15 obrigaria a ressortear episódios, forçando o paciente maior no support e o menor no
query, ou a descartar combinações, deixando a grade desigual. As duas saídas distorcem a
comparação, então k = 15 e k = 20 ficaram de fora. Isso também é um achado: com poucos pacientes
por classe, a própria base limita quantos exemplos se podem usar sem vazamento, e a avaliação por
imagem esconde esse limite.

### 6.4 Episódios pareados e sementes (D29, D31)

Cada episódio tem um gerador de números aleatórios próprio, criado a partir de
`[semente, magnificação, etapa, k, número do episódio]`. O protocolo não entra nessa chave, então
o episódio número i sorteia as mesmas 3 classes nos três protocolos; muda só de onde vêm as
imagens. Isso permite comparar os protocolos episódio a episódio, como comparar dois modelos nos
mesmos folds de uma validação cruzada.

Como a chave não depende de mais nada, cada episódio é reprodutível sozinho: as divisões e os
episódios saíram idênticos no Mac e no Windows.

## 7. O modelo

### 7.1 Backbone

MobileNetV3-Small-0.75 pré-treinada no ImageNet-1K, a mesma do código do artigo
(`timm`: `tf_mobilenetv3_small_075.in1k`, D6), sem a camada de classificação. Cada imagem vira um
vetor de 1.024 números (o embedding). Tem cerca de 1 milhão de parâmetros; o artigo mostra que
arquiteturas leves ficam perto das grandes.

### 7.2 Entrada na resolução nativa (D35)

O artigo usa 384×384. Aqui as imagens entram inteiras, em 700×456, sem redimensionar e sem
deformar: as imagens de 700×460 perdem 2 linhas de pixels em cima e 2 embaixo para igualar às 74
imagens de 700×456 da base. Em histologia, a forma e o tamanho dos núcleos são informação
diagnóstica, e esticar para um quadrado achataria as células. A rede aceita esse formato porque
termina numa média global sobre a área da imagem. O custo é 2,2 vezes mais pixels.

A normalização é a do pré-treino do modelo (média 0,5 e desvio 0,5 por canal), sem data
augmentation (D16).

### 7.3 Classificador do episódio

Uma camada linear de 1.024 entradas para 3 saídas, seguida de sigmoide em cada saída, treinada
com `BCEWithLogitsLoss` sobre rótulos one-hot. Isso é o mesmo que três regressões logísticas
one-vs-rest sobre o embedding, uma por classe.

Para uma imagem $x$, com embedding $z = f(x) \in \mathbb{R}^{1024}$ dado pelo backbone $f$, a
probabilidade de pertencer à classe $c$ é

$$
p_c(x) = \sigma\left(w_c^\top z + b_c\right), \qquad \sigma(t) = \frac{1}{1 + e^{-t}},
$$

com um vetor de pesos $w_c$ e um viés $b_c$ por classe. A perda (`BCEWithLogitsLoss`) é a
entropia cruzada binária média sobre as $n$ imagens do lote e as $C$ classes:

$$
\mathcal{L} = -\frac{1}{nC} \sum_{i=1}^{n} \sum_{c=1}^{C}
\Big[ y_{ic} \log p_c(x_i) + (1 - y_{ic}) \log\big(1 - p_c(x_i)\big) \Big],
$$

onde $y_{ic} = 1$ se a imagem $i$ é da classe $c$ e 0 caso contrário. As probabilidades das
classes não precisam somar 1: cada saída é uma regressão logística independente.

## 8. Treino e avaliação

### 8.1 Fase 1: meta-treino (`src/train.py`)

Um backbone por (semente, magnificação, divisão): a divisão por paciente gera o backbone usado
pelo P-PAC e pelo P-MIX, e a divisão por imagem gera o do P-IMG. O treino nunca lê o meta-teste.

- Rede: backbone + camada linear de 4 saídas (DC, F, MC, TA), treinados juntos como num
  transfer learning comum.
- Dados: todas as imagens do meta-treino, sem pesos por classe (D39). O desequilíbrio é grande:
  a 40× na semente 0, o DC tem 509 das 651 imagens.
- Otimização: AdamW com lr 1e-4, lote de 64 imagens montado com 4 micro-lotes de 16 e acumulação
  de gradientes, porque 64 imagens a 700×456 não cabem na memória da GPU (D43). A perda de cada
  micro-lote $m$ é ponderada pelo tamanho dele dentro do lote $B$, e os gradientes são somados
  antes do passo do otimizador:
  $\mathcal{L}_B = \sum_{m \subset B} \frac{|m|}{|B|}\, \mathcal{L}_m$.
  Isso dá a mesma perda média do lote de 64; a única diferença é que a BatchNorm calcula as
  estatísticas em cada micro-lote.
- Ao fim de cada época, a meta-validação avalia o backbone nas mesmas 100 tarefas (D30), com
  k = 5 e k_tst = 10 (D32), e calcula a HM média. As tarefas são fixas para a diferença entre
  épocas vir do backbone, e não do sorteio.
- A meta-validação segue o protocolo do backbone (D34): episódios por paciente na divisão por
  paciente e por imagem na divisão por imagem.
- Early stopping: para depois de 10 épocas sem melhora da HM (paciência 10), com no máximo 150
  épocas. Guarda-se a época de maior HM.
- A época 0 é o backbone só do ImageNet, avaliado antes de qualquer treino, e concorre com as
  outras (D38, D48). Se nenhuma época superar a época 0, o backbone escolhido é o próprio
  ImageNet.
- A camada de 4 saídas é descartada no fim; segue só o backbone.

### 8.2 Fase 2: adaptação em cada episódio (`src/adapt.py`)

Para cada episódio de meta-validação ou meta-teste:

1. O backbone fica congelado. Os embeddings de todas as imagens da etapa são calculados uma vez,
   e cada episódio só usa os vetores já prontos. Isso dá o mesmo resultado que passar as imagens
   pelo backbone a cada episódio, porque o backbone não muda e não há augmentation.
2. Cria-se uma camada linear nova (1.024 → 3), com a inicialização padrão do PyTorch sorteada
   pelo gerador do episódio. Os pesos iniciais são os mesmos nos três protocolos e nos dois
   backbones (D45).
3. 100 passos de AdamW com lr 0,005. Em cada passo, usa-se uma amostra de metade do support,
   que tem $3k$ imagens: $b = \max\big(1, \lfloor 0{,}5 \cdot 3k \rfloor\big)$ imagens, ou seja
   1 imagem com $k = 1$, 7 com $k = 5$ e 15 com $k = 10$ (D37).
4. A camada prevê as probabilidades das 30 imagens do query, e calculam-se as métricas.
5. A camada é descartada.

A adaptação roda na CPU (D44), onde o resultado é determinístico; leva cerca de 20 ms por
episódio.

### 8.3 Dois backbones avaliados no meta-teste (D54)

Cada episódio do meta-teste é avaliado com dois backbones:

- `meta`: o melhor backbone do meta-treino;
- `imagenet`: o backbone só do ImageNet, que nunca viu nenhum paciente da base.

Com o `imagenet` não existe V2. A diferença entre protocolos com ele mostra o efeito do V1
sozinho, e a diferença entre `meta` e `imagenet` no mesmo protocolo mostra quanto o meta-treino
acrescenta.

## 9. O que foi mantido e o que mudou em relação ao artigo

| Item | Artigo (MetaChest) | Aqui | Por quê |
|---|---|---|---|
| Método | BatchBased | BatchBased | Método fixo (D7) |
| Backbone | MobileNetV3-Small-0.75, ImageNet | igual | D6 |
| Saída | sigmoide por classe + BCE | igual | D8 |
| Meta-treino | AdamW 1e-4, lote 64, até 150 épocas, paciência 10 | igual (lote 64 em micro-lotes) | Tabela 3 do artigo; D43 |
| Adaptação | 100 passos, 50% do support, lr 0,005 | igual | Código do artigo (D10) |
| Episódio | 3 classes, 2 seen + 1 unseen | igual | D11 |
| Entrada | 384×384 | 700×456 nativa | Preservar a escala do tecido (D35) |
| Precisão | bf16 | fp32 | Sem GPU NVIDIA (D18) |
| k da mval / k_tst | 30 / 30 | 5 / 10 | Poucos pacientes por classe (D32, D28) |
| k no meta-teste | vários, até 30 | 1, 5, 10 | Viabilidade no P-PAC (D27) |
| Episódios no meta-teste | 10.000 | 1.000 | Custo; no piloto, 1.000 cobriram todas as imagens do meta-teste |
| Tarefas da mval | novas a cada época | as mesmas 100 | Early stopping menos sujeito à sorte (D30) |
| Época 0 | não existe | concorre no early stopping | Mostra se o meta-treino ajuda (D48) |
| Paciente | ignorado | controlado (protocolos) | Objeto do TCC |
| Weight decay | não informado | padrão do AdamW no PyTorch (0,01) | Sem valor no artigo |

## 10. Métricas e fórmulas

Todas as métricas são calculadas por episódio, sobre as 30 imagens do query, em `src/adapt.py`.

Notação usada nesta seção, para um episódio:

- $Q$: as imagens do query ($|Q| = 30$, 10 por classe), indexadas por $i$;
- $\mathcal{C} = \{s_1, s_2, u\}$: as 3 classes do episódio, duas seen ($S = \{s_1, s_2\}$) e uma
  unseen ($u$);
- $y_{ic} \in \{0, 1\}$: 1 se a imagem $i$ é da classe $c$;
- $p_{ic} \in [0, 1]$: a probabilidade que o classificador dá à classe $c$ para a imagem $i$
  (seção 7.3);
- $\mathbb{1}[\cdot]$: vale 1 se a condição é verdadeira e 0 se é falsa.

### 10.1 AUC seen e AUC unseen

A AUC (área sob a curva ROC) é a probabilidade de o modelo dar nota maior a um exemplo positivo do
que a um negativo, sorteados ao acaso. 50 é chute; 100 é ordenar todos perfeitamente.

Para um conjunto de pares (imagem, classe) $P$, separados em positivos
$P^+ = \{(i, c) \in P : y_{ic} = 1\}$ e negativos $P^- = \{(i, c) \in P : y_{ic} = 0\}$:

$$
\mathrm{AUC}(P) = \frac{100}{|P^+|\,|P^-|} \sum_{a \in P^+} \sum_{b \in P^-}
\Big( \mathbb{1}[p_a > p_b] + \tfrac{1}{2}\,\mathbb{1}[p_a = p_b] \Big).
$$

A AUC seen junta as duas colunas seen numa única tarefa binária (média "micro", como no código do
artigo):

$$
\mathrm{AUC}_{\text{seen}} = \mathrm{AUC}\big(\{(i, c) : i \in Q,\ c \in S\}\big),
$$

60 pares (30 imagens × 2 classes), dos quais 20 positivos. A AUC unseen usa só a coluna da classe
unseen:

$$
\mathrm{AUC}_{\text{unseen}} = \mathrm{AUC}\big(\{(i, u) : i \in Q\}\big),
$$

30 pares, dos quais 10 positivos. No código, as duas vêm do `torchmetrics` (`multilabel_auroc`
com `average="micro"` e `binary_auroc`), que calcula a mesma área pela curva ROC.

### 10.2 HM (métrica principal)

$$
\mathrm{HM} = \frac{2 \cdot \mathrm{AUC}_{\text{seen}} \cdot \mathrm{AUC}_{\text{unseen}}}
{\mathrm{AUC}_{\text{seen}} + \mathrm{AUC}_{\text{unseen}}}
$$

É a média harmônica das duas AUCs. Ela só fica alta se as duas forem altas: com
$\mathrm{AUC}_{\text{seen}} = 90$ e $\mathrm{AUC}_{\text{unseen}} = 50$, a média comum seria 70,
mas a HM é 64,3.

### 10.3 AUC por classe

A AUC de cada coluna do episódio, separadamente:

$$
\mathrm{AUC}_c = \mathrm{AUC}\big(\{(i, c) : i \in Q\}\big), \qquad c \in \mathcal{C}.
$$

Ficam nas colunas `auc_DC`, `auc_F` etc. dos CSVs (vazias para as classes que não estão no
episódio). Uma AUC abaixo de 50 quer dizer que o modelo ordena aquela classe ao contrário.

### 10.4 Acurácia por imagem (`acc_img`)

A classe prevista e a classe verdadeira de cada imagem são

$$
\hat{y}_i = \arg\max_{c \in \mathcal{C}} p_{ic}, \qquad y_i = \text{a classe } c \text{ com } y_{ic} = 1,
$$

e a acurácia é

$$
\mathrm{acc\_img} = \frac{100}{|Q|} \sum_{i \in Q} \mathbb{1}[\hat{y}_i = y_i].
$$

Como o query tem exatamente 10 imagens por classe, ela é igual à acurácia balanceada por imagem
(D46). O acaso dá cerca de 33%.

### 10.5 Acurácia balanceada por paciente (`acc_bal_pac`)

Seja $Q_j$ o conjunto das imagens do paciente $j$ no query e $J_c$ o conjunto dos pacientes da
classe $c$ no query. Em três passos:

$$
a_j = \frac{1}{|Q_j|} \sum_{i \in Q_j} \mathbb{1}[\hat{y}_i = y_i]
\qquad \text{(acerto do paciente } j\text{)}
$$

$$
A_c = \frac{1}{|J_c|} \sum_{j \in J_c} a_j
\qquad \text{(média dos pacientes da classe } c\text{)}
$$

$$
\mathrm{acc\_bal\_pac} = \frac{100}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} A_c
\qquad \text{(média das 3 classes)}
$$

Cada paciente pesa igual dentro da classe e cada classe pesa igual no total. Ela corrige o peso
de pacientes com muitas imagens, mas não remove vazamento.

### 10.6 Exposição ao vazamento

Seja $\mathrm{pac}(i)$ o paciente da imagem $i$, $\mathcal{P}_{\text{sup}}$ o conjunto dos
pacientes do support do episódio e $\mathcal{P}_{\text{mtrn}}$ o conjunto dos pacientes do
meta-treino do backbone:

$$
\mathrm{frac\_V1} = \frac{1}{|Q|} \sum_{i \in Q} \mathbb{1}\big[\mathrm{pac}(i) \in \mathcal{P}_{\text{sup}}\big],
\qquad
\mathrm{frac\_V2} = \frac{1}{|Q|} \sum_{i \in Q} \mathbb{1}\big[\mathrm{pac}(i) \in \mathcal{P}_{\text{mtrn}}\big].
$$

Nos CSVs elas ficam entre 0 e 1; nas tabelas, em %. Para o backbone `imagenet`,
$\mathcal{P}_{\text{mtrn}} = \varnothing$, então $\mathrm{frac\_V2} = 0$. No P-PAC as duas são
zero por construção, e isso é conferido em todos os episódios.

### 10.7 Agregação

Para uma métrica $M$ e uma combinação fixa (backbone, protocolo, magnificação, $k$), seja
$M_{s,e}$ o valor no episódio $e$ da semente $s$, com $E = 1.000$ episódios e as 5 sementes
$s \in \{0, 10, 42, 109, 2026\}$.

1. Média da rodada (uma por semente):

$$
\bar{M}_s = \frac{1}{E} \sum_{e=1}^{E} M_{s,e}
$$

2. Valor das tabelas, média ± desvio padrão entre as sementes (P6):

$$
\bar{M} = \frac{1}{5} \sum_{s} \bar{M}_s,
\qquad
\mathrm{dp} = \sqrt{\frac{1}{5 - 1} \sum_{s} \big(\bar{M}_s - \bar{M}\big)^2}.
$$

O desvio entre sementes mostra quanto o resultado muda quando mudam os pacientes de cada etapa.

### 10.8 Diferença de HM entre protocolos (Δ_total)

O Δ_total é uma diferença de desempenho, e não uma medida direta de vazamento (essa é a
exposição, seção 10.6). Ele é atribuído ao vazamento porque P-IMG e P-PAC só diferem na regra de
separação. Calculado dentro de cada semente, com as médias de rodada da HM:

$$
\Delta_s = \overline{\mathrm{HM}}{}^{\,\text{P-IMG}}_s - \overline{\mathrm{HM}}{}^{\,\text{P-PAC}}_s,
\qquad
\Delta\%_s = 100 \cdot \frac{\Delta_s}{\overline{\mathrm{HM}}{}^{\,\text{P-PAC}}_s}.
$$

Depois, como na seção 10.7, reporta-se a média e o desvio padrão de $\Delta_s$ entre as 5
sementes, e a média de $\Delta\%_s$.

Exemplo real (backbone `meta`, 40×, k = 5):

| Semente | P-IMG | P-PAC | Δ | Δ% |
|---|---|---|---|---|
| 0 | 91,5 | 56,7 | 34,7 | 61,2% |
| 10 | 94,2 | 62,9 | 31,3 | 49,8% |
| 42 | 93,0 | 57,5 | 35,5 | 61,8% |
| 109 | 92,4 | 51,4 | 41,0 | 79,8% |
| 2026 | 90,5 | 69,2 | 21,4 | 30,9% |

Resumo: Δ = 32,8 ± 7,3 pontos e Δ% = 57% (média das 5 porcentagens). A média do Δ é igual à
diferença das médias (92,3 − 59,5), porque média é linear:
$\frac{1}{5}\sum_s \Delta_s = \overline{\mathrm{HM}}{}^{\,\text{P-IMG}} - \overline{\mathrm{HM}}{}^{\,\text{P-PAC}}$.
O desvio de 7,3 vem das 5 diferenças, e não dos desvios de cada protocolo. Já a média dos
$\Delta\%_s$ (57%) é um pouco diferente da razão das médias ($32{,}8 / 59{,}5 = 55\%$).

### 10.9 Separar V1 e V2 (em aberto, P7)

A ideia inicial era

$$
\Delta_{V2} \approx \mathrm{HM}^{\text{P-IMG}} - \mathrm{HM}^{\text{P-MIX}},
\qquad
\Delta_{V1} \approx \mathrm{HM}^{\text{P-MIX}} - \mathrm{HM}^{\text{P-PAC}},
$$

de modo que $\Delta_{V1} + \Delta_{V2} = \Delta_{\text{total}}$. Isso supõe que P-IMG e P-MIX
tenham o mesmo V1, e não têm: a divisão por paciente concentra poucos pacientes por classe em cada
etapa, então no P-MIX o query divide paciente com o support com mais frequência (seções 6.2 e
13.5). Com o backbone `imagenet`, a subtração chega a dar "V2" negativo. A proposta é medir o V2
pela diferença `meta` − `imagenet` no P-IMG e o V1 controlando o `frac_V1`, a decidir na análise.

## 11. A grade de experimentos

| Fator | Valores |
|---|---|
| Sementes (divisões de pacientes) | 0, 10, 42, 109, 2026 |
| Magnificações | 40×, 100×, 200×, 400× |
| Protocolos | P-IMG, P-MIX, P-PAC |
| k | 1, 5, 10 |
| Backbones avaliados | meta, imagenet |
| Episódios por combinação | 1.000 no meta-teste; 100 fixos na meta-validação |

- 40 treinos (5 sementes × 4 magnificações × 2 divisões).
- 20 avaliações, cada uma com 18.000 linhas (3 protocolos × 3 k × 1.000 episódios × 2
  backbones), 360.000 linhas no total.
- Tempo: 11,1 h numa GPU AMD RX 7600 (ROCm), com 16 a 18 s por época de treino; no piloto, a
  avaliação de uma rodada levou 17 min.

Arquivos: `results/train/{mag}_{divisao}_seed{S}/` (curva por época e configuração),
`results/mtst/{mag}_seed{S}.csv` (uma linha por episódio), `..._pacientes.csv.gz` (pacientes do
support e do query de cada episódio, para auditoria) e `..._config.json` (commit, versões e
constantes de cada rodada). A análise está em `notebooks/04_resultados.ipynb`.

## 12. Verificações

- Testes automáticos (`tests/`): nenhum paciente em duas etapas; nenhum paciente no support e no
  query no P-PAC; mesmos tamanhos entre as divisões; nenhuma imagem dos dois lados; rótulos
  embaralhados dão AUC perto de 50; com 20 imagens, a perda do treino vai a quase zero; mesma
  semente gera as mesmas divisões e episódios.
- Na grade: `frac_V1` e `frac_V2` iguais a zero em todos os episódios do P-PAC; `frac_V2` zero no
  P-MIX; auditoria sem nenhum paciente dos dois lados no P-PAC.
- Reprodutibilidade entre máquinas: divisões, episódios e cache de imagens idênticos no Mac (M1) e
  no Windows (RX 7600); no teste de 2 épocas, perda e HM da meta-validação coincidiram nas duas
  GPUs.
- Contraprova com o scikit-learn (Etapa 6, branch `contraprova`): trocando a camada linear do
  PyTorch por `LogisticRegression` one-vs-rest (C = 1) nos mesmos embeddings e episódios do piloto
  (18.000 episódios), a HM média muda no máximo 0,64 ponto, a correlação por episódio fica entre
  0,93 e 0,99 e o Δ_total muda menos de 1 ponto.

## 13. Resultados

Números de `notebooks/04_resultados.ipynb`. HM em %, média das 5 sementes ± desvio entre
sementes.

### 13.1 HM no meta-teste, backbone meta-treinado

| Magnificação | k | P-IMG | P-MIX | P-PAC |
|---|---|---|---|---|
| 40× | 1 | 81,6 ± 1,2 | 75,9 ± 3,4 | 57,5 ± 6,2 |
| | 5 | 92,3 ± 1,4 | 91,0 ± 1,7 | 59,5 ± 6,7 |
| | 10 | 95,4 ± 0,9 | 95,4 ± 1,1 | 59,8 ± 6,9 |
| 100× | 1 | 82,0 ± 2,1 | 72,8 ± 2,4 | 55,1 ± 1,1 |
| | 5 | 92,3 ± 1,5 | 88,8 ± 1,8 | 56,7 ± 0,6 |
| | 10 | 94,9 ± 1,1 | 94,3 ± 1,2 | 56,4 ± 1,3 |
| 200× | 1 | 80,1 ± 1,0 | 71,7 ± 3,2 | 52,0 ± 7,2 |
| | 5 | 91,1 ± 0,7 | 89,2 ± 1,9 | 53,6 ± 8,9 |
| | 10 | 94,4 ± 0,7 | 94,8 ± 1,1 | 53,9 ± 9,3 |
| 400× | 1 | 79,6 ± 2,0 | 72,2 ± 5,9 | 55,7 ± 9,0 |
| | 5 | 90,4 ± 1,3 | 87,8 ± 3,2 | 56,3 ± 10,4 |
| | 10 | 93,6 ± 0,8 | 93,1 ± 1,8 | 56,8 ± 10,4 |

### 13.2 HM no meta-teste, backbone só ImageNet (sem V2)

| Magnificação | k | P-IMG | P-MIX | P-PAC |
|---|---|---|---|---|
| 40× | 1 | 68,8 ± 0,5 | 76,4 ± 3,5 | 58,0 ± 6,3 |
| | 5 | 84,7 ± 0,2 | 91,2 ± 1,8 | 60,1 ± 6,8 |
| | 10 | 90,0 ± 0,3 | 95,3 ± 1,1 | 60,6 ± 7,0 |
| 100× | 1 | 64,2 ± 0,4 | 72,0 ± 1,8 | 53,3 ± 3,5 |
| | 5 | 80,1 ± 0,6 | 88,0 ± 1,2 | 54,2 ± 4,6 |
| | 10 | 87,1 ± 0,5 | 93,8 ± 0,9 | 54,3 ± 4,9 |
| 200× | 1 | 62,4 ± 0,9 | 70,8 ± 2,2 | 52,4 ± 5,6 |
| | 5 | 78,5 ± 0,8 | 88,8 ± 0,8 | 54,0 ± 6,3 |
| | 10 | 86,1 ± 0,9 | 94,6 ± 0,6 | 54,6 ± 6,5 |
| 400× | 1 | 61,1 ± 0,9 | 67,7 ± 1,3 | 48,3 ± 2,8 |
| | 5 | 76,3 ± 1,0 | 86,3 ± 1,3 | 48,4 ± 3,8 |
| | 10 | 83,4 ± 0,9 | 92,4 ± 1,2 | 49,2 ± 4,7 |

### 13.3 Diferença de HM entre protocolos (Δ_total)

Δ_total = P-IMG − P-PAC, em pontos de HM (média ± desvio entre sementes) e em % do P-PAC:

| Magnificação | k | Δ meta | Δ% meta | Δ imagenet | Δ% imagenet |
|---|---|---|---|---|---|
| 40× | 1 | 24,1 ± 6,9 | 43% | 10,9 ± 6,0 | 20% |
| | 5 | 32,8 ± 7,3 | 57% | 24,6 ± 6,9 | 42% |
| | 10 | 35,5 ± 7,2 | 61% | 29,4 ± 7,1 | 50% |
| 100× | 1 | 26,9 ± 1,8 | 49% | 10,9 ± 3,3 | 21% |
| | 5 | 35,5 ± 1,5 | 63% | 25,9 ± 4,2 | 49% |
| | 10 | 38,5 ± 1,2 | 68% | 32,8 ± 4,4 | 61% |
| 200× | 1 | 28,1 ± 7,2 | 56% | 10,0 ± 5,0 | 20% |
| | 5 | 37,5 ± 9,0 | 74% | 24,5 ± 6,1 | 47% |
| | 10 | 40,6 ± 9,2 | 80% | 31,5 ± 6,3 | 59% |
| 400× | 1 | 24,0 ± 7,5 | 45% | 12,8 ± 2,6 | 27% |
| | 5 | 34,2 ± 9,9 | 64% | 27,9 ± 3,5 | 58% |
| | 10 | 36,7 ± 10,2 | 68% | 34,2 ± 4,5 | 71% |

Juntando as magnificações, o Δ médio do backbone meta é 25,8 (k = 1), 35,0 (k = 5) e 37,8
(k = 10) pontos; numa única rodada, vai de 11,6 a 51,6. O P-IMG ficou acima do P-PAC em todas
as 20 rodadas, em todo k, nos dois backbones.

### 13.4 Outras métricas (média das 20 rodadas, backbone meta)

| Protocolo | k | AUC seen | AUC unseen | HM | acc_img | acc_bal_pac |
|---|---|---|---|---|---|---|
| P-IMG | 1 | 88,9 | 75,7 | 80,8 | 68,6 | 67,7 |
| | 5 | 95,3 | 88,5 | 91,5 | 81,7 | 80,8 |
| | 10 | 96,7 | 92,7 | 94,6 | 85,6 | 85,0 |
| P-MIX | 1 | 76,9 | 71,7 | 73,1 | 57,5 | 56,4 |
| | 5 | 91,7 | 87,4 | 89,2 | 78,0 | 76,7 |
| | 10 | 96,0 | 93,1 | 94,4 | 85,8 | 84,9 |
| P-PAC | 1 | 61,7 | 55,3 | 55,1 | 42,3 | 41,5 |
| | 5 | 63,0 | 56,7 | 56,5 | 43,7 | 42,6 |
| | 10 | 63,0 | 57,0 | 56,7 | 43,9 | 42,8 |

A acurácia por imagem e a balanceada por paciente diferem em cerca de 1 ponto: nesta base,
contar por imagem ou por paciente quase não muda a acurácia. A acurácia do P-PAC (42% a 44%) fica
pouco acima do acaso com 3 classes (33%).

### 13.5 Exposição ao vazamento

Média em % do query (backbone meta):

| Protocolo | frac_V1 k = 1 | k = 5 | k = 10 | frac_V2 |
|---|---|---|---|---|
| P-IMG | 20,8 | 59,4 | 76,1 | 66,7 |
| P-MIX | 39,7 | 82,4 | 93,0 | 0 |
| P-PAC | 0 | 0 | 0 | 0 |

O `frac_V2` do P-IMG é sempre 66,7% (20 de 30 imagens): na divisão por imagem, todo paciente das
classes seen tem imagens no meta-treino, então as 20 imagens seen do query sempre têm V2, e as 10
da classe unseen nunca.

A HM sobe com o `frac_V1`. Nos episódios do P-IMG e do P-MIX (backbone meta), a HM média é 76,8
quando de 1% a 25% do query divide paciente com o support e 93,6 quando 75% a 100% divide. No P-PAC,
sem nenhum paciente em comum, a HM média é 56,1.

### 13.6 Efeito do meta-treino (meta − imagenet, mesmos episódios)

Diferença da HM calculada em cada rodada; média ± desvio padrão entre as 20 rodadas:

| Protocolo | k = 1 | k = 5 | k = 10 |
|---|---|---|---|
| P-IMG | +16,7 ± 2,7 | +11,6 ± 2,6 | +7,9 ± 1,9 |
| P-MIX | +1,4 ± 3,4 | +0,6 ± 1,7 | +0,4 ± 0,8 |
| P-PAC | +2,1 ± 5,2 | +2,4 ± 5,8 | +2,0 ± 5,5 |

O meta-treino ajuda muito só quando há V2: no P-IMG, o backbone treinou com os mesmos pacientes
do meta-teste, e o ganho é grande e estável. No P-MIX e no P-PAC, o ganho médio é pequeno e o
desvio entre rodadas é maior que ele: em algumas rodadas o meta-treino ajuda, em outras piora.

Nos treinos, a divisão por paciente escolheu a época 0 ou 1 em 12 dos 20 casos (mediana da melhor
época: 1), isto é, o backbone ficou praticamente igual ao do ImageNet. A divisão por imagem
escolheu épocas de 21 a 63 (mediana 36,5), com HM média da meta-validação de 93,4, contra 58,1 na
divisão por paciente.

### 13.7 Variação e extremos

- No P-PAC, a média de uma rodada vai de 42,1 (200×, semente 109) a 74,2 (400×, semente 42) com o
  backbone meta, uma amplitude de 28 a 32 pontos. No P-IMG a amplitude é de 4 a 7 pontos.
- Por episódio, o desvio padrão da HM é de 17 a 20 no P-PAC e de 5 a 13 no P-IMG. No P-PAC,
  36% a 39% dos episódios do backbone meta têm HM abaixo de 50; no P-IMG com k ≥ 5, quase nenhum.
- AUC por classe no P-PAC (k = 5, backbone meta): DC 71,8, F 73,0, MC 47,5, TA 63,0, PT 58,6,
  PC 54,7. Por semente, F, MC e TA ficam abaixo de 50 em alguma delas (F 41,4 na semente 109;
  TA 39,6 na semente 0). São classes com 2 pacientes no meta-teste: um vai para o support e o outro
  para o query, e o classificador pode aprender traços do paciente do support que não valem para o
  outro.

### 13.8 Resumo dos achados

1. Separar por imagem infla a HM em 24 a 41 pontos (43% a 80%), em todas as magnificações e k.
2. Sem vazamento, a HM fica entre 52 e 60 e não melhora com mais exemplos; com vazamento, sobe de
   ~80 para ~95 de k = 1 a k = 10. O ganho com k vem do vazamento.
3. O V1 sozinho já explica a maior parte: o P-MIX fica de 16 a 41 pontos acima do P-PAC, e com o
   backbone `imagenet` (sem V2) o P-MIX supera o P-IMG.
4. O meta-treino só ajuda de verdade quando há V2 (+8 a +17 no P-IMG); sem vazamento, acrescenta
   cerca de 2 pontos.
5. Sem vazamento, o resultado depende muito de quais pacientes caíram em cada etapa; com vazamento,
   o resultado parece estável e preciso.

## 14. Limitações e pontos em aberto

- Poucos pacientes por classe: F tem 1 paciente no meta-treino, e várias classes têm 2 na
  meta-validação e no meta-teste. O resultado sem vazamento depende muito do sorteio, o que as 5
  sementes tentam cobrir.
- A unidade de paciente (D21) é uma inferência sobre as letras do identificador. Também não dá para
  garantir que registros diferentes sejam pessoas diferentes.
- A paciência 10 do artigo pode ter parado cedo 5 treinos da divisão por paciente que ainda estavam
  se recuperando na meta-validação. Uma análise de sensibilidade com paciência maior está em
  discussão.
- A separação entre V1 e V2 por subtração não funciona (P7, seção 10.9).
- A entrada em 700×456 e o uso de fp32 desviam do artigo; os números não se comparam diretamente
  aos da MetaChest, que é outra base. Os micro-lotes mudam levemente o comportamento da BatchNorm.
- 1.000 episódios por combinação, contra 10.000 no artigo. No piloto (semente 0, 40×), eles
  cobriram todas as imagens do meta-teste; nas outras rodadas isso não foi conferido.
- A contraprova com o scikit-learn foi feita só na semente 0 a 40×, e se ela entra no método ainda
  está em decisão.
