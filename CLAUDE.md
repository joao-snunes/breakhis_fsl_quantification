# CLAUDE.md

Instruções para o Claude Code neste repositório.

## O projeto

TCC de graduação: **quantificar o vazamento de dados por imagem × por paciente em
avaliação few-shot** de imagens histológicas H&E (base BreakHis), usando o método
**BatchBased + MobileNetV3-Small-0.75** do artigo MetaChest (Montalvo-Lezama &
Fuentes-Pineda, 2026).

Leia antes de qualquer tarefa:
- `docs/PROJETO.md`: contexto, decisões, protocolos, métricas e etapas de desenvolvimento.
- `docs/REUSO_CODIGO.md`: o que copiar, simplificar e criar a partir do código de referência.

## Código de referência

- Repositório do artigo: `../meta-cxr`, pasta irmã deste repositório
  (https://github.com/bereml/meta-cxr). Use-o **só como referência**: copie trechos
  quando o `REUSO_CODIGO.md` indicar e não o modifique.

## Organização do Git: código na `main`, documentação na branch `docs`

O repositório começa **privado** e será tornado público quando o trabalho estiver
maduro. Para a `main` ficar limpa e apresentável, a documentação viva fica numa
**branch órfã** (`docs`), sem histórico em comum com a `main`, montada como pasta
via `git worktree`:

```
breakhis_fsl_quantification/   ← branch main (código)
├── .gitignore                 ← ignora docs/, CLAUDE.md, .claude/, data/, results/
├── CLAUDE.md                  ← fora do Git; uma linha só: @docs/CLAUDE.md
└── docs/                      ← worktree da branch docs
    ├── CLAUDE.md  PROJETO.md  REUSO_CODIGO.md  DIARIO.md
```

- Mudanças em `docs/` são commitadas **dentro de `docs/`** (vão para a branch
  `docs`); mudanças de código, na raiz (vão para a `main`). Uma tarefa que mexe nos
  dois gera dois commits.
- Nunca fazer merge entre `main` e `docs`.
- Trabalho que depende de análise e decisão do aluno (ex.: EDA) vai para a branch
  `staging`, criada a partir da `main`; o merge na `main` só acontece depois da
  decisão.
- Mostrar o texto dos commits ao aluno antes de commitar. Commits só com título (sem
  descrição) e **sem** linha `Co-Authored-By`.
- Configurar numa máquina nova, depois do `git clone`:
  ```
  git worktree add docs docs
  echo @docs/CLAUDE.md > CLAUDE.md
  ```

## Ambientes e hardware

- **Não há GPU NVIDIA.** Máquinas de desenvolvimento: macOS (Apple M1, GPU integrada
  via MPS) e Windows (AMD RX 7600, via ROCm ou DirectML).
- O dispositivo é escolhido por `src/device.py` (`auto`: cuda → mps → directml →
  cpu). O ROCm aparece no PyTorch como `cuda`. Nunca usar `.cuda()` fixo.
- Precisão padrão **fp32**; bf16/`GradScaler` só como opção em CUDA.
- Ambiente gerenciado com **uv**, Python 3.12 (`uv sync`, `uv run ...`).

## Regras inegociáveis

1. **Separação por paciente é sagrada.** Em P-PAC, nenhum paciente pode aparecer em
   duas etapas, nem no support e no query do mesmo episódio. Todo código que monta
   divisões ou episódios precisa de `assert` e de teste em `tests/`.
2. **O meta-teste só é usado para a nota final.** Early stopping, escolha de
   checkpoint e qualquer ajuste usam apenas a meta-validação.
3. **O método é fixo:** BatchBased, MobileNetV3-Small-0.75 (`timm`
   `tf_mobilenetv3_small_075.in1k`, o mesmo do `meta-cxr`), sigmoide por classe,
   `BCEWithLogitsLoss` e os hiperparâmetros da seção 9.4 do `PROJETO.md`. Não trocar
   de arquitetura nem de método sem pedido explícito.
4. **Comparações justas:** P-IMG e P-PAC devem ter o mesmo número de imagens por
   classe e por etapa na mesma semente.
5. **Reprodutibilidade:** toda aleatoriedade passa por uma semente explícita; cada
   resultado grava a configuração completa que o gerou.
6. **Configuração em YAML** (`configs/`), não em dezenas de argumentos de linha de comando.
7. **PyTorch puro** (sem PyTorch Lightning), para o loop ficar legível.

## Como trabalhar com o aluno

- O aluno tem experiência em ML tradicional (scikit-learn), mas **é iniciante em
  deep learning**. Precisa entender e defender cada linha na banca.
- Trabalhe em **passos pequenos**, uma etapa do `PROJETO.md` por vez, e pare ao fim
  de cada etapa para revisão.
- Use a skill **ponytail** ao escrever código: enxuto, objetivo, sem abstrações
  especulativas.
- Explique cada peça nova com o paralelo em ML tradicional quando houver (ex.:
  cabeça linear + sigmoide = regressão logística one-vs-rest; separação por
  paciente = `GroupKFold`).
- Comentários e docstrings em **português**; nomes de variáveis e funções podem
  ser em inglês.
- Toda peça nova vem com uma forma de verificação (teste, `assert` ou contraprova).
- Ao final de cada sessão, sugira a entrada do dia para `docs/DIARIO.md`.

## Convenções

- Python 3.12, `torch`, `torchvision`, `timm`, `torchmetrics`, `pandas`, `numpy`,
  `scikit-learn`, `matplotlib`, `pyyaml`, `tqdm`, `pytest`.
- Dados em `data/` e resultados em `results/`, ambos fora do Git.
- Resultados: CSV com **uma linha por episódio** (colunas na seção 8.4 do `PROJETO.md`).
- Siglas: `mtrn`, `mval`, `mtst` (etapas); `P-IMG`, `P-MIX`, `P-PAC` (protocolos);
  subtipos `A, F, PT, TA, DC, LC, MC, PC`.
