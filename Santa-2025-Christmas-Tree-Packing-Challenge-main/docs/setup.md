# Setup e execução

Este repo usa **Python (NumPy)** e tem partes aceleradas em **JAX** (opcional, mas recomendado para SA/L2O). Há também uma extensão C++ opcional (`fastcollide`) para acelerar o scorer local.

## Setup (recomendado)

```bash
bash scripts/setup_venv.sh
```

Requer **Python 3.12+** (este repo fixa `3.12.3` em `.python-version`). Se preferir manual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Recomendado (JAX habilita SA/L2O):
python -m pip install -U -e ".[train,notebooks]"

# Opcional (dev tooling):
python -m pip install -U -e ".[dev]"

# Opcional (acelera polygons_intersect no scorer local):
python scripts/build/build_fastcollide.py
```

Se você tiver GPU/CUDA, instale o pacote JAX adequado ao seu ambiente.

## Notebooks (VS Code/Jupyter)

Depois do setup, selecione o interpretador/kernel da `.venv` (assim o `ipykernel` instalado via `.[notebooks]` será usado).

Notebooks recomendados:
* `notebooks/01_generate_and_score.ipynb` (gerar + score + log)
* `notebooks/02_optimization_hunt.ipynb` (hunt/otimização para melhores resultados)

### Otimizar a partir de um CSV existente (ex: score 70.78)

No `notebooks/02_optimization_hunt.ipynb`:
* aponte `BASE_SUBMISSION` para o seu arquivo (`Path("/caminho/para/submission.csv")`)
* ajuste `TARGET_SCORE = 69.0` (ou `None` para não parar cedo)
* rode em `MODE="full"` e aumente seeds/iters conforme quiser

O notebook valida em `OVERLAP_MODE="kaggle"` e faz repair automático quando necessário, então o `submission.csv` final fica sem overlap.

## Estrutura do código

* `santa_packing/`: pacote principal (geometria, colisão, SA/L2O, scorer, etc.).
  * `santa_packing/workflow.py`: workflow de alto nível (generate → improve → validar/scorar → archive) e CLI único (`python -m santa_packing`).
  * `santa_packing/cli/`: CLIs menores/internas (`generate_submission`, `improve_submission`, `autofix_submission`, `score_submission`).
  * `santa_packing/_tools/`: ferramentas pesadas de experimento (hunt, sweep/ensemble, treino, bench, etc.).
  * `santa_packing/main.py`: runner de SA batch (JAX; gera `best_packing.png`).
* `bin/`: binaries C++ (solvers e pós-otimização).
* `scripts/`: wrappers e rotinas auxiliares (build, training, submission, etc.).
* `configs/`: configs JSON para evitar flags longas.
* `runs/`: logs/artefatos de treino/experimentos.
* `tests/`: testes unitários.

## Configs (JSON/YAML)

Os comandos principais carregam configs por padrão (quando presentes):

* `python -m santa_packing`: `configs/submit_strong.json` (fallback `configs/submit.json`)
* `generate_submission`: `configs/submit.json`
* `sweep_ensemble`: `configs/ensemble.json`

Para sobrescrever, passe flags normalmente; para trocar/ignorar config: `--config ...` ou `--no-config`.
Mais detalhes em `configs/README.md`.

## Comandos rápidos

Atalhos (Makefile):

```bash
make test
make lint
make format
```

Gerar um submission (baseline):

```bash
python -m santa_packing
```

O comando acima já gera, melhora, valida/scora e exporta um `submission.csv` pronto para enviar ao Kaggle (além de arquivar artefatos em `submissions/`).

Opcional: re-scorar um CSV manualmente (útil para checar outros arquivos):

```bash
python -m santa_packing.cli.score_submission submission.csv --overlap-mode kaggle --pretty
```

Para detalhes de SA, veja `docs/sa.md`. Para ensemble/sweep, veja `docs/ensemble.md`. Para treino (L2O/meta-init/heatmap), veja `docs/l2o.md`.
