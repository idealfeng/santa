# Contribuindo

Obrigado por querer contribuir! Este repositório é um **baseline/laboratório**: mudanças pequenas, bem testadas e com documentação clara ajudam bastante.

## Setup do ambiente

Siga `docs/setup.md` para instruções completas (setup, notebooks, estrutura e comandos).

TL;DR (recomendado):

```bash
bash scripts/setup_venv.sh
```

Opcional (acelera o scorer local): `make build-fastcollide`.

## Rodando testes

```bash
make test
```

Ou:

```bash
python -m pytest
```

## Lint/format

Este repo usa `ruff` (linter + formatter) e `pre-commit`.

```bash
make lint
make format
```

Para auto-corrigir (imports/estilo) quando possível:

```bash
make lint-fix
```

Opcional:

```bash
pre-commit install
pre-commit run -a
```

## Convenções de código

* **Docstrings:** mantenha docstrings em módulos e funções públicas explicando propósito, parâmetros e retornos.
* **Tipagem:** prefira type hints nas APIs públicas.
* **Nomenclatura:** use identificadores em **inglês**, `snake_case` para funções/variáveis e `PascalCase` para classes.
* **Prefixos:** quando fizer sentido (módulos com múltiplas técnicas), use `sa_`/`l2o_` no nome para diferenciar rotinas (ex.: `run_sa_batch`, `train_l2o`).
* **Estilo:** `ruff format` é a fonte de verdade para formatação.

## Onde colocar documentação

* `README.md` deve permanecer curto (primeiros passos + links).
* Documentação detalhada por tema fica em `docs/` (ex.: `docs/sa.md`, `docs/ensemble.md`).

## Checklist antes de abrir PR

* `make test` passa.
* `make lint` e `make format` aplicados.
* Atualizou `docs/`/`README.md` quando necessário.
* Atualizou `CHANGELOG.md` para mudanças relevantes.
