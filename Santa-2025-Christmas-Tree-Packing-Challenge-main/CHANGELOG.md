# Changelog

Registra mudanças relevantes entre versões deste baseline.

O formato segue a ideia do *Keep a Changelog* (seções por tipo de mudança).

## Unreleased

### Added
* `docs/` com documentação por tema.
* `CONTRIBUTING.md` e `CHANGELOG.md`.
* Configs centralizadas em `configs/` e loader compartilhado em `santa_packing/cli/config_utils.py`.
* Testes unitários em `tests/` (geometria, colisão e SA) e CI rodando `make test`.
* READMEs em `scripts/*/README.md` explicando o papel dos scripts e apontando para as CLIs do pacote.
* Makefile expandido com alvos para tarefas comuns (ex.: `generate-submission`, `sweep-ensemble`, `train-*`, `check`).

### Changed
* `README.md` simplificado e com links para `docs/`.
* CLIs principais passaram a carregar configs padrão (com override via flags).
* Notebook `notebooks/run_l2o.ipynb` e scripts foram padronizados para invocar `python -m santa_packing.cli.*`.
* Padronização de nomenclatura: `snake_case` em funções/métodos, classes em `PascalCase`, e rotinas específicas com `sa_`/`l2o_` no nome quando aplicável.
* `ruff` passou a validar ordenação de imports (substitui `isort`) e o CI agora roda lint + format-check.
