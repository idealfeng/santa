[![CI](https://github.com/Fear-Hungry/Santa-2025-Christmas-Tree-Packing-Challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/Fear-Hungry/Santa-2025-Christmas-Tree-Packing-Challenge/actions/workflows/ci.yml)

# Santa 2025 — Christmas Tree Packing Challenge (baseline)

Este repositório é um **baseline/laboratório** para o desafio de *packing* 2D do Santa 2025: gerar um `submission.csv` com `(x, y, deg)` para `n=1..200` sem overlap e com bom score.

## Comece aqui

```bash
bash scripts/setup_venv.sh
python -m santa_packing
# -> escreve ./submission.csv e arquiva o run em ./submissions/<timestamp>...
```

Via Python (script/notebook): `from santa_packing.workflow import solve`

## Melhorar o score (pipeline atual)

O pipeline padrão (`python -m santa_packing`) já roda pós-processamento (subset-smoothing + polish do `n=200`)
e valida em `--overlap-mode kaggle` (mesma semântica do Kaggle: “touching” é permitido).

Reprodução do melhor run local atual (atingiu `~72.816` com validação *strict*):

```bash
# Requer os binários C++ em `bin/` (compact_contact + post_opt).
# Roda vários seeds, faz ensemble por n, aplica smoothing e post-opt.
best_csv=$(python -m santa_packing._tools.hunt_compact_contact \
  --base submission.csv \
  --out-dir /tmp/hunt_cc \
  --seeds 4000..4127 --jobs 16 \
  --smooth-window 199 --post-opt)

cp "$best_csv" submission.csv
python -m santa_packing.cli.score_submission submission.csv --nmax 200 --overlap-mode strict --pretty
```

## Submissão no Kaggle (CLI)

```bash
.venv/bin/kaggle competitions submit -c santa-2025 -f submission.csv -m "kaggle-valid (improve_submission)"
```

## Kaggle: “ERROR” (overlap/touch)

O Kaggle permite encostar (touching), então `--overlap-mode kaggle` é equivalente a `strict` (touching permitido).  
Se você quiser uma validação mais conservadora (às custas de densidade/score), use `--overlap-mode conservative`.

Para “auto-fixar” um CSV existente, use:

```bash
python -m santa_packing.cli.autofix_submission submission.csv \
  --out submission_kaggle.csv --overlap-mode kaggle
python -m santa_packing.cli.score_submission submission_kaggle.csv --nmax 200 --overlap-mode kaggle --pretty
```

## Configuração (reprodutibilidade)

O workflow (`python -m santa_packing`) carrega config por padrão (quando existir):

* `configs/submit_strong.json` (default do workflow; preset “pesado”)
* `configs/submit.json` (preset mais leve; use via `--config`)
* `configs/ensemble.json` (default para `sweep_ensemble`)

Para sobrescrever, passe flags normalmente; para trocar/ignorar config: use `--config ...` ou `--no-config`.

## Documentação (por tema)

Detalhes extensos (roadmap, guia de L2O, instruções de ambiente) foram movidos para `docs/`:

* `docs/overview.md` — panorama da competição + estratégia
* `docs/roadmap.md` — roadmap de otimização (alto ROI)
* `docs/setup.md` — setup, estrutura do código e comandos rápidos
* `docs/sa.md` — SA (proposals, vizinhança, objetivos) e exemplos
* `docs/l2o.md` — L2O, meta-init e heatmap meta
* `docs/ensemble.md` — sweep/ensemble/repair

## Desenvolvimento

```bash
make test
make lint
make format
```
