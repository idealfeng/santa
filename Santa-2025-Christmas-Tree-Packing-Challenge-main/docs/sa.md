# Simulated Annealing (SA)

Este repo oferece SA em JAX tanto como runner “direto” (`santa_packing/main.py`) quanto integrado ao gerador de `submission.csv` (`santa_packing/cli/generate_submission.py`).

## Runner direto (uma instância)

```bash
python -m santa_packing.main \
  --n_trees 25 --batch_size 128 --n_steps 1000 \
  --proposal mixed --neighborhood --objective packing
```

O runner gera `best_packing.png` no diretório atual.

## SA dentro do gerador de submission

Exemplo híbrido lattice + SA (para `n` até `--sa-nmax`):

```bash
python -m santa_packing.cli.generate_submission \
  --out submission.csv --nmax 200 \
  --sa-nmax 30 --sa-steps 400 --sa-batch 64 \
  --sa-proposal mixed --sa-neighborhood --sa-objective packing
```

## Vizinhança (movimentos estruturados)

Quando habilitada, a “vizinhança” adiciona movimentos que aceleram o refino:
* `swap`: útil quando o objetivo está alinhado ao *prefix* (ordem importa)
* `teleport`: borda → interior (busca por “pockets”)
* `compact/push`: empurra para o centro (reduz bbox rápido)

No runner direto, use `--neighborhood`. No gerador, use `--sa-neighborhood` e/ou `--refine-neighborhood` (quando aplicável).

## Objetivos

* `packing`: minimiza o lado do quadrado que encaixota um único `n`.
* `prefix`: otimiza uma solução “mãe” onde os prefixos `1..N` contam no custo (alinhado ao score oficial quando você reaproveita um packing ordenado).

Dica: para *mother-prefix*, combine `--sa-objective prefix` + probabilidade de `swap` não-zero.

## Notas de estabilidade/qualidade

* O SA inclui heurísticas como **push-to-center**, **adaptação de sigma** (alvo de aceitação) e **reheating** (quando estagna). Veja flags em `santa_packing/main.py`.
* Para score local, você pode usar `--no-overlap` apenas para estimativa rápida; para reproduzir a checagem do Kaggle localmente, valide com `--overlap-mode kaggle` (touching permitido). Se quiser ser mais conservador, use `--overlap-mode conservative`.
