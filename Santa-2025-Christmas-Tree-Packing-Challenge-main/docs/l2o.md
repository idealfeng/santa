# L2O (Learning to Optimize) e treino

Ideia: substituir (ou guiar) movimentos aleatórios do SA por uma policy `f_θ` que, dado o estado atual, sugere deslocamentos `(dx, dy, dtheta)` e/ou a escolha da árvore a mover.

Implementação principal: `santa_packing/l2o.py` (policies MLP/GNN e losses).

## Treinar L2O (REINFORCE)

Treino simples (N fixo):

```bash
python -m santa_packing._tools.train_l2o --n 10 --train-steps 200 --out runs/l2o_policy.npz
```

Usar no gerador:

```bash
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200 \
  --l2o-model runs/l2o_policy.npz --l2o-nmax 10
```

Treinar com GNN (kNN simples):

```bash
python -m santa_packing._tools.train_l2o --n 10 --train-steps 200 --policy gnn --knn-k 4 --out runs/l2o_gnn_policy.npz
```

## Treino multi-N (dataset sintético)

```bash
python -m santa_packing._tools.train_l2o --n-list 25,50,100 --train-steps 200 \
  --init mix --dataset-size 128 --dataset-out runs/l2o_dataset.npz \
  --policy gnn --knn-k 4 --out runs/l2o_gnn_policy.npz
```

## Behavior cloning (imitar SA)

1) Coletar dataset de SA (estados + deslocamentos aceitos):

```bash
python -m santa_packing._tools.collect_sa_dataset --n-list 25,50,100 --runs-per-n 5 --steps 400 \
  --init mix --best-only --out runs/sa_bc_dataset.npz
```

2) Treinar para imitar deslocamentos aceitos:

```bash
python -m santa_packing._tools.train_l2o_bc --dataset runs/sa_bc_dataset.npz --policy gnn --knn-k 4 --train-steps 500 \
  --out runs/l2o_bc_policy.npz
```

## Meta-init (boa inicialização antes do SA)

Treinar:

```bash
python -m santa_packing._tools.train_meta_init --n-list 25,50,100 --train-steps 50 --es-pop 6 \
  --sa-steps 100 --sa-batch 16 --objective packing --out runs/meta_init.npz
```

Usar no gerador:

```bash
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200 \
  --sa-nmax 50 --sa-steps 400 --meta-init-model runs/meta_init.npz
```

## Heatmap meta (prioriza quais árvores mover)

Treinar:

```bash
python -m santa_packing._tools.train_heatmap_meta --n-list 10,20,30 --train-steps 50 --es-pop 6 \
  --heatmap-steps 200 --out runs/heatmap_meta.npz
```

Usar no gerador:

```bash
python -m santa_packing.cli.generate_submission --out submission.csv --nmax 200 \
  --heatmap-model runs/heatmap_meta.npz --heatmap-nmax 10 --heatmap-steps 200
```
