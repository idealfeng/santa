# Sweep / ensemble / repair

Como o score oficial soma contribuições para `n=1..200`, um ensemble “por instância” (por `n`) costuma ser forte:

* rode múltiplas variações do solver (seeds/receitas)
* para cada `n`, escolha o candidato com menor `s_n`
* produza um `submission.csv` final combinando “o melhor de cada”

## Config padrão

`sweep_ensemble` carrega `configs/ensemble.json` por padrão (quando existir). Para trocar/ignorar: `--config ...` ou `--no-config`.

Sobre overlap-check:

* `--overlap-check selected` usa a checagem *strict* (não considera “touching” como overlap).
* Se você quiser ser super conservador (menos denso, mas mais “safe”), valide depois com `python -m santa_packing.cli.score_submission ... --overlap-mode conservative`.

## Sweep + ensemble por instância

Exemplo com receitas inline:

```bash
python -m santa_packing._tools.sweep_ensemble --nmax 200 --seeds 1,2,3 \
  --recipe hex:"--lattice-pattern hex --lattice-rotations 0,15,30" \
  --recipe square:"--lattice-pattern square --lattice-rotations 0,15,30" \
  --out submission_ensemble.csv
```

Exemplo usando um portfólio JSON:

```bash
python -m santa_packing._tools.sweep_ensemble --nmax 200 --seeds 1..3 --jobs 3 \
  --recipes-json scripts/submission/portfolios/mixed.json \
  --out submission_ensemble.csv
```

## Repair (pós-processamento)

O gerador (`python -m santa_packing.cli.generate_submission`) inclui mecanismos de reparo simples para `n` pequenos (hill-climb e GA com reparo de overlaps via `--hc-*` e `--ga-*`).
