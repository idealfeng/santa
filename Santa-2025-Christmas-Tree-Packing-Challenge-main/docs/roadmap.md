# Roadmap de otimização (alto ROI)

Este roadmap lista as ideias que normalmente dão mais retorno por unidade de tempo para melhorar o score.

Veja também:
* `docs/sa.md` — SA (proposals/vizinhança) e exemplos
* `docs/ensemble.md` — multi-start + seleção por `n`

## 1) Baseline que “não erra”

* Coloque centros em **grade** ou **lattice triangular/hex** (densidade melhor que grade).
* Use uma rotação fixa (ex.: `deg=0`) inicialmente.
* Gere solução para todos `n` pegando os **n primeiros pontos** de uma lista ordenada por “proximidade do centro” (isso ajuda os `n` pequenos a terem caixa menor). ([Kaggle][1])

## 2) Insight que geralmente vale muito: “solução-mãe + recorte”

Em vez de resolver 200 problemas independentes:

* Otimize bem um packing para **N=200**.
* Defina uma **ordem de inclusão** das árvores (camadas do centro para fora).
* Para cada `n`, use o prefixo das `n` primeiras árvores.

Isso costuma dar um salto grande porque você gasta *compute* pesado em **uma** configuração e “reaproveita” para todas.

## 3) SA / Hill-climb em cima da base

Movimentos típicos:

* escolher árvore `i`
* propor `dx, dy, dθ` pequenos
* aceitar se:
  * não colide
  * melhora `s_n` (ou melhora um custo suavizado)
  * ou aceita por temperatura (SA)

Neste repo o SA suporta “vizinhança” (movimentos estruturados):
* **swap**: permuta duas árvores (útil quando `objective=prefix`/mother-prefix)
* **teleport**: move uma árvore da borda para um “pocket” perto de uma âncora no interior
* **compact/push**: passos *greedy* que empurram árvores em direção ao centro

Sugestão: veja `docs/sa.md` e os flags de `santa_packing/main.py` e `santa_packing/cli/generate_submission.py`.

## 4) Ensemble = seleção por instância (não média)

Como o score é soma por `n`, dá para fazer um ensemble forte:

* Rodar 5–20 variações do solver (seeds, parâmetros, padrões de lattice, etc.)
* Para cada `n`, **pegar a configuração com menor `s_n`** dentre as candidatas
* Montar um submission final misturando “o melhor de cada”

Veja `docs/ensemble.md`. ([Kaggle][2])

## 5) Micro-otimizações que decidem leaderboard

* **Spatial hashing / grid de vizinhança**: checar colisão só com árvores próximas (corta `O(n^2)`).
* **Numba/C++**: collision check costuma ser gargalo; 15 vértices favorece otimização agressiva. ([Kaggle][3])
* **Precisão numérica/margens**: você quer encostar sem “quase-colidir” (evitar erro na avaliação). ([AI Competition Hub][4])

[1]: https://www.kaggle.com/code/koushikkumardinda/santa-2025-christmas-tree-packing-challenge?utm_source=chatgpt.com "Santa 2025 - Christmas Tree Packing Challenge"
[2]: https://www.kaggle.com/code/muhammadibrahim3093/ensembling-santa-2025?utm_source=chatgpt.com "Ensembling_Santa_2025"
[3]: https://www.kaggle.com/code/jekiwantaufik/santa-2025-christmas-tree-packing-challenge-2?utm_source=chatgpt.com "Santa 2025 - Christmas Tree Packing Challenge 2"
[4]: https://www.competehub.dev/en/competitions/kagglesanta-2025 "Santa 2025 - Christmas Tree Packing Challenge - CompeteHub"
