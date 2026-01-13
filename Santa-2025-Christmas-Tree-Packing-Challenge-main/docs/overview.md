# Santa 2025 — Visão geral

Este projeto resolve o *Santa 2025 (Christmas Tree Packing)* como um problema de **otimização geométrica 2D**: você constrói uma solução de *packing* (posições + rotações) que minimiza a métrica oficial.

Veja também:
* `docs/roadmap.md` — roadmap de otimização (alto ROI)
* `docs/setup.md` — como configurar e rodar

## Resumo da competição

* **Tipo de problema:** otimização geométrica 2D (*packing*). Não é “treinar modelo”; é construir uma solução geométrica. ([AI Competition Hub][1])
* **Instâncias:** você precisa entregar uma configuração para **n = 1..200** árvores. ([AI Competition Hub][1])
* **Submission:** para cada `id` (ex.: `002_1`) você entrega **posição `(x,y)` e rotação `deg`**; os valores são **strings com prefixo `s`** (ex.: `s0.0`). ([AI Competition Hub][1])
* **Restrições importantes:**
  * **Sem sobreposição** (*overlap*). ([AI Competition Hub][1])
  * `-100 <= x,y <= 100`. ([AI Competition Hub][1])
* **Métrica (minimização):**
  \[
  \text{score} = \sum_{n=1}^{200}\frac{s_n^2}{n}
  \]
  onde `s_n` é o lado do **quadrado** que “encaixota” a configuração daquele `n`. Em notebooks públicos, isso é implementado via **bounding box axis-aligned**: pega `min/max` de todos os vértices e usa `s_n = max(width, height)`. ([Kaggle][2])

## O que “dá score alto” na prática

Você sobe score com 3 coisas (ordem de importância):

1. **Geometria correta + checagem de colisão robusta**  
   Se a checagem falha (falso negativo), você toma erro no submission. Se for conservadora demais, você reduz densidade e piora score.

2. **Boa configuração base (packing “estrutural”)**  
   Tipicamente: algum **tiling/lattice** (triangular/hex) + padrão de rotações (ex.: alternar rotações por linha/coluna) e depois ajuste fino.

3. **Meta-heurística para refino**  
   O espaço de busca é contínuo e grande (≈ `3n` variáveis por instância: `x,y,deg`). Abordagens comuns: **Simulated Annealing (SA)**, hill-climb, mutações estilo GA, etc.

## Checklist inicial (para não travar)

1. **Reproduzir a métrica localmente** (igual à do Kaggle) e ter um `score(submission)` rápido. ([AI Competition Hub][1])
2. **Implementar a forma do “tree” como polígono** + transformações rígidas (rotação + translação). Em notebooks públicos aparece como polígono com **15 vértices**. ([Kaggle][4])
3. **Implementar colisão eficiente**:
   * filtro grosso: **círculo envolvente**
   * filtro fino: **interseção de polígonos** (SAT se convexo; se côncavo, triangule/decomponha).
4. **Ter um baseline determinístico** que sempre gera solução válida (mesmo que ruim). Só depois colocar SA/GA.

## Snippet mínimo (formato de submission)

Formato exigido (com `s` prefixado) ([AI Competition Hub][1]):

```python
import pandas as pd


def fmt(x: float) -> str:
    return f"s{x:.9f}"


rows = []
for n in range(1, 201):
    for i in range(n):
        x, y, deg = 0.0, 0.0, 0.0
        rows.append({"id": f"{n:03d}_{i}", "x": fmt(x), "y": fmt(y), "deg": fmt(deg)})

pd.DataFrame(rows).to_csv("submission.csv", index=False)
```

## Se você quer top 5%: plano curto e eficaz

1. Pegar um baseline público (tiling/greedy) e **rodar score local** (garantir que você consegue submeter sem erro).
2. Rodar **SA com vizinhança** + multi-start para `N=200` (veja `docs/sa.md`).
3. Definir uma boa **ordem de inclusão** (centro → borda) e gerar todos `n` por recorte (*mother-prefix*).
4. Rodar várias seeds e fazer **seleção por `n`** (ensemble por instância). ([Kaggle][6])

[1]: https://www.competehub.dev/en/competitions/kagglesanta-2025 "Santa 2025 - Christmas Tree Packing Challenge - CompeteHub"
[2]: https://www.kaggle.com/code/jekiwantaufik/santa-2025-christmas-tree-packing-challenge?utm_source=chatgpt.com "Santa 2025 - Christmas Tree Packing Challenge"
[4]: https://www.kaggle.com/code/jekiwantaufik/santa-2025-christmas-tree-packing-challenge-2?utm_source=chatgpt.com "Santa 2025 - Christmas Tree Packing Challenge 2"
[6]: https://www.kaggle.com/code/muhammadibrahim3093/ensembling-santa-2025?utm_source=chatgpt.com "Ensembling_Santa_2025"
