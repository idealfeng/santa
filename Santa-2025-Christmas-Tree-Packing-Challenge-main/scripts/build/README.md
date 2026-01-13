# scripts/build

Build helpers for optional native components.

## fastcollide (opcional)

`fastcollide` é uma extensão C++ usada para acelerar o *scorer* local (predicado `polygons_intersect_strict`).

Opções:

* Via Makefile: `make build-fastcollide`
* Direto: `python scripts/build/build_fastcollide.py`

Esses scripts existem porque o build do extension não é um subcomando “puro” do pacote.

