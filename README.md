# Problem sumy podzbioru ***(Subset Sum Problem)***

## Opis Problemu:

Problem sumy podzbioru należy do klasy problemów NP-zupełnych. Polega on na sprawdzeniu, czy dla danej listy liczb całkowitych istnieje taki podzbiór tych liczb, którego suma jest równa zadanej wartości docelowej. 

W kontekście metaheurystyki będziemy rozważać wersję optymalizacyjną mającą na celu znaleźć podzbiór `S'`, którego suma `sum(S')` jest jak najbliższa docelowej sumie `T`

Funkcja celu do minimalizacji to `|sum(S') - T|`.
Optymalne rozwiązanie ma wartość funkcji celu równą 0.


## Definicja formalna

Dla danego zbioru liczb całkowitych \( S = \{s_1, s_2, ..., s_n\} \) oraz liczby całkowitej \( T \), celem jest ustalenie, czy istnieje podzbiór \( S' \subseteq S \), taki że suma elementów w tym podzbiorze jest równa \( T \), czyli:

\[
\sum_{x \in S'} x = T
\]
