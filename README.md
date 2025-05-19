<!--This file is formatted in Markdown notation.-->
# Streptomyces growth model
The Streptomyces growth model simulates the hyphal growth patterns that are formed by the bacteria Streptomyces. Events are chosen based on the Gillespie algorithm and happen stochastically.

## Table of Contents  
- [Instalation](#instalation-id)
- [Simulation Initialization](#simulation-initialization-id)
    - [Elements](#elements-id)  
    - [Reactions](#reactions-id)
    

## Instalation <a id="instalation-id"></a>
[Context here]

## Simulation Initialization <a id="simulation-initialization-id"></a>
Initialize all the objects and parameters that are used in the simulation.
Elements and reactions are used for the chemical interactions within the simulation.


### Elements <a id="elements-id"></a>
Elements form the basis for the chemical activity within the system.
For each unique element you need to define the initial amount.

```python
Element(name: str, symbol: str (optional), initial_amount: int (default is 0))
```

### Reactions <a id="reactions-id"></a>
Reactions define the interactions of the elements.
Define which elements are used as a reactant that form the product.
Each reaction has a rate, this directly influences the propensity and therefore the activity in the system.

```python
Reaction(name: str, rate: float, reactants: dict[Element, int], products: dict[Element, int])
```

## Extra thanks
A special thanks to my suppervisor dr. Roeland Merks. 
Without his guidance and advanced expertise in moddeling this project would have never existed.
Also a special thanks to Luis and Danny who helped on the biological relevance of the project. 
