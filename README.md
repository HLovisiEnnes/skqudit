# SKQudit: Solovay-Kitaev for qudits with LSH acceleration
SKQudit (pronounced s-q-dit) is a Python package implementing the Solovay–Kitaev algorithm for arbitrary-dimensional qudits. 

In a nutshell, the Solovay–Kitaev algorithm (SK) approximates any target matrix in $SU(d)$ using a finite universal gate set, up to any desired accuracy. Unlike qubit-focused implementations (for example, [Qiskit's](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.SolovayKitaev)), SKQudit works directly with gates in $SU(d)$ where the dimension $d$ does not need to be a power of two. This makes it suitable for experiments with general qudit compilation.

SKQudit combines classical SK recursion with two approximate-search techniques designed to make high-dimensional net lookup more practical:

- Locality-sensitive hashing (LSH) using the geometry of $SU(d)$;
- Meet-in-the-middle search for querying large nets of gates.

Together, these methods accelerate approximate gate lookup and make the algorithm more usable under memory constraints, including on a personal laptop.


## Why SKQudit?

Most implementations of Solovay–Kitaev focus on qubits. SKQudit was developed to experiment with synthesis for general qudits, where the geometry of $SU(d)$, memory usage, and approximate nearest-neighbor search become central implementation issues.

The package is intended for:

- experiments with qudit gate synthesis;
- testing Solovay–Kitaev variants in arbitrary dimension;
- studying tradeoffs between accuracy, circuit length, runtime, and memory;
- educational demonstrations of quantum compiling beyond the qubit case.

## Algorithmic Background

The main algorithm follows the classical SK construction described by [Dawson and Nielsen](https://arxiv.org/abs/quant-ph/0505030). The implementation then modifies the net lookup step using two acceleration strategies.

### Locality-Sensitive Hashing

Instead of performing a naive linear search over the entire net, SKQudit uses probabilistic locality-sensitive hashing. The hash functions are designed using geometric information from the group $SU(d)$, allowing candidate gates close to the target to be retrieved more efficiently.

### Meet-in-the-Middle Search

SKQudit also supports a meet-in-the-middle strategy for querying nets. Rather than storing and searching only complete products of gates, the algorithm decomposes the search into smaller components and combines them during lookup. This can reduce memory requirements and make deeper nets more practical.

## Relation to Prior Work

After releasing the initial version of this package, I became aware of the paper ["Optimization of the Solovay–Kitaev Algorithm"](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.052332) by Pham, Van Meter, and Horsman. Their work uses a related search-space expansion idea and GNAT-based nearest-neighbor search to accelerate Solovay–Kitaev.

SKQudit differs in implementation choices and scope: it is designed for arbitrary qudit dimensions, while their implementation focuses on the qubit case.

# Installation
You can install the project with **pip**.

```bash
# Clone the repository
git clone https://github.com/HLovisiEnnes/skqudit/
cd project

# Option 1: Install in editable mode (recommended for development)
pip install -e .

# Option 2: Install normally
pip install .

# Verify installation
python -m project --help
```

# Example of use
Here is an example of use of the package. For more examples, see the [tutorial notebook](notebooks/tutorial.ipynb).

```python
from skqudit import (
    InstructionSet,
    SimpleNet,
    solovay_kitaev
)
from skqudit.utils import su_matrix, gell_mann_su3

# Example in dimension 3
d = 3

# Use Gell-Mann matrices as universal gates
s1, s2 = gell_mann_su3()

# Fix the universal gate set
instr = { 
    's1': s1,
    's2': s2
}
instr_set = InstructionSet(instr)

# Build a net up to some layer depth
layers = 12
net = SimpleNet(instr_set)
net.build_net(layers)

# Generate random SU(d) gate
gate_to_approx = su_matrix(d)

# Run SK and plot history of errors and number of gates
depth = 6
best_gate, hist = solovay_kitaev(
    gate_to_approx, 
    depth, 
    net, 
    epsilon_0=0.1,  
    scale=2,
    method='meet_in_the_middle',
    bucket_params = {'k': 1, 'bucket_size': 0.1,  'bucket_robustness': 0}
    return_history=True,
    verbosity=3
)
plot_history(hist)
```

# To come
Here is a list of features that I plan to add at some point in a nearby future.
- Local nets for more stable look-ups close to the identity.
- Full vectorization of the net look-up subroutine.
- [Kuperberg's version of the algorithm](https://arxiv.org/abs/2306.13158), which should imply in substantial asymptotic speed ups.

Suggestions, questions, and collaborations are welcomed and appreciated :)

# Citation
To cite this project, please use the bib entry below.

```bibtex
@software{skqudit,
  author = {Henrique Ennes},
  title = {SKQudit: Solovay-Kitaev for qudits with LSH acceleration},
  year = {2026},
  url = {https://github.com/HLovisiEnnes/skqudit},
  version = {0.0.1}
}
```

# Acknowledgement
Special thanks to [Chih-Kang Huang](https://github.com/chih-kang-huang) for testing the package.
