# SKQudit: Solovay-Kitaev for qudits with LSH acceleration
SKQudit (pronounced *s-q-dit*) is a Python implementation of the Solovay-Kitaev algorithm for qudits. SK's approximates any matrix in $SU(d)$ with a universal gate set within any desired level of accuracy. Unlike other implementations of SK (for example, [Qiksit's](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.SolovayKitaev)), the gate's dimension $d$ is not restricted to a power of 2, allows its applications to the more general *qudits*.

The main algorithm is based on the description by [Dawrson and Nielsen](https://arxiv.org/abs/quant-ph/0505030), but also uses two additional techniques that significantly speed up computations:

- The geometry of the group $SU(d)$ for a probabilistic [*Locality Sensitive Hashing*](https://en.wikipedia.org/wiki/Locality-sensitive_hashing).
- A [*meet in the middle*](https://www.geeksforgeeks.org/dsa/meet-in-the-middle/) approach for querying nets of gates. Combined with the LSH, not only this makes the algorithm faster, but also allows it to be executed even when RAM is a constraint (e.g., on a personal laptop).

Refer to the [tutorial notebook](notebooks/tutorial.ipynb) for extra details on the SK algorithm and our implementation.

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
