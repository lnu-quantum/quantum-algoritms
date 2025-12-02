# Quantum Circuits
test

This repository contains Python implementations of quantum algorithms, including Deutsch's Algorithm, the Deutsch-Jozsa Algorithm, and Grover's Algorithm. These algorithms are simulated on a classical computer using the QuTiP library. The code is heavily commented and intended for educational purposes. For Grover's Algorithm, multiple methods are provided for certain objectives, which are flagged in the code.

## Requirements

To run the scripts, you need the following:

- Python 3.8 <= version <= 3.13
- QuTiP (Quantum Toolbox in Python)
- NumPy
- Matplotlib

You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Deutsch's Algorithm

Deutsch's Algorithm is a deterministic quantum algorithm devised by David Deutsch in 1985. It demonstrates that a quantum computer can solve a specific problem in fewer steps than a classical computer. The problem is to determine whether a Boolean function `f` is constant (e.g., `f(0)=f(1)`) or balanced (e.g., `f(0)≠f(1)`). A classical computer requires two queries to solve this, while Deutsch's Algorithm requires only one.

### Quantum Circuit

```
        +----+
|0> H --| Uf |-- H ------- M
|1> H --|    |--------------
        +----+
```

### Usage

Run the script from the command line:

```bash
python Deutsch.py
```

By default, the script runs for a constant function `f(0)=1, f(1)=1` and determines that `f` is constant.

## The Deutsch-Jozsa Algorithm

The Deutsch-Jozsa Algorithm, a generalization of Deutsch's Algorithm, was devised by David Deutsch and Richard Jozsa in 1992. It determines whether a Boolean function is constant or balanced. The algorithm is deterministic and significantly faster than its classical counterpart for large inputs.

### Quantum Circuit

For two qubits plus a control qubit:

```
|0> H --+----+-- H ------- M
|0> H --| Uf |-- H ------- M
|1> H --+----+--------------
```

### Usage

Run the script from the command line:

```bash
python Deutsch-Jozsa.py
```

By default, the script runs for a balanced function and determines that `f` is balanced.

## Grover's Algorithm

Grover's Algorithm is a quantum search algorithm that finds a specific item in an unsorted database with quadratic speedup compared to classical algorithms. It was devised by Lov Grover in 1996.

### Quantum Circuit

The circuit involves repeated application of the Grover iteration, which consists of:

1. Oracle marking the solution.
2. Diffusion operator amplifying the probability of the solution.

### Usage

Run the script from the command line:

```bash
python Grover.py
```

The script demonstrates Grover's Algorithm for a small database. You can modify the input parameters to experiment with different database sizes and target items.

## References

For more information on these algorithms, refer to:

- Nielsen, Michael A., and Isaac L. Chuang. *Quantum Computation and Quantum Information*. 10th Anniversary Edition. Cambridge: Cambridge University Press (2010).
  - Section 1.4.3 for Deutsch's Algorithm
  - Section 1.4.4 for the Deutsch-Jozsa Algorithm
  - Section 6.2 for Grover's Algorithm

