import math
import random
from fractions import Fraction

import numpy as np
import sympy

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit_aer import AerSimulator

# Будую унітарну матрицю U, що діє на n_bits кубітів:
# U |y> = | (a^(2^power_index) * y) mod N > для y in [0..N-1]
# Станам y >= N залишаю як і є (тотожне відображення).

# Спрацює тільки для невеликих N (бо розмір матриці = 2^n_bits).

def build_modular_mult_unitary(a: int, power_index: int, N: int, n_bits: int) -> np.ndarray:
    dim = 1 << n_bits
    U = np.zeros((dim, dim), dtype=complex)
    factor = pow(a, 1 << power_index, N)
    coprime = math.gcd(a, N) == 1

    for y in range(dim):
        if coprime and y < N:
            y2 = (factor * y) % N
        else:
            y2 = y
        U[y2, y] = 1.0
    return U

# Повертаю QuantumCircuit для QPE:
# count register: n_count кубітів;
# work register: n_work кубітів (для чисел mod N).

# На робочому регістрі початково ставлю |1>.

def make_phase_estimation_circuit(a: int, N: int, n_count: int = None):
    n_work = max(1, N.bit_length())
    if n_count is None:
        n_count = 2 * n_work

    qr_count = QuantumRegister(n_count, "count")
    qr_work  = QuantumRegister(n_work, "work")
    cr_count = ClassicalRegister(n_count, "c")
    qc = QuantumCircuit(qr_count, qr_work, cr_count, name=f"QPE_a{a}_N{N}")

    # Hadamard на лічильний регістр.
    qc.h(qr_count)

    qc.x(qr_work[0])

    qc.barrier()

    # Контрольовані блоки U^(2^k).
    for k in range(n_count):
      U = build_modular_mult_unitary(a, k, N, n_work)
      U_gate = UnitaryGate(U, label=f"U_{a}^{1<<k}")
      ctrl = U_gate.control(1)
      qc.append(ctrl, [qr_count[k]] + [qr_work[i] for i in range(n_work)])


    qc.barrier()

    # Інверсний QFT на лічильному регістрі.
    qc.append(QFT(n_count, inverse=True, do_swaps=True), qr_count)

    qc.measure(qr_count, cr_count)

    return qc
  
# Перетворюю двійковий рядок в дробове значення.

def extract_r_from_counts(measured_bin: str, n_count: int, N: int) -> int:
    value = int(measured_bin, 2)
    denom = 1 << n_count
    frac = Fraction(value, denom).limit_denominator(N)
    return frac.denominator

class Shor:
    def __init__(self, N: int, simulator=None, n_count: int = None, max_attempts: int = -1):
        self.N = N
        self.simulator = simulator or AerSimulator()
        self.n_work = max(1, N.bit_length())
        self.n_count = n_count or (2 * self.n_work)
        self.last_qpe = None
        self.last_measured = None
        self.max_attempts = max_attempts

    def _prechecks(self):
        if self.N <= 3:
            return (1, self.N)
        if self.N % 2 == 0:
            return (2, self.N // 2)
        if sympy.isprime(self.N):
            return (1, self.N)
        # Перевіряю на степінь.
        max_e = int(math.log2(self.N))
        for e in range(max_e, 1, -1):
            root = round(self.N ** (1 / e))
            if root ** e == self.N:
                return (root, e)
        return None

    def _quantum_period_finding(self, a: int):
        # Збираю схему і запускаю на симуляторі.
        qpe = make_phase_estimation_circuit(a, self.N, n_count=self.n_count)
        self.last_qpe = qpe
        transpiled = transpile(qpe, self.simulator)
        tqc = transpile(qpe, self.simulator)
        shots = 1
        job = self.simulator.run(tqc, shots=shots, memory=True)
        res = job.result()
        mem = res.get_memory()
        measured = mem[0]
        self.last_measured = measured
        r_candidate = extract_r_from_counts(measured, self.n_count, self.N)
        return r_candidate, measured

    def _classical_postprocessing(self, a: int, r: int):
        if r % 2 == 1:
            return None
        x = pow(a, r // 2, self.N)
        if x == self.N - 1:
            return None
        p = math.gcd(x - 1, self.N)
        q = math.gcd(x + 1, self.N)
        if 1 < p < self.N and 1 < q < self.N:
            return (p, q)
        return None

    def factor(self):
        trivial = self._prechecks()
        if trivial:
            return trivial

        # Генерую кандидати a.
        candidates = [a for a in range(2, self.N) if math.gcd(a, self.N) == 1]
        random.shuffle(candidates)
        limit = len(candidates) if self.max_attempts == -1 else min(self.max_attempts, len(candidates))

        for i in range(limit):
            a = candidates[i]
            g = math.gcd(a, self.N)
            if g != 1:
                return (g, self.N // g)

            print(f"\n[Спроба {i+1}/{limit}] перевірка a = {a}")

            # Квантове знаходження порядку.
            r, measured = self._quantum_period_finding(a)
            print(f"Виміряний бітовий рядок: {measured} -> r_candidate = {r}")

            if r is None or r == 0:
                print("Некоректне r -> повторна спроба")
                continue

            res = self._classical_postprocessing(a, r)
            if res:
                print(f"[ЗНАЙДЕНО] дільники: {res[0]} * {res[1]}")
                return res
            else:
                print("Постобробка не вдалася -> спроба наступного a")

        print("Не вдалося знайти нетривіальні дільники")
        return None

# Перевіряю. Факторизую 15.
if __name__ == "__main__":
    N = 15
    shor = Shor(N, simulator=AerSimulator(), n_count=None, max_attempts=10)
    result = shor.factor()
    print("\nРезультат:", result)

    try:
        display(shor.last_qpe.draw(output='mpl', fold=-1))
        print(f"\nОстанні виміряні біти: {shor.last_measured}")
    except Exception as e:
        print("Візуалізація недоступна:", e)
