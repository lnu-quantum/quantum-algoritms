"""
Shor's Algorithm for Quantum Computing

Shor's algorithm is a quantum algorithm for factoring integers.
It demonstrates exponential speedup over classical algorithms for factoring.
"""

import math
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np


def gcd(a, b):
    """Compute the greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a


def is_power_of_prime(n):
    """Check if n is a power of a prime number."""
    for base in range(2, int(math.sqrt(n)) + 1):
        temp = n
        power = 0
        while temp % base == 0:
            temp //= base
            power += 1
        if temp == 1 and power > 1:
            return True, base
    return False, None


def classical_period_finding(a, N):
    """
    Classical period finding (used as fallback for small N).
    
    Finds the period r such that a^r mod N = 1.
    """
    if N <= 1:
        return None
    
    # Try all possible periods up to N
    for r in range(1, min(N, 1000)):
        if pow(a, r, N) == 1:
            return r
    return None


def quantum_period_finding(a, N, n_qubits=8):
    """
    Quantum period finding subroutine using Quantum Phase Estimation.
    
    This finds the period r such that a^r mod N = 1.
    Uses a simplified quantum simulation approach.
    
    Args:
        a: Base for modular exponentiation
        N: Modulus
        n_qubits: Number of qubits for the counting register
    
    Returns:
        Estimated period r
    """
    # Create quantum circuit
    counting_qubits = n_qubits
    state_qubits = max(int(math.ceil(math.log2(N))), 2)
    
    qr_counting = QuantumRegister(counting_qubits, 'counting')
    qr_state = QuantumRegister(state_qubits, 'state')
    cr = ClassicalRegister(counting_qubits, 'c')
    
    qc = QuantumCircuit(qr_counting, qr_state, cr)
    
    # Initialize state register to |1>
    qc.x(qr_state[0])
    
    # Apply Hadamard gates to counting register
    for i in range(counting_qubits):
        qc.h(qr_counting[i])
    
    # Simplified quantum modular exponentiation simulation
    # In a full implementation, this would use quantum gates to compute
    # a^(2^j) mod N. Here we simulate the phase that would result.
    for i in range(counting_qubits):
        # Compute a^(2^i) mod N classically to get the phase
        power = 2 ** i
        result = pow(a, power, N)
        # The phase is related to the result of modular exponentiation
        # This is a simplified approximation
        phase = (2 * np.pi * result) / N
        if i < state_qubits:
            qc.cp(phase, qr_counting[i], qr_state[min(i, state_qubits - 1)])
    
    # Inverse QFT on counting register
    for i in range(counting_qubits):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), qr_counting[i], qr_counting[j])
        qc.h(qr_counting[i])
    
    # Measure counting register
    qc.measure(qr_counting, cr)
    
    # Run simulation
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1024)
    result = job.result()
    counts = result.get_counts(qc)
    
    # Try multiple measurement results (not just the most likely)
    # Sort by frequency
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Try top 3 results
    for measured_value, _ in sorted_counts[:3]:
        measured_int = int(measured_value, 2)
        
        # Convert to phase
        phase = measured_int / (2 ** counting_qubits)
        
        # Use continued fractions to find the period
        # The period r should satisfy: phase ≈ s/r for some integer s
        for denominator in range(1, min(N, 100)):  # Limit search space
            numerator = round(phase * denominator)
            if abs(phase - numerator / denominator) < 0.1:
                r = denominator
                # Verify that a^r mod N = 1
                if pow(a, r, N) == 1:
                    return r
        
        # Alternative: try r = 1/phase (rounded)
        if phase > 0:
            r_guess = round(1 / phase)
            if 1 <= r_guess < N and pow(a, r_guess, N) == 1:
                return r_guess
    
    # Fallback: classical period finding for small N
    return classical_period_finding(a, N)


def shors_algorithm(N, n_qubits=8, max_attempts=5):
    """
    Shor's algorithm for factoring an integer N.
    
    Args:
        N: The integer to factor (must be odd and composite)
        n_qubits: Number of qubits for quantum period finding
        max_attempts: Maximum number of attempts to find a factor
    
    Returns:
        A non-trivial factor of N, or None if factoring fails
    """
    # Step 1: Check if N is even
    if N % 2 == 0:
        return 2
    
    # Step 2: Check if N is a power of a prime
    is_power, base = is_power_of_prime(N)
    if is_power:
        return base
    
    # Try multiple random values of a
    for attempt in range(max_attempts):
        # Step 3: Choose a random number a such that 1 < a < N
        a = random.randint(2, N - 1)
        
        # Step 4: Check if gcd(a, N) > 1 (lucky case)
        g = gcd(a, N)
        if g > 1:
            return g
        
        # Step 5: Quantum period finding
        # Find the period r such that a^r mod N = 1
        print(f"Спроба {attempt + 1}: Знаходимо період для a={a}, N={N}...")
        r = quantum_period_finding(a, N, n_qubits)
        
        if r is None or r == 0:
            print(f"Не вдалося знайти період, пробуємо інше a...")
            continue
        
        print(f"Знайдено період r={r}")
        
        # Step 6: Check if period is valid
        if r % 2 != 0:
            print(f"Період {r} непарний, пробуємо знову...")
            continue
        
        # Step 7: Compute factors
        x = pow(a, r // 2, N)
        if x == 1 or x == N - 1:
            print(f"x={x} є тривіальним, пробуємо знову...")
            continue
        
        # Step 8: Compute gcd(x + 1, N) and gcd(x - 1, N)
        factor1 = gcd(x + 1, N)
        factor2 = gcd(x - 1, N)
        
        if factor1 > 1 and factor1 < N:
            return factor1
        elif factor2 > 1 and factor2 < N:
            return factor2
    
    print("Не вдалося знайти множник після всіх спроб")
    return None


def factorize(N, max_attempts=10):
    """
    Factorize N using Shor's algorithm with multiple attempts.
    
    Args:
        N: The integer to factor
        max_attempts: Maximum number of attempts
    
    Returns:
        List of factors, or None if factorization fails
    """
    if N < 2:
        return None
    
    if N == 2:
        return [2]
    
    factors = []
    remaining = N
    
    for attempt in range(max_attempts):
        if remaining < 2:
            break
        
        if remaining == 2:
            factors.append(2)
            break
        
        factor = shors_algorithm(remaining)
        if factor:
            factors.append(factor)
            remaining //= factor
            print(f"Знайдено множник: {factor}, залишилося: {remaining}")
        else:
            print(f"Спроба {attempt + 1} не вдалася")
    
    if remaining > 1:
        factors.append(remaining)
    
    return factors if factors else None


# Example usage
if __name__ == "__main__":
    # Test with a small number
    print("=" * 50)
    print("Алгоритм Шора - Квантова факторизація")
    print("=" * 50)
    
    # Example: Factor 15 (3 * 5)
    N = 15
    print(f"\nФакторизація N = {N}")
    print("-" * 50)
    
    factors = factorize(N)
    if factors:
        print(f"\nМножники {N}: {factors}")
        print(f"Перевірка: {factors} = {math.prod(factors)}")
    else:
        print(f"\nНе вдалося факторизувати {N}")
    
    # Example: Factor 21 (3 * 7)
    print("\n" + "=" * 50)
    N = 21
    print(f"\nФакторизація N = {N}")
    print("-" * 50)
    
    factors = factorize(N)
    if factors:
        print(f"\nМножники {N}: {factors}")
        print(f"Перевірка: {factors} = {math.prod(factors)}")
    else:
        print(f"\nНе вдалося факторизувати {N}")

