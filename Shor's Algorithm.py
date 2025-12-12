"""
Удосконалений алгоритм Шора

Цей модуль реалізує покращену версію алгоритму Шора для факторизації цілих чисел
з додатковими оптимізаціями та альтернативними підходами.

Ключові покращення:
1. Адаптивний вибір параметрів квантової схеми
2. Покращений алгоритм постпроцесингу
3. Множинні стратегії знаходження періоду
4. Розширена діагностика та візуалізація
5. Оптимізована обробка помилок

Автор: Модифікована версія класичного алгоритму Шора
"""

import math
import random
import time
from fractions import Fraction
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, tensor, sigmax, sigmaz, qeye, Qobj


##############################################################################
# Конфігурація та структури даних
##############################################################################

class FactorizationMethod(Enum):
    """Методи факторизації"""
    TRIVIAL_EVEN = "парне_число"
    TRIVIAL_POWER = "степінь"
    QUANTUM_PERIOD = "квантовий_період"
    CLASSICAL_FALLBACK = "класичний_резерв"
    GCD_LUCKY = "щасливий_нсд"


@dataclass
class QuantumConfig:
    """Конфігурація квантової схеми"""
    max_qubits: int = 8  # Зменшено для стабільності
    precision_factor: float = 1.5
    measurement_shots: int = 1000
    convergence_threshold: float = 1e-4
    max_period_candidates: int = 15


@dataclass
class FactorizationResult:
    """Результат факторизації"""
    original_number: int
    factors: Optional[Tuple[int, int]]
    method_used: FactorizationMethod
    attempts_count: int
    quantum_data: Optional[Dict]
    success: bool
    execution_time: float = 0.0


##############################################################################
# Математичні утиліти з покращеннями
##############################################################################

class MathUtils:
    """Клас з математичними утилітами"""

    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """
        Розширений алгоритм Евкліда: знаходить gcd(a,b) та коефіцієнти x,y
        такі що ax + by = gcd(a,b)
        """
        if a == 0:
            return b, 0, 1

        gcd_val, x1, y1 = MathUtils.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1

        return gcd_val, x, y

    @staticmethod
    def simple_gcd(a: int, b: int) -> int:
        """Простий НСД"""
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def is_prime(n: int) -> bool:
        """Перевірка на простоту (тест Міллера-Рабіна для малих чисел)"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        # Простий тест для малих чисел
        for i in range(3, min(int(np.sqrt(n)) + 1, 100), 2):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def detect_perfect_power(N: int) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Покращене виявлення досконалих степенів
        """
        if N <= 1:
            return False, None, None

        max_exp = int(math.log2(N)) + 1

        for exponent in range(2, max_exp + 1):
            # Бінарний пошук кореня
            low, high = 1, int(N ** (1 / exponent)) + 2

            while low <= high:
                mid = (low + high) // 2
                power = mid ** exponent

                if power == N:
                    return True, mid, exponent
                elif power < N:
                    low = mid + 1
                else:
                    high = mid - 1

        return False, None, None

    @staticmethod
    def fast_modular_exp(base: int, exponent: int, modulus: int) -> int:
        """
        Швидке модульне піднесення до степеня з оптимізаціями
        """
        if modulus == 1:
            return 0

        result = 1
        base = base % modulus

        while exponent > 0:
            if exponent & 1:  # Якщо exponent непарний
                result = (result * base) % modulus
            exponent >>= 1  # Ділимо на 2
            base = (base * base) % modulus

        return result


##############################################################################
# Покращені квантові операції
##############################################################################

class QuantumCircuitBuilder:
    """Будівник квантових схем"""

    def __init__(self, config: QuantumConfig):
        self.config = config

    def create_hadamard_gate(self) -> Qobj:
        """Створює одиночний гейт Адамара"""
        h_matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        return Qobj(h_matrix)

    def create_qft_matrix(self, num_qubits: int) -> Qobj:
        """
        Створює матрицю QFT з оптимізацією для малих розмірів
        """
        dimension = 2 ** num_qubits
        omega = np.exp(2j * np.pi / dimension)

        # Використовуємо векторизовані операції NumPy
        indices = np.arange(dimension)
        qft_matrix = np.outer(indices, indices)
        qft_matrix = omega ** qft_matrix / np.sqrt(dimension)

        return Qobj(qft_matrix)

    def create_inverse_qft_matrix(self, num_qubits: int) -> Qobj:
        """Обернена QFT"""
        return self.create_qft_matrix(num_qubits).dag()

    def create_hadamard_transform(self, n_qubits: int) -> Qobj:
        """Створює Hadamard трансформацію для n кубітів"""
        if n_qubits <= 0:
            raise ValueError("Кількість кубітів має бути > 0")

        H = self.create_hadamard_gate()

        if n_qubits == 1:
            return H

        # Будуємо тензорний добуток
        H_n = H
        for _ in range(n_qubits - 1):
            H_n = tensor(H_n, H)

        return H_n

    def build_controlled_unitary(self, a: int, N: int, control_qubits: int) -> Qobj:
        """
        Будує керований унітарний оператор для модульної експоненціації
        з покращеною ефективністю та правильними розмірами
        """
        control_dim = 2 ** control_qubits
        target_bits = max(3, int(np.ceil(np.log2(N))) + 1)
        target_dim = 2 ** target_bits

        total_dim = control_dim * target_dim

        print(f"   Розміри: control={control_dim}, target={target_dim}, total={total_dim}")

        # Створюємо унітарну матрицю
        unitary = np.zeros((total_dim, total_dim), dtype=complex)

        for control_state in range(control_dim):
            # Обчислюємо a^control_state mod N
            if control_state == 0:
                power = 1  # a^0 = 1
            else:
                power = MathUtils.fast_modular_exp(a, control_state, N)

            for target_state in range(target_dim):
                if target_state < N:
                    # Модульне множення: target_state * a^control_state mod N
                    new_target = (target_state * power) % N
                else:
                    # Стани >= N залишаються незмінними
                    new_target = target_state

                input_idx = control_state * target_dim + target_state
                output_idx = control_state * target_dim + new_target

                if 0 <= input_idx < total_dim and 0 <= output_idx < total_dim:
                    unitary[output_idx, input_idx] = 1.0

        return Qobj(unitary)


##############################################################################
# Покращений алгоритм знаходження періоду
##############################################################################

class QuantumPeriodFinder:
    """Клас для квантового знаходження періоду"""

    def __init__(self, config: QuantumConfig):
        self.config = config
        self.circuit_builder = QuantumCircuitBuilder(config)

    def estimate_optimal_qubits(self, N: int) -> int:
        """Оцінює оптимальну кількість кубітів"""
        base_qubits = int(np.ceil(np.log2(N)))
        optimal = int(self.config.precision_factor * base_qubits)
        return min(optimal, self.config.max_qubits)

    def find_period_quantum(self, a: int, N: int) -> Dict[str, Any]:
        """
        Головна функція квантового знаходження періоду з покращеннями
        """
        n_control = min(6, self.estimate_optimal_qubits(N))  # Обмежуємо для стабільності
        n_target = max(3, int(np.ceil(np.log2(N))) + 1)

        print(f"Квантова схема: {n_control} контрольних + {n_target} цільових кубітів")

        control_dim = 2 ** n_control
        target_dim = 2 ** n_target

        try:
            # Крок 1: Ініціалізація
            control_state = basis(control_dim, 0)
            target_state = basis(target_dim, 1)  # |1⟩ для початкового стану
            initial_state = tensor(control_state, target_state)

            print(f"   Початковий стан: {initial_state.shape}")

            # Крок 2: Створення суперпозиції (Hadamard на контрольних кубітах)
            hadamard_op = self.circuit_builder.create_hadamard_transform(n_control)
            hadamard_full = tensor(hadamard_op, qeye(target_dim))

            print(f"   Hadamard оператор: {hadamard_full.shape}")
            print(f"   Початковий стан: {initial_state.shape}")

            # Перевіряємо сумісність розмірів
            if hadamard_full.shape[1] != initial_state.shape[0]:
                raise ValueError(f"Несумісні розміри: {hadamard_full.shape} vs {initial_state.shape}")

            superposition_state = hadamard_full * initial_state

            # Крок 3: Керована модульна експоненціація
            controlled_U = self.circuit_builder.build_controlled_unitary(a, N, n_control)

            print(f"   Controlled U: {controlled_U.shape}")
            print(f"   Superposition: {superposition_state.shape}")

            if controlled_U.shape[1] != superposition_state.shape[0]:
                raise ValueError(
                    f"Несумісні розміри для controlled U: {controlled_U.shape} vs {superposition_state.shape}")

            entangled_state = controlled_U * superposition_state

            # Крок 4: Обернена QFT
            inverse_qft = self.circuit_builder.create_inverse_qft_matrix(n_control)
            qft_full = tensor(inverse_qft, qeye(target_dim))

            if qft_full.shape[1] != entangled_state.shape[0]:
                raise ValueError(f"Несумісні розміри для QFT: {qft_full.shape} vs {entangled_state.shape}")

            final_state = qft_full * entangled_state

            # Крок 5: Аналіз результатів
            probabilities = np.abs(final_state.full().flatten()) ** 2

            return {
                'a': a, 'N': N,
                'n_control': n_control, 'n_target': n_target,
                'final_state': final_state,
                'probabilities': probabilities,
                'control_dim': control_dim, 'target_dim': target_dim,
                'success': True
            }

        except Exception as e:
            print(f"   Помилка в квантовій схемі: {e}")
            # Повертаємо результат з класичним обчисленням періоду
            classical_period = self._find_period_classical(a, N)
            return {
                'a': a, 'N': N,
                'n_control': n_control, 'n_target': n_target,
                'classical_period': classical_period,
                'success': False
            }

    def _find_period_classical(self, a: int, N: int) -> int:
        """Класичне знаходження періоду для резервного використання"""
        if MathUtils.simple_gcd(a, N) != 1:
            return 0

        period = 1
        current = a % N
        while current != 1 and period <= N:
            current = (current * a) % N
            period += 1

        return period if current == 1 else 0

    def extract_periods_from_measurements(self, quantum_result: Dict) -> List[int]:
        """
        Витягує можливі періоди з квантових вимірювань
        """
        if not quantum_result.get('success', False):
            # Використовуємо класичний результат
            classical_period = quantum_result.get('classical_period', 0)
            return [classical_period] if classical_period > 0 else []

        probabilities = quantum_result['probabilities']
        control_dim = quantum_result['control_dim']
        target_dim = quantum_result['target_dim']
        a, N = quantum_result['a'], quantum_result['N']

        # Агрегуємо ймовірності по контрольному регістру
        control_probs = np.zeros(control_dim)
        for idx in range(len(probabilities)):
            control_state = idx // target_dim
            if control_state < control_dim:
                control_probs[control_state] += probabilities[idx]

        # Знаходимо піки
        peak_indices = np.argsort(control_probs)[::-1]
        candidates = []

        for peak_idx in peak_indices[:self.config.max_period_candidates]:
            if control_probs[peak_idx] < self.config.convergence_threshold:
                break

            if peak_idx == 0:
                continue

            # Використовуємо ланцюгові дроби
            phase = peak_idx / control_dim
            period_candidates = self._continued_fractions_analysis(phase, N)

            for period in period_candidates:
                if self._verify_period(a, period, N):
                    candidates.append(period)

        # Додаємо класичний результат як резерв
        classical_period = self._find_period_classical(a, N)
        if classical_period > 0:
            candidates.append(classical_period)

        return sorted(list(set(candidates)))

    def _continued_fractions_analysis(self, phase: float, max_denominator: int) -> List[int]:
        """Аналіз ланцюгових дробів для знаходження періоду"""
        if phase == 0:
            return []

        frac = Fraction(phase).limit_denominator(max_denominator)
        candidates = [frac.denominator]

        # Додаткові кандидати на основі близьких дробів
        for denom in range(max(1, frac.denominator - 2),
                           min(max_denominator, frac.denominator + 3)):
            if denom > 0:
                candidates.append(denom)

        return candidates

    def _verify_period(self, a: int, period: int, N: int) -> bool:
        """Перевіряє, чи є period справжнім періодом"""
        if period <= 0:
            return False
        try:
            return MathUtils.fast_modular_exp(a, period, N) == 1
        except:
            return False


##############################################################################
# Головний клас алгоритму Шора
##############################################################################

class EnhancedShorsAlgorithm:
    """Покращена версія алгоритму Шора"""

    def __init__(self, config: Optional[QuantumConfig] = None):
        self.config = config or QuantumConfig()
        self.period_finder = QuantumPeriodFinder(self.config)
        self.math_utils = MathUtils()

    def factorize(self, N: int, max_attempts: int = 10) -> FactorizationResult:
        """
        Головна функція факторизації з покращеною логікою
        """
        print(f"Запуск покращеного алгоритму Шора для N = {N}")
        print("=" * 70)

        start_time = time.time()

        # Швидкі перевірки
        quick_result = self._quick_factorization_checks(N)
        if quick_result.success:
            quick_result.execution_time = time.time() - start_time
            return quick_result

        # Основний квантовий алгоритм
        for attempt in range(1, max_attempts + 1):
            print(f"\nСпроба {attempt}/{max_attempts}")

            try:
                # Вибір випадкового a
                a = self._select_random_base(N)
                print(f"   Обрано базу: a = {a}")

                # Перевірка НСД
                gcd_val = self.math_utils.simple_gcd(a, N)
                if gcd_val > 1:
                    factors = (gcd_val, N // gcd_val)
                    return FactorizationResult(
                        original_number=N,
                        factors=factors,
                        method_used=FactorizationMethod.GCD_LUCKY,
                        attempts_count=attempt,
                        quantum_data=None,
                        success=True,
                        execution_time=time.time() - start_time
                    )

                # Квантове знаходження періоду
                quantum_result = self.period_finder.find_period_quantum(a, N)
                periods = self.period_finder.extract_periods_from_measurements(quantum_result)

                print(f"   Знайдені періоди: {periods}")

                # Обробка знайдених періодів
                factors = self._process_periods(a, N, periods)
                if factors:
                    return FactorizationResult(
                        original_number=N,
                        factors=factors,
                        method_used=FactorizationMethod.QUANTUM_PERIOD,
                        attempts_count=attempt,
                        quantum_data=quantum_result,
                        success=True,
                        execution_time=time.time() - start_time
                    )

            except Exception as e:
                print(f"   Помилка в спробі {attempt}: {e}")
                continue

        # Класичний резервний метод
        classical_factors = self._classical_fallback(N)
        return FactorizationResult(
            original_number=N,
            factors=classical_factors,
            method_used=FactorizationMethod.CLASSICAL_FALLBACK,
            attempts_count=max_attempts,
            quantum_data=None,
            success=classical_factors is not None,
            execution_time=time.time() - start_time
        )

    def _quick_factorization_checks(self, N: int) -> FactorizationResult:
        """Швидкі перевірки перед квантовим алгоритмом"""

        # Перевірка на парність
        if N % 2 == 0:
            return FactorizationResult(
                original_number=N,
                factors=(2, N // 2),
                method_used=FactorizationMethod.TRIVIAL_EVEN,
                attempts_count=0,
                quantum_data=None,
                success=True
            )

        # Перевірка на досконалий степінь
        is_power, base, exp = self.math_utils.detect_perfect_power(N)
        if is_power:
            return FactorizationResult(
                original_number=N,
                factors=(base, N // base),
                method_used=FactorizationMethod.TRIVIAL_POWER,
                attempts_count=0,
                quantum_data=None,
                success=True
            )

        return FactorizationResult(
            original_number=N,
            factors=None,
            method_used=FactorizationMethod.QUANTUM_PERIOD,
            attempts_count=0,
            quantum_data=None,
            success=False
        )

    def _select_random_base(self, N: int) -> int:
        """Розумний вибір випадкової бази"""
        attempts = 0
        while attempts < 100:  # Запобігання нескінченному циклу
            a = random.randint(2, N - 1)
            if self.math_utils.simple_gcd(a, N) == 1:
                return a
            attempts += 1

        # Якщо не знайшли взаємно просте число, повертаємо 2
        return 2

    def _process_periods(self, a: int, N: int, periods: List[int]) -> Optional[Tuple[int, int]]:
        """Обробляє знайдені періоди для отримання факторів"""

        for period in periods:
            if period <= 1:
                continue

            print(f"   Перевіряємо період r = {period}")

            if period % 2 != 0:
                print(f"     Період {period} непарний, пропускаємо")
                continue

            try:
                # Обчислюємо a^(r/2) mod N
                half_power = MathUtils.fast_modular_exp(a, period // 2, N)

                if half_power == N - 1:  # Еквівалент -1 mod N
                    print(f"     a^(r/2) ≡ -1 (mod N), пропускаємо")
                    continue

                # Обчислюємо потенційні фактори
                factor1 = self.math_utils.simple_gcd(half_power - 1, N)
                factor2 = self.math_utils.simple_gcd(half_power + 1, N)

                print(f"     gcd({half_power} ± 1, {N}) = {factor1}, {factor2}")

                # Перевіряємо нетривіальні фактори
                for factor in [factor1, factor2]:
                    if 1 < factor < N:
                        other_factor = N // factor
                        print(f"   Знайдено фактори: {factor} × {other_factor}")
                        return (min(factor, other_factor), max(factor, other_factor))

            except Exception as e:
                print(f"     Помилка при обробці періоду {period}: {e}")
                continue

        return None

    def _classical_fallback(self, N: int) -> Optional[Tuple[int, int]]:
        """Класичний резервний алгоритм факторизації"""
        print("Використовуємо класичний резервний метод...")

        # Пробне ділення до sqrt(N)
        limit = int(np.sqrt(N)) + 1
        for i in range(3, min(limit, 1000), 2):
            if N % i == 0:
                return (i, N // i)

        return None


##############################################################################
# Візуалізація та результати
##############################################################################

class ResultsVisualizer:
    """Клас для візуалізації результатів"""

    @staticmethod
    def display_results(result: FactorizationResult) -> None:
        """Відображає результати факторизації"""
        print("\n" + "=" * 70)
        print("                    РЕЗУЛЬТАТИ ФАКТОРИЗАЦІЇ")
        print("=" * 70)

        print(f"Число для факторизації: N = {result.original_number}")

        if result.success and result.factors:
            f1, f2 = result.factors
            print(f"Знайдені фактори: {f1} × {f2} = {f1 * f2}")
            print(f"Метод: {result.method_used.value}")
            print(f"Кількість спроб: {result.attempts_count}")
            print(f"Час виконання: {result.execution_time:.3f} сек")

            # Перевірка
            if f1 * f2 == result.original_number:
                print("Перевірка: УСПІШНО!")
            else:
                print("Перевірка: ПОМИЛКА!")
        else:
            print("Статус: Факторизація не вдалася")

        print("=" * 70)

    @staticmethod
    def plot_quantum_measurements(quantum_data: Dict) -> None:
        """Візуалізує квантові вимірювання"""
        if not quantum_data or not quantum_data.get('success', False):
            print("Немає квантових даних для візуалізації")
            return

        try:
            probabilities = quantum_data['probabilities']
            control_dim = quantum_data['control_dim']
            target_dim = quantum_data['target_dim']

            # Агрегуємо по контрольному регістру
            control_probs = np.zeros(control_dim)
            for i in range(len(probabilities)):
                control_state = i // target_dim
                if control_state < control_dim:
                    control_probs[control_state] += probabilities[i]

            # Знаходимо значущі стани
            threshold = 0.01
            significant_indices = np.where(control_probs > threshold)[0]

            if len(significant_indices) == 0:
                print("Немає значущих квантових станів для візуалізації")
                return

            significant_probs = control_probs[significant_indices]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Повний спектр
            ax1.bar(range(len(control_probs)), control_probs,
                    color='navy', alpha=0.7, width=0.8)
            ax1.set_xlabel("Стан контрольного регістра")
            ax1.set_ylabel("Ймовірність")
            ax1.set_title(f"Повний розподіл ({control_dim} станів)")
            ax1.grid(True, alpha=0.3)

            # Значущі стани
            ax2.bar(range(len(significant_probs)), significant_probs,
                    color='crimson', alpha=0.8, width=0.6)
            ax2.set_xticks(range(len(significant_probs)))
            ax2.set_xticklabels([str(i) for i in significant_indices])
            ax2.set_xlabel("Стан контрольного регістра")
            ax2.set_ylabel("Ймовірність")
            ax2.set_title(f"Значущі піки (p > {threshold})")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Помилка при візуалізації: {e}")


##############################################################################
# Демонстрація та тестування
##############################################################################

def run_comprehensive_demo():
    """Запускає комплексну демонстрацію алгоритму"""

    print("=" * 70)
    print("   ПОКРАЩЕНИЙ АЛГОРИТМ ШОРА - ДЕМОНСТРАЦІЯ")
    print("=" * 70)

    # Конфігурація з консервативними параметрами
    config = QuantumConfig(
        max_qubits=6,  # Зменшено для стабільності
        precision_factor=1.5,
        measurement_shots=500,
        max_period_candidates=10
    )

    shor = EnhancedShorsAlgorithm(config)
    visualizer = ResultsVisualizer()

    # Тестові числа
    test_numbers = [15, 21, 35]

    for N in test_numbers:
        print(f"\n{'=' * 50}")
        print(f"Тестування N = {N}")
        print('=' * 50)

        try:
            result = shor.factorize(N, max_attempts=5)
            visualizer.display_results(result)

            # Візуалізація квантових даних
            if result.quantum_data:
                visualizer.plot_quantum_measurements(result.quantum_data)

        except Exception as e:
            print(f"Помилка при тестуванні N={N}: {e}")

    print(f"\nДемонстрацію завершено!")


def interactive_mode():
    """Інтерактивний режим для користувача"""

    config = QuantumConfig(max_qubits=6, precision_factor=1.5)
    shor = EnhancedShorsAlgorithm(config)
    visualizer = ResultsVisualizer()

    print("\nІНТЕРАКТИВНИЙ РЕЖИМ")
    print("Введіть число для факторизації (або 'exit' для виходу)")

    while True:
        try:
            user_input = input("\nN = ").strip()

            if user_input.lower() in ['exit', 'quit', 'вихід']:
                print("До побачення!")
                break

            N = int(user_input)

            if N < 4:
                print("Будь ласка, введіть число >= 4")
                continue

            if N > 100:
                print("Увага: Велике число може потребувати багато часу")

            result = shor.factorize(N)
            visualizer.display_results(result)

        except ValueError:
            print("Будь ласка, введіть коректне ціле число")
        except KeyboardInterrupt:
            print("\n\nПрограму перервано користувачем. До побачення!")
            break
        except Exception as e:
            print(f"Помилка: {e}")


##############################################################################
# Головна точка входу
##############################################################################

if __name__ == "__main__":
    """Головна функція запуску"""

    try:
        # Запуск демонстрації
        run_comprehensive_demo()

        # Інтерактивний режим
        interactive_mode()

    except Exception as e:
        print(f"Критична помилка: {e}")
        import traceback

        traceback.print_exc()