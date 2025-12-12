"""
Алгоритм Шора

Цей модуль реалізує алгоритм Шора для факторизації цілих чисел.

Алгоритм Шора — квантовий алгоритм, розроблений Пітером Шором у 1994 році,
який може факторизувати великі цілі числа експоненціально швидше за будь-який
відомий класичний алгоритм. Це має значення для криптографії, оскільки
алгоритм може зламати RSA шифрування.

Основні компоненти:
1. Квантове знаходження періоду (quantum period finding)
2. Квантове перетворення Фур'є (QFT)
3. Модульне піднесення до степеня
4. Класичний постпроцесинг (алгоритм ланцюгових дробів)

Основні функції:
- qft(n): Квантове перетворення Фур'є для n кубітів
- inverse_qft(n): Обернене QFT
- quantum_period_finding(a, N): Знаходження періоду функції a^x mod N
- shor_factor(N): Головна функція факторизації
- results(N, factors): Виведення результатів
- visualize_results(state): Візуалізація ймовірностей станів

Залежності: qutip, numpy, matplotlib, fractions
"""

import math
from fractions import Fraction
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as _np
from numpy import *
from qutip import basis, tensor, sigmax, sigmaz, qeye, Qobj


##############################################################################
# Допоміжні функції
##############################################################################

def hadamard_transform(n):
    """
    Генерує матрицю Адамара для одного кубіта.

    Параметри:
        n (int): Розмір матриці (завжди 1).

    Повертає:
        Qobj: Матриця Адамара.
    """
    h = (1.0 / _np.sqrt(2.0)) * _np.array([[1, 1], [1, -1]])
    return Qobj(h)


def gcd(a: int, b: int) -> int:
    """
    Обчислює найбільший спільний дільник (НСД) двох чисел.

    Параметри:
        a (int): Перше число.
        b (int): Друге число.

    Повертає:
        int: НСД чисел a та b.
    """
    while b:
        a, b = b, a % b
    return a


def is_power(N: int) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Перевіряє, чи є N степенем якогось числа (N = a^k).

    Параметри:
        N (int): Число для перевірки.

    Повертає:
        Tuple: (True, база, степінь) якщо N є степенем, інакше (False, None, None).
    """
    for k in range(2, int(math.log2(N)) + 1):
        a = int(round(N ** (1 / k)))
        for candidate in [a - 1, a, a + 1]:
            if candidate > 1 and candidate ** k == N:
                return True, candidate, k
    return False, None, None


def mod_exp(a: int, x: int, N: int) -> int:
    """
    Обчислює модульне піднесення до степеня: a^x mod N.

    Параметри:
        a (int): База.
        x (int): Показник степеня.
        N (int): Модуль.

    Повертає:
        int: Результат a^x mod N.
    """
    result = 1
    a = a % N
    while x > 0:
        if x % 2 == 1:
            result = (result * a) % N
        x = x >> 1
        a = (a * a) % N
    return result


##############################################################################
# Квантове перетворення Фур'є (QFT)
##############################################################################

def qft_matrix(n: int) -> Qobj:
    """
    Генерує матрицю квантового перетворення Фур'є для n кубітів.

    QFT перетворює стани обчислювальної бази у суперпозиції з
    фазами, що кодують інформацію про частоту (період).

    Параметри:
        n (int): Кількість кубітів.

    Повертає:
        Qobj: Унітарна матриця QFT розміром 2^n × 2^n.
    """
    N = 2 ** n
    omega = _np.exp(2j * _np.pi / N)
    qft = _np.zeros((N, N), dtype=complex)

    for j in range(N):
        for k in range(N):
            qft[j, k] = omega ** (j * k) / _np.sqrt(N)

    return Qobj(qft)


def inverse_qft_matrix(n: int) -> Qobj:
    """
    Генерує обернену матрицю QFT для n кубітів.

    Параметри:
        n (int): Кількість кубітів.

    Повертає:
        Qobj: Обернена унітарна матриця QFT.
    """
    return qft_matrix(n).dag()


##############################################################################
# Модульна експоненціація (квантова версія)
##############################################################################

def controlled_mod_exp_operator(a: int, N: int, n_count: int) -> Qobj:
    """
    Створює оператор керованої модульної експоненціації.

    Цей оператор реалізує: |x⟩|y⟩ → |x⟩|y * a^x mod N⟩

    Для симуляції ми створюємо оператор, що діє на комбінований
    простір лічильного регістра та цільового регістра.

    Параметри:
        a (int): База для експоненціації.
        N (int): Модуль.
        n_count (int): Кількість кубітів у лічильному регістрі.

    Повертає:
        Qobj: Унітарний оператор модульної експоненціації.
    """
    # Розмір лічильного регістра
    count_dim = 2 ** n_count
    # Розмір цільового регістра (достатньо для представлення N)
    n_target = int(_np.ceil(_np.log2(N))) + 1
    target_dim = 2 ** n_target

    # Загальний розмір простору
    total_dim = count_dim * target_dim

    # Створюємо матрицю оператора
    U = _np.zeros((total_dim, total_dim), dtype=complex)

    for x in range(count_dim):
        for y in range(target_dim):
            # Обчислюємо новий стан цільового регістра
            if y < N:
                new_y = (y * mod_exp(a, x, N)) % N
            else:
                new_y = y  # Стани >= N залишаються незмінними

            # Індекси у загальному просторі
            in_idx = x * target_dim + y
            out_idx = x * target_dim + new_y
            U[out_idx, in_idx] = 1

    return Qobj(U)


##############################################################################
# Квантове знаходження періоду
##############################################################################

def quantum_period_finding(a: int, N: int, n_count: int = None) -> Dict[str, Any]:
    """
    Реалізує квантове знаходження періоду функції f(x) = a^x mod N.

    Це серце алгоритму Шора. Алгоритм знаходить період r такий,
    що a^r ≡ 1 (mod N).

    Квантова схема:
    1. Ініціалізація: |0⟩^n ⊗ |1⟩
    2. Застосування H до лічильного регістра
    3. Керована модульна експоненціація
    4. Обернене QFT на лічильному регістрі
    5. Вимірювання

    Параметри:
        a (int): База (випадкове число, взаємно просте з N).
        N (int): Число для факторизації.
        n_count (int): Кількість кубітів у лічильному регістрі.

    Повертає:
        Dict: Словник з результатами, включаючи знайдений період.
    """
    # Визначаємо кількість кубітів
    n_target = int(_np.ceil(_np.log2(N))) + 1
    if n_count is None:
        n_count = 2 * n_target  # Стандартна точність

    # Обмежуємо для симуляції на класичному комп'ютері
    if n_count > 10:
        n_count = 10
        print(f"Увага: Обмежено до {n_count} кубітів для симуляції")

    count_dim = 2 ** n_count
    target_dim = 2 ** n_target

    print(f"Параметри схеми:")
    print(f"  - Лічильний регістр: {n_count} кубітів ({count_dim} станів)")
    print(f"  - Цільовий регістр: {n_target} кубітів ({target_dim} станів)")
    print(f"  - Загальний розмір: {count_dim * target_dim} станів")

    # Крок 1: Ініціалізація станів
    # Лічильний регістр: |0...0⟩
    # Цільовий регістр: |1⟩ (початковий стан для a^0 mod N = 1)
    count_init = basis(count_dim, 0)
    target_init = basis(target_dim, 1)
    psi = tensor(count_init, target_init)

    # Крок 2: Застосування Адамара до лічильного регістра
    # Створює рівномірну суперпозицію всіх x від 0 до 2^n_count - 1
    H_count = qft_matrix(n_count).dag() * qft_matrix(n_count)  # Ідентичність для перевірки
    
    # Насправді H^⊗n можна виразити через QFT:
    # Але простіше створити безпосередньо
    H = hadamard_transform(1)
    H_n = H
    for _ in range(n_count - 1):
        H_n = tensor(H_n, H)
    
    # Тензорний добуток з одиничним оператором на цільовому регістрі
    H_full = tensor(H_n, qeye(target_dim))
    psi = H_full * psi

    # Крок 3: Керована модульна експоненціація
    # |x⟩|1⟩ → |x⟩|a^x mod N⟩
    U_mod_exp = controlled_mod_exp_operator(a, N, n_count)
    psi = U_mod_exp * psi

    # Крок 4: Обернене QFT на лічильному регістрі
    QFT_inv = inverse_qft_matrix(n_count)
    QFT_inv_full = tensor(QFT_inv, qeye(target_dim))
    psi = QFT_inv_full * psi

    # Крок 5: Аналіз результатів
    # Ймовірності для кожного стану
    probabilities = abs(psi.full().flatten()) ** 2

    # Знаходимо стани з найвищою ймовірністю
    peak_indices = _np.argsort(probabilities)[::-1]

    # Аналізуємо перші кілька піків
    periods_found = []
    for idx in peak_indices[:10]:
        if probabilities[idx] < 0.001:
            continue
        # Розділяємо на лічильний та цільовий регістри
        count_state = idx // target_dim
        target_state = idx % target_dim

        # Використовуємо алгоритм ланцюгових дробів
        if count_state > 0:
            phase = count_state / count_dim
            frac = Fraction(phase).limit_denominator(N)
            r = frac.denominator
            if r > 1 and mod_exp(a, r, N) == 1:
                periods_found.append(r)

    # Видаляємо дублікати та сортуємо
    periods_found = sorted(set(periods_found))

    return {
        'a': a,
        'N': N,
        'n_count': n_count,
        'n_target': n_target,
        'state': psi,
        'probabilities': probabilities,
        'periods_found': periods_found,
        'count_dim': count_dim,
        'target_dim': target_dim
    }


##############################################################################
# Алгоритм ланцюгових дробів
##############################################################################

def continued_fraction_expansion(x: float, max_terms: int = 20) -> List[int]:
    """
    Обчислює ланцюговий дріб для числа x.

    Параметри:
        x (float): Число для розкладу.
        max_terms (int): Максимальна кількість термінів.

    Повертає:
        List[int]: Коефіцієнти ланцюгового дробу.
    """
    coefficients = []
    for _ in range(max_terms):
        integer_part = int(x)
        coefficients.append(integer_part)
        fractional_part = x - integer_part
        if fractional_part < 1e-10:
            break
        x = 1 / fractional_part
    return coefficients


def convergents(coefficients: List[int]) -> List[Tuple[int, int]]:
    """
    Обчислює конвергенти (наближення) з коефіцієнтів ланцюгового дробу.

    Параметри:
        coefficients (List[int]): Коефіцієнти ланцюгового дробу.

    Повертає:
        List[Tuple[int, int]]: Список (чисельник, знаменник) для кожного конвергента.
    """
    convergent_list = []
    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0

    for a in coefficients:
        h_new = a * h_curr + h_prev
        k_new = a * k_curr + k_prev
        convergent_list.append((h_new, k_new))
        h_prev, h_curr = h_curr, h_new
        k_prev, k_curr = k_curr, k_new

    return convergent_list


##############################################################################
# Головна функція факторизації
##############################################################################

def shor_factor(N: int, max_attempts: int = 10) -> Dict[str, Any]:
    """
    Головна функція алгоритму Шора для факторизації числа N.

    Параметри:
        N (int): Число для факторизації.
        max_attempts (int): Максимальна кількість спроб.

    Повертає:
        Dict: Словник з результатами, включаючи знайдені фактори.
    """
    print("=" * 60)
    print(f"Алгоритм Шора: Факторизація N = {N}")
    print("=" * 60)

    # Перевірка: N має бути >= 2
    if N < 2:
        return {'N': N, 'factors': None, 'error': 'N має бути >= 2'}

    # Перевірка: N не має бути парним
    if N % 2 == 0:
        print(f"N = {N} є парним. Тривіальний фактор: 2")
        return {'N': N, 'factors': (2, N // 2), 'method': 'парне число'}

    # Перевірка: N не має бути степенем
    is_pow, base, exp = is_power(N)
    if is_pow:
        print(f"N = {N} = {base}^{exp}. Тривіальний фактор: {base}")
        return {'N': N, 'factors': (base, N // base), 'method': 'степінь числа'}

    # Перевірка: N має бути складеним
    # (Для простоти, пропускаємо повну перевірку на простоту)

    factors = None
    attempt = 0
    qpf_result = None

    while factors is None and attempt < max_attempts:
        attempt += 1
        print(f"\n--- Спроба {attempt} ---")

        # Крок 1: Вибір випадкового a, взаємно простого з N
        a = _np.random.randint(2, N)
        g = gcd(a, N)
        print(f"Вибрано a = {a}")

        if g > 1:
            # Знайдено нетривіальний дільник!
            print(f"Пощастило! gcd({a}, {N}) = {g} є фактором!")
            factors = (g, N // g)
            break

        # Крок 2: Квантове знаходження періоду
        print(f"Запуск квантового знаходження періоду для a = {a}")
        qpf_result = quantum_period_finding(a, N)

        # Крок 3: Аналіз результатів
        for r in qpf_result['periods_found']:
            print(f"Перевіряємо період r = {r}")

            # Перевірка: r має бути парним
            if r % 2 != 0:
                print(f"  r = {r} непарний, пропускаємо")
                continue

            # Обчислюємо потенційні фактори
            x = mod_exp(a, r // 2, N)
            if x == N - 1:  # x ≡ -1 (mod N)
                print(f"  a^(r/2) ≡ -1 (mod N), пропускаємо")
                continue

            factor1 = gcd(x - 1, N)
            factor2 = gcd(x + 1, N)

            print(f"  gcd({x} - 1, {N}) = {factor1}")
            print(f"  gcd({x} + 1, {N}) = {factor2}")

            if 1 < factor1 < N:
                factors = (factor1, N // factor1)
                print(f"Знайдено фактори: {factors}")
                break
            if 1 < factor2 < N:
                factors = (factor2, N // factor2)
                print(f"Знайдено фактори: {factors}")
                break

        # Додаткова спроба з класичним пошуком періоду (для малих N)
        if factors is None:
            print("Класичний пошук періоду...")
            r = 1
            while r <= N:
                if mod_exp(a, r, N) == 1:
                    break
                r += 1
            if r <= N and r % 2 == 0:
                x = mod_exp(a, r // 2, N)
                if x != N - 1:
                    factor1 = gcd(x - 1, N)
                    factor2 = gcd(x + 1, N)
                    if 1 < factor1 < N:
                        factors = (factor1, N // factor1)
                        print(f"Знайдено фактори (класичний метод): {factors}")
                    elif 1 < factor2 < N:
                        factors = (factor2, N // factor2)
                        print(f"Знайдено фактори (класичний метод): {factors}")

    return {
        'N': N,
        'factors': factors,
        'attempts': attempt,
        'qpf_result': qpf_result
    }


##############################################################################
# Результати та візуалізація
##############################################################################

def results(shor_result: Dict[str, Any]) -> None:
    """
    Виводить результати алгоритму Шора.

    Параметри:
        shor_result (Dict): Результат виконання shor_factor().
    """
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТИ АЛГОРИТМУ ШОРА")
    print("=" * 60)

    N = shor_result['N']
    factors = shor_result.get('factors')

    if factors:
        f1, f2 = factors
        print(f"Число для факторизації: N = {N}")
        print(f"Знайдені фактори: {f1} × {f2} = {f1 * f2}")
        if f1 * f2 == N:
            print("Перевірка: ✓ Коректно!")
        else:
            print("Перевірка: ✗ Помилка!")
    else:
        print(f"Число для факторизації: N = {N}")
        print("Фактори не знайдено")

    print("=" * 60)


def visualize_results(qpf_result: Dict[str, Any]) -> None:
    """
    Візуалізує ймовірності квантових станів після квантового знаходження періоду.

    Параметри:
        qpf_result (Dict): Результат quantum_period_finding().
    """
    if qpf_result is None:
        print("Немає даних для візуалізації")
        return

    probabilities = qpf_result['probabilities']
    count_dim = qpf_result['count_dim']
    target_dim = qpf_result['target_dim']

    # Агрегуємо ймовірності по лічильному регістру
    count_probs = _np.zeros(count_dim)
    for i in range(len(probabilities)):
        count_state = i // target_dim
        count_probs[count_state] += probabilities[i]

    # Показуємо тільки ненульові ймовірності
    significant = count_probs > 0.01
    indices = _np.where(significant)[0]
    values = count_probs[significant]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(count_probs)), count_probs, color='purple', alpha=0.7)
    plt.xlabel("Стан лічильного регістра")
    plt.ylabel("Ймовірність")
    plt.title(f"Розподіл ймовірностей (всі {count_dim} станів)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.bar(range(len(values)), values, color='blue', alpha=0.7)
    plt.xticks(range(len(values)), [str(i) for i in indices], rotation=45)
    plt.xlabel("Стан лічильного регістра")
    plt.ylabel("Ймовірність")
    plt.title(f"Значущі стани (p > 0.01)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


##############################################################################
# Головний блок
##############################################################################

if __name__ == "__main__":
    """
    Демонстрація алгоритму Шора для факторизації числа N = 15.
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        АЛГОРИТМ ШОРА - КВАНТОВА ФАКТОРИЗАЦІЯ             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Приклад 1: Факторизація N = 15 (= 3 × 5)
    N = 15
    print(f"Приклад: Факторизація числа N = {N}")
    print("-" * 60)

    shor_result = shor_factor(N)
    results(shor_result)

    # Візуалізація (якщо доступна)
    if shor_result.get('qpf_result'):
        visualize_results(shor_result['qpf_result'])

    # Додаткові приклади (закоментовані)
    # print("\n" + "=" * 60)
    # print("Приклад 2: N = 21")
    # shor_result_21 = shor_factor(21)
    # results(shor_result_21)
    #
    # print("\n" + "=" * 60)
    # print("Приклад 3: N = 35")
    # shor_result_35 = shor_factor(35)
    # results(shor_result_35)
