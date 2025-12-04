"""
Grover_qiskit_2qubit.py
Простий приклад алгоритму Гровера для пошуку одного "позначеного" елемента
в невпорядкованій базі з 4 елементів (2 кубіти).
Використовується бібліотека Qiskit (новий API без execute).
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer


def grover_oracle_2qubit(target: str) -> QuantumCircuit:
    """
    Створює оракул для 2-кубітної системи.

    target: двійковий рядок з 2 символів: '00', '01', '10' або '11'
    Оракул має змінювати знак амплітуди лише для стану target.

    Ідея:
    1. Якщо target не '11', то ми робимо X на тих кубітах,
       де target має '0', щоб перетворити потрібний стан у |11>.
    2. Застосовуємо контрольований-Z (через H-CNOT-H на останньому кубіті).
    3. Повертаємо X назад, щоб відкотити підготовку.
    """

    if len(target) != 2 or any(c not in "01" for c in target):
        raise ValueError("target повинен бути рядком з двох символів '0' або '1', напр. '11'.")

    qc = QuantumCircuit(2)

    # Крок 1: підготовка до CZ – приводимо target до стану |11>
    # reversed, бо qubit 0 відповідає молодшому біту
    for i, bit in enumerate(reversed(target)):
        if bit == '0':
            qc.x(i)

    # Крок 2: реалізація контрольованого-Z через H-CNOT-H на цільовому кубіті
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)

    # Крок 3: відкот X, щоб повернути початкову базу
    for i, bit in enumerate(reversed(target)):
        if bit == '0':
            qc.x(i)

    return qc


def diffusion_operator_2qubit() -> QuantumCircuit:
    """
    Оператор дифузії (інверсія відносно середнього) для 2 кубітів.

    Стандартна побудова:
    1. Hadamard на всі кубіти.
    2. X на всі кубіти.
    3. Оператор, який змінює знак стану |11> (через H-CNOT-H).
    4. X на всі кубіти.
    5. Hadamard на всі кубіти.
    """

    qc = QuantumCircuit(2)

    # 1. H на всі кубіти
    qc.h([0, 1])

    # 2. X на всі кубіти
    qc.x([0, 1])

    # 3. Знак зміни для |11>
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)

    # 4. X назад
    qc.x([0, 1])

    # 5. H назад
    qc.h([0, 1])

    return qc


def build_grover_circuit_2qubit(target: str) -> QuantumCircuit:
    """
    Створює повний квантовий цикл алгоритму Гровера для 2 кубітів.

    target: який стан ми шукаємо (наприклад '11').

    Кроки:
    1. Ініціалізація: приводимо обидва кубіти в суперпозицію (H на кожен).
    2. Застосовуємо оракул (мітить потрібний елемент зміною фази).
    3. Застосовуємо оператор дифузії (підсилює амплітуду позначеного елементу).
    4. Вимірюємо обидва кубіти.
    """

    qc = QuantumCircuit(2, 2)

    # 1. Рівномірна суперпозиція
    qc.h([0, 1])

    # 2. Оракул
    oracle = grover_oracle_2qubit(target)
    qc.compose(oracle, inplace=True)

    # 3. Оператор дифузії
    diffusion = diffusion_operator_2qubit()
    qc.compose(diffusion, inplace=True)

    # 4. Вимірювання
    qc.measure([0, 1], [0, 1])

    return qc


def run_grover(target: str = "11", shots: int = 1024):
    """
    Запускає алгоритм Гровера на симуляторі для заданого target.

    target: рядок '00'/'01'/'10'/'11' – який стан вважаємо "знайденим".
    shots: скільки разів запускати схему для збору статистики вимірювань.
    """

    # Створюємо квантову схему
    circuit = build_grover_circuit_2qubit(target)
    print("Квантова схема алгоритму Гровера:")
    print(circuit.draw())  # гарний ASCII-рисунок схеми

    # Беремо симулятор (Aer з окремого пакету qiskit-aer)
    simulator = Aer.get_backend("qasm_simulator")

    # НОВИЙ СТИЛЬ: спочатку транспілюємо під backend
    t_circuit = transpile(circuit, backend=simulator)

    # Запускаємо виконання без execute, через backend.run
    job = simulator.run(t_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    print("\nРезультати вимірювань (counts):")
    print(counts)

    # Найчастіше зустрічається бітовий рядок – це наш "знайдений" елемент
    most_probable_state = max(counts, key=counts.get)
    print(f"\nНайімовірніший результат: {most_probable_state}")
    print(f"Очікуваний target:         {target}")


if __name__ == "__main__":
    # За замовчуванням шукаємо стан '11'.
    # Можеш змінити його на '00', '01' або '10' і подивитися, що буде.
    run_grover(target="11", shots=1024)
