# Простая программа для шагов 1–5
# Все входные данные задаются прямо здесь.

from math import sqrt, erfc, log
#from scipy import stats

# ==== 0) ВХОДНЫЕ ДАННЫЕ ====
# ЗАМЕНИТЕ на свои значения (табл. 1.1):
data = [
    1.3, 2.8, 5.6, 7.2, 9.8, 11.7, 19.5, 8.2, 6, 3.1, 1.2
]

# Из табл. 1.2:
gamma_for_mean = 0.81    # доверительная вероятность для ДИ СРЕДНЕГО (п.5а)
#delta_for_mean = 0.30     # максимально допустимая погрешность для СРЕДНЕГО (п.5б), в тех же единицах
eps_beta = 0.30 # максимальная вероятностная погрешность
t_beta = 1.31 # из приложения 4
phi = 0.1026 # из приложения 2
# ==== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====
def mean(values):
    return sum(values) / len(values)

def std_unbiased(values):
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    return sqrt(sum((x - m) ** 2 for x in values) / (n - 1))

# ==== РАСЧЁТЫ ====
n0 = len(data)
xbar0 = mean(data) # оценка математического ожидания
s0 = std_unbiased(data) # оценка среднего квадратичного отклонения

# 2) 95% интервал через максимальную вероятностную погрешность Iβ=[a~−εβ;a~+εβ]
lo_X = xbar0 - 2 * s0
hi_X = xbar0 + 2 * s0

#95%-доверительный интервал через стандартное отклонение I0.95=[M~x^−2σ~x^;M~x^+2σ~x^]
#lo_X = xbar0 - 2 * s0
#hi_X = xbar0 + 2 * s0

# 3) Отсекаем наблюдения вне [lo_X, hi_X]
refined = [x for x in data if lo_X <= x <= hi_X]
n1 = len(refined)
xbar1 = mean(refined) if n1 > 0 else float('nan') #пересчитываем математическое ожидание
s1 = std_unbiased(refined) if n1 > 1 else (0.0 if n1 == 1 else float('nan')) #пересчитываем среднее отклонение

# 5a) ДИ для математического ожидания при γ (используем очищенную выборку)
# Шаг 5a: ДИ для матожидания при β=0.81 (используем очищенную выборку)
if n1 >= 2:
    t_beta = 1.31  # из таблицы для β = 0.81
    eps_beta_mean = t_beta * s1 / (n1 ** 0.5)
    lo_mu = xbar1 - eps_beta_mean
    hi_mu = xbar1 + eps_beta_mean
else:
    eps_beta_mean = float('nan')
    lo_mu = float('nan')
    hi_mu = float('nan')

# 5b) Доверительная вероятность для заданной погрешности δ вокруг среднего (нормальное приближение)
if n1 >= 2 and s1 > 0:
    z_stat = eps_beta / (s1 / sqrt(n1))
    beta = 2*phi

# Вариант мой с формулой обратной функции Лапласа
t_beta_new = z_stat*(beta/2)
eps_beta_2 = (s1/sqrt(n1))*t_beta_new

#Перерасчет доверительного интервала с новым заданным eps_beta
# lo_mu = xbar1 - eps_beta
# hi_mu = xbar1 + eps_beta


# ==== ВЫВОД ====
print("Шаг 1) Среднее по исходным данным")
print(f"  n = {n0}")
print(f"  x̄ = {xbar0:.6g}")
#print(f"  s  = {s0:.6g}")
print()

print("Шаг 2) 95%-интервал для самой случайной величины X")
print(f" σ̃ₓ̂ = {s0}")
print(f" εβ= {eps_beta}")
print(f"  Интервал: [{lo_X:.6g}, {hi_X:.6g}]")
print()

print("Шаг 3) Отсев аномальных наблюдений вне интервала шага 2")
print(f"  Удалено наблюдений: {n0 - n1}")
print(f"  Осталось: n_refined = {n1}")
print()

print("Шаг 4) Уточнённая оценка среднего (после отсечения)")
print(f"  x̄_refined = {xbar1:.6g}")
#print(f"  s_refined  = {s1:.6g}")
print()

# --- Шаг 5a: доверительный интервал для мат. ожидания ---
print("Шаг 5a) Доверительный интервал для математического ожидания (β = 0.81)")
print(f"  εβ (для M(x)) = {eps_beta_mean:.6g}")
print(f"  tβ = {t_beta},  n = {n1}")
print(f"  Интервал: [{lo_mu:.6g}, {hi_mu:.6g}]")
print()

# # --- Шаг 5b: расчёт доверительной вероятности ---
# print("Шаг 5b) Проверка качества оценивания математического ожидания")
# print(f"  z = (εβ * √n) / σ̃ = {z_stat:.6g}")
# print(f"  По таблице Ф₀({z_stat:.2f}) = {phi:.4f}")  # сюда подставь своё значение 0.1026
# print(f"  β = 2 * Ф₀({z_stat:.2f}) = {2 * phi:.4f}")
# print()

print("Шаг 5b) Проверка качества оценивания математического ожидания")
print(f"  εβ (заданная макс. погрешность) = {eps_beta:.6g}")
print(f"  z = (εβ * √n) / σ̃ = {z_stat:.6g}")
print(f"  По таблице Ф₀({z_stat:.2f}) = {phi:.4f}")     # например, 0.1026
print(f"  β = 2 * Ф₀({z_stat:.2f}) = {beta:.4f}")         # β, полученная из таблицы
#print(f"  Новая εβ (для M(x)) = tβ * s / √n = {eps_beta_2:.6g}")
#print(f"  tβ (по новой β) = {t_beta:.4f}")
print(f"  Интервал по заданной максимальной вероятной погрешностью: [{xbar1 - eps_beta:.6g}, {xbar1 + eps_beta:.6g}]")
print()