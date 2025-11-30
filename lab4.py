import numpy as np
import statsmodels.api as sm



x = np.array([-2, 0, 1, 2, 4], dtype=float)
y = np.array([18, 3, 1, 3, 21], dtype=float)

# Строим матрицу для уравнения регрессии вида y = a0*x^2 + a1*x + a2
"""[[ 4. -2.  1.]
 [ 0.  0.  1.]
 [ 1.  1.  1.]
 [ 4.  2.  1.]
 [16.  4.  1.]]"""
X = np.column_stack([
    x**2,   # столбец для a0
    x,      # столбец для a1
    np.ones_like(x)  # столбец для a2 (свободный член)
])

# Находим произведение транспонированной матрицы F^T на исходную
XtX = X.T @ X

# Найдём произведение транспонированной матрицы F^T на вектор Y
Xty = X.T @ y

# Решаем систему нормальных уравнений
a = np.linalg.solve(XtX, Xty)

# Полученные оценки коэффициентов регрессии:
print("Полученные оценки коэффициентов регрессии:", a)

# model = sm.OLS(y, X)
# results = model.fit()
#
# print("/n")
# print(results.summary())

# ---------------------------
# 2. Построение ŷ = X a
# ---------------------------
y_hat = X @ a

# ---------------------------
# 3. Сумма квадратов отклонений SST и SSE
# ---------------------------
SST = np.sum((y - np.mean(y))**2)
SSE = np.sum((y - y_hat)**2)
SSR = SST - SSE

print("SST =", SST)
print("SSR =", SSR)
print("SSE =", SSE)

# ---------------------------
# 4. Дисперсии по методичке
# ---------------------------
sigma_tilde2 = SST / (n - 1)     # общая дисперсия
sigma_res2   = SSE / (n - m)     # остаточная дисперсия

print("tilde_sigma^2 =", sigma_tilde2)
print("sigma_res^2 =", sigma_res2)

# ---------------------------
# 5. F-критерий по формуле методички
# ---------------------------
F_emp = sigma_tilde2 / sigma_res2
F_crit = f.ppf(1 - 0.01, n - 1, n - m)   # F(α, f1=n-1, f2=n-m)

print("\nF эмпирическое =", F_emp)
print("F критическое =", F_crit)

if F_emp > F_crit:
    print("→ Уравнение регрессии адекватно экспериментальным данным (по методичке).")
else:
    print("→ Уравнение регрессии НЕ адекватно.")

# ---------------------------
# 6. t-критерий для коэффициентов по методичке
# ---------------------------

# Остаточная дисперсия SSE/(n-m)
s2 = sigma_res2
cov = np.linalg.inv(XtX) * s2
stderr = np.sqrt(np.diag(cov))

t_emp = a / stderr
t_crit = t.ppf(1 - 0.01/2, n - m)   # двусторонний критерий

print("\nt-эмп:", t_emp)
print("t-крит:", t_crit)

for i, ti in enumerate(t_emp):
    if abs(ti) > t_crit:
        print(f"Коэффициент a{i} – значим")
    else:
        print(f"Коэффициент a{i} – НЕ значим")