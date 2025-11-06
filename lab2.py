from typing import Iterable
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

m_array = [3, 11, 29, 26, 22, 7, 2]
J_list = [(-2.5, -2), (-2, -1.5), (-1.5, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1)]
intervals = [] # ширина h_l
phi = [] #высота P_l
def p_calculate(m_list: Iterable[float], m: int) -> float:
    total = sum(m_list)
    return [m/total for m in m_list]

def gistogram_maker(j_list: Iterable[float], p_list: Iterable[float])->tuple[list[float], list[float]]:

    intervals = []
    h = []
    phi = [] #φ

    h = [x_right - x_left for (x_left, x_right) in j_list]
    # for x_left, x_right in j_list:
    #     #x_left, x_right = j_list[l]
    #     h.append(x_right - x_left)
    for item1, item2 in zip(p_list,h):
        phi.append(item1/item2)
    # for l in p_list:
    #     for l_2 in h:
    #         phi.append(l/l_2)

    intervals = [(x_left + x_right)/2 for (x_left, x_right) in j_list]
    # intervals = list()
    #
    # for i, (x_left, x_right) in enumerate(j_list):
    #     prev_left, prev_right = j_list[i - 1]
    #     if x_left!=prev_left and x_left!=prev_right:
    #         # print("x_left: ", x_left)
    #         # print("x_left-1: ", prev_left)
    #         # print("x_right: ", x_right)
    #         # print("x_right-1: ", prev_right)
    #         intervals.append(x_left)
    #     if x_right!=x_right-1 and x_left!=x_right-1:
    #         intervals.append(x_right)
    #
    # print("Bounder intervals: ", intervals)
    # print("Half intervals: ", intervals_half)

    return intervals, phi

# Оценка мат. ожидания
def mo_calc(j_list, result):
    x = [(0.5*(x_left+x_right)) for x_left, x_right in j_list]
    m = float(0)
    for x_l, P_l in zip(x,result):
        m += x_l*P_l
    return m

# Оценка СКО
def delta(result, mo, j_list):
    D = float(0)
    x = [(0.5 * (x_left + x_right)) for x_left, x_right in j_list]
    for x, p in zip(x,result):
        D += (x**2)*p
    D -= mo**2
    d = math.sqrt(D)
    return d

# Вычисление аппроксимирующей кривой нормального распределения
def norm_law(delta, mo, j_list):

    x = list()
    f_x = list()

    for i, (x_left, x_right) in enumerate(j_list):
        prev_left, prev_right = j_list[i - 1]
        if x_left!=prev_left and x_left!=prev_right:
            # print("x_left: ", x_left)
            # print("x_left-1: ", prev_left)
            # print("x_right: ", x_right)
            # print("x_right-1: ", prev_right)
            x.append(x_left)
        if x_right!=x_right-1 and x_left!=x_right-1:
            x.append(x_right)
    for i in x:
        f_x.append(round((1/(delta*math.sqrt(2*math.pi)))*math.exp(-(((i-mo)**2)/(2*delta**2))), 3))
    return f_x, x

def excel_reader(filename):
    df = pd.read_excel(filename)
    return df

def hypothesis_check(mo, delta, j_list):

    phi_c = list()
    df = excel_reader('table.xlsx')
    df.columns = df.columns.str.strip()
    #print(df.columns)
    arg_1, arg_2 = 0.0, 0.0
    for x_left, x_right in j_list:
        arg_1 = round(((x_right - mo) / delta), 2)
        arg_2 = round(((x_left - mo) / delta), 2)
        if arg_1 in df["x"].values:
            f_1 = df.loc[df["x"]==arg_1, "Ф1(x)"].values[0]
            #print(f_1)
        else:
            df = df.sort_values("x").reset_index(drop=True)
            mask_left = df["x"] <= arg_1  # все точки левее (или равные)
            mask_right = df["x"] >= arg_1  # все точки правее (или равные)

            # последний слева и первый справа
            x1 = df.loc[mask_right, "x"].min()
            x2 = df.loc[mask_left, "x"].max()

            y1 = df.loc[df["x"] == x1, "Ф1(x)"].values[0]
            y2 = df.loc[df["x"] == x2, "Ф1(x)"].values[0]

            f_1 = round(((arg_1 - x2) / (x1 - x2)) * y1 + ((arg_1 - x1) / (x2 - x1)) * y2, 4)
        if arg_2 in df["x"].values:
            f_2 = df.loc[df["x"]==arg_2, "Ф1(x)"].values[0]
            #print(f_2)
        else:
            df = df.sort_values("x").reset_index(drop=True)
            mask_left = df["x"] <= arg_2  # все точки левее (или равные)
            mask_right = df["x"] >= arg_2  # все точки правее (или равные)

            # последний слева и первый справа
            x1 = df.loc[mask_right, "x"].min()
            x2 = df.loc[mask_left, "x"].max()

            y1 = df.loc[df["x"] == x1, "Ф1(x)"].values[0]
            y2 = df.loc[df["x"] == x2, "Ф1(x)"].values[0]

            f_2 = round(((arg_2-x2)/(x1-x2))*y1+((arg_2-x1)/(x2-x1))*y2, 4)

        phi_c.append(f_1-f_2)

    return phi_c

def smooth_curve(delta, mo, start, end):
    x = np.arange(start, end, 0.01)
    y = list()
    for i in x:
        y.append((1/(delta*math.sqrt(2*math.pi)))*math.exp(-(((i-mo)**2)/(2*delta**2))))
    return x, y

def print_table_rows(headers, rows, precision=3):
    """
    headers: список названий строк
    rows: список списков со значениями
    """
    for header, values in zip(headers, rows):
        formatted = "  ".join(f"{v:.{precision}f}" for v in values)
        print(f"{header:<20}{formatted}")

# пример вызова:
headers = [
    "p*ₗ:",
    "pₗ:",                      # ← красивый вариант из Unicode
    "p*ₗ − pₗ:",
    "(p*ₗ − pₗ))²:",
    "n(p*ₗ − pₗ))² / pₗ:",
]

def main():


    # 1. Найти статистические вероятности попадания значений случайной величины в интервалы Jl, l=(1,7) ̅ по заданному числу попаданий ml
    result = p_calculate(m_array,len(m_array))
    print(f'Jₗ {J_list}')
    print(f'mₗ {m_array}')
    print(f'P*ₗ {result}\n')
    #TODO: table view

    # 2. Построить гистограмму распределения экспериментальных данных.
    intervals, phi = gistogram_maker(J_list, result)
    #print("Intervals length: ", len(intervals))
    #print("Phi length: ", len(phi))
    #print(f'Высота (φ*ₗ): {phi}')
    #print(f'Ширина: {intervals}')



    # 3. Найти теоретическую плотность нормального распределения в соответствии с методом моментов, полученную кривую нанести на гистограмму распределения.

    MO = mo_calc(J_list, result)
    delta_result = delta(result, MO, J_list)

    print(f'Оценка мат. ожидания: {MO}')
    print(f'Оценка СКО: {delta_result}\n')

    f_x, x = norm_law(delta_result,MO,J_list)
    print(f'x: {x}')
    print(f'f(xₗ): {f_x}\n')

    x, y = smooth_curve(delta_result,MO,-2.5,1)

    # 4. Проверить гипотезу о соответствии статистического и теоретического распределений
    # ( т. е. гипотезу о нормальном распределении случайной величины) методом К. Пирсона при уровне значимости: α = 0,025 – для четных вариантов

    # f1_table = excel_reader('table.xlsx')
    # print(f1_table)

    # Расчитываем вероятность попадания случайной величины в диапозоны Jₗ (pₗ)
    check_result = hypothesis_check(MO,delta_result,J_list)
    check_result_float = list()
    for np_x in check_result:
        check_result_float.append(round(np_x.item(), 3))
    # Расчитываем p*ₗ-pₗ
    pl_star_minus_pl = []
    for pl, pl_star in zip(check_result_float, result):
        pl_star_minus_pl.append(round(pl_star-pl, 3))
    # Расчитываем (p*ₗ-pₗ)^2
    pl_star_minus_pl_2 = []
    for pl in pl_star_minus_pl:
        pl_star_minus_pl_2.append(round(pl**2, 3))
    # Расчитываем n(p*ₗ-pₗ)^2/pₗ

    final_pl = []
    n = sum(m_array)
    for pl, phi_l in zip(pl_star_minus_pl_2, result):
        final_pl.append(round((n*pl)/phi_l, 3))
    # Находим наблюдаемое значение показателя согласованности гипотезы
    u = sum(final_pl)

    rows = [
        check_result_float,
        result,
        pl_star_minus_pl,
        pl_star_minus_pl_2,
        final_pl,
    ]

    print(f'Jₗ: {J_list}\n')
    print("РЕЗУЛЬТАТЫ ВЫЧИСЛЕНИЙ:\n")
    print_table_rows(headers, rows)
    print("\nНаблюдаемое значение u:", round(u, 3))

    # Графики

    plt.bar(
        x=intervals,
        height=phi,
        width=0.5,
        edgecolor='black',
        align='center'
    )
    plt.plot(
        x,
        y,
        'r-',
        linewidth=2,
        label='Теоретическая кривая распределения'
    )

    plt.xlabel("x")
    plt.ylabel("φ*ₗ")
    plt.title("Гистограмма распределения")
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.show()

if __name__ == '__main__':
    main()

