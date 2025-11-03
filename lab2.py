from typing import Iterable
import matplotlib.pyplot as plt
import math

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
    phi = []

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

def mo_calc(j_list, result):
    x = [(0.5*(x_left+x_right)) for x_left, x_right in j_list]
    m = float(0)
    for x_l, P_l in zip(x,result):
        m += x_l*P_l
    return m

def delta(result, mo, j_list):
    D = float(0)
    x = [(0.5 * (x_left + x_right)) for x_left, x_right in j_list]
    for x, p in zip(x,result):
        D += (x**2)*p
    D -= mo**2
    d = math.sqrt(D)
    return d

def norm_law(delta, mo, j_list):

    x = list()
    f_x = list()
    pi = 3.14
    e = 2.71828

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
        f_x.append(1/(delta*math.sqrt(2*pi))*e**-(((i-mo)**2)/2*delta**2))
    return f_x, x

def main():


    # 1. Найти статистические вероятности попадания значений случайной величины в интервалы Jl, l=(1,7) ̅ по заданному числу попаданий ml
    result = p_calculate(m_array,len(m_array))
    print(f'Jl {J_list}')
    print(f'ml {m_array}')
    print(f'Pl {result}')
    #TODO: table view

    # 2. Построить гистограмму распределения экспериментальных данных.
    intervals, phi = gistogram_maker(J_list, result)
    print("Intervals length: ", len(intervals))
    print("Phi length: ", len(phi))
    print(f'Высота: {phi}')
    print(f'Ширина: {intervals}')



    # 3. Найти теоретическую плотность нормального распределения в соответствии с методом моментов, полученную кривую нанести на гистограмму распределения.

    MO = mo_calc(J_list, result)
    delta_result = delta(result, MO, J_list)

    print(f'Эмпирическое МО: {MO}')
    print(f'Эмпирическая дисперсия {delta_result}')

    f_x, x = norm_law(delta_result,MO,J_list)
    print(f'f_x: {f_x}')

    plt.bar(
        x=intervals,
        height=phi,
        width=0.5,
        edgecolor='black',
        align='center'
    )
    plt.plot(
        x,
        f_x,
        'r-',
        linewidth=2,
        label='Нормальная кривая'
    )

    plt.xlabel("x")
    plt.ylabel("ф*_l")
    plt.title("Гистограмма распределения")
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.show()

if __name__ == '__main__':
    main()

