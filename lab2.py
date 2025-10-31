from typing import Iterable
import matplotlib.pyplot as plt

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

    return intervals, phi
def main():


    # 1. Найти статистические вероятности попадания значений случайной величины в интервалы Jl, l=(1,7) ̅ по заданному числу попаданий ml
    result = p_calculate(m_array,len(m_array))
    print(f'Jl {J_list}')
    print(f'ml {m_array}')
    print(f'Pl {result}')
    #TODO: table view

    # 2. Построить гистограмму распределения экспериментальных данных.
    intervals, phi = gistogram_maker(J_list, result)
    print(f'Высота: {phi}')
    print(f'Ширина: {intervals}')

    plt.bar(
        x=intervals,
        height=phi,
        width=0.5,
        edgecolor='black',
        align='center'
    )

    plt.xlabel("x")
    plt.ylabel("ф*_l")
    plt.title("Гистограмма")
    plt.show()




if __name__ == '__main__':
    main()

