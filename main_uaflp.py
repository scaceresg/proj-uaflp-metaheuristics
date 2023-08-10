import numpy as np
from alg_lobo_gris import AlgLoboGris
import matplotlib.pyplot as plt

# Sportswear manuf
# n_dptos = 11
# areas_dptos = np.array([50.00, 195.50, 83.27, 100.39, 67.20, 153.00, 
#                         153.00, 131.25, 52.90, 52.90, 100.00])
# fl_materiales = np.array([[  0,   0,   0,   0, 0,   0, 0, 0, 0, 0, 0],
#                           [  0,   0,   0,   0, 0,   0, 0, 0, 0, 0, 0],
#                           [  0, 110,   0,   0, 0,   0, 0, 0, 0, 0, 0],
#                           [  0, 110,   0,   0, 0,   0, 0, 0, 0, 0, 0],
#                           [  0,   0,   0,   0, 0,   0, 0, 0, 0, 0, 0],
#                           [110,   0,   0,   0, 0,   0, 0, 0, 0, 0, 0],
#                           [110,   0,   0, 110, 0,   0, 0, 0, 0, 0, 0],
#                           [  0,   0, 110,   0, 0, 110, 0, 0, 0, 0, 0],
#                           [ 11,   0,   0,  11, 0,   0, 0, 0, 0, 0, 0],
#                           [  0,   0,   0,   0, 4,   0, 0, 0, 4, 0, 0],
#                           [  0,   2,   2,   2, 1,   0, 0, 2, 1, 1, 0]])
# lados_inst = np.array([25.00, 45.60]) # ancho, largo
# max_rel_aspecto = 4
# nom_dpts = ['Recepción', 'Confección', 'Estampado', 'Terminación', 
#             'Administrativa', 'Almacén MP', 'Almacén PT', 'Corte',
#             'Calidad', 'Diseño', 'Descanso']

# solucion = np.array([[6, 1, 7, 4, 9, 11, 8, 3, 2, 10, 5], # best
#                      [0, 0, 0, 0, 0,  1, 0, 0, 0,  0, 1]])

# solucion = np.array([[1, 7, 9, 4, 10, 5, 6, 8, 3, 2, 11], # chosen
#                      [0, 0, 0, 0,  0, 1, 0, 0, 0, 0,  1]])

# O7
# n_dptos = 7
# areas_dptos = np.array([16, 16, 16, 36, 9, 9, 9])
# fl_materiales = np.array([[0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0],
#                           [5, 3, 2, 0, 0, 0, 0],
#                           [0, 0, 0, 4, 0, 0, 0],
#                           [0, 0, 0, 4, 0, 0, 0],
#                           [1, 1, 1, 0, 2, 1, 0]])
# lados_inst = np.array([8.54, 13]) # ancho, largo
# max_rel_aspecto = 4

# solucion = np.array([[3, 5, 7, 1, 4, 6, 2], 
#                      [0, 0, 1, 0, 0, 0, 1]])

data_inst = 'inst_vC10R-a.txt'

# Main()
lobo_gris = AlgLoboGris(archivo_datos=data_inst)

n_iters = 1

while n_iters < 11:

    resultados_gwo = lobo_gris.optimizacion_lobo_gris(tam_manada=15, n_lideres=3, theta_1=0.4, 
                                                      theta_2=0.4, theta_3=0.6, tiempo_lim=600)
    
    plt.plot(resultados_gwo['vals_fitness'], label=f'iter_{n_iters}')

    print(resultados_gwo['vals_fitness'][-1])

    n_iters += 1

plt.show()