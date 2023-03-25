import numpy as np
from uaflp import ModeloUAFLP

# n_dptos = 11
# areas = np.array([50.00, 195.50, 83.27, 100.39, 67.20, 153.00, 
#                   153.00, 131.25, 52.90, 52.90, 100.00])
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

# O7
n_dptos = 7
areas = np.array([16, 16, 16, 36, 9, 9, 9])
fl_materiales = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [5, 3, 2, 0, 0, 0, 0],
                          [0, 0, 0, 4, 0, 0, 0],
                          [0, 0, 0, 4, 0, 0, 0],
                          [1, 1, 1, 0, 2, 1, 0]])
lados_inst = np.array([8.54, 13]) # ancho, largo
max_rel_aspecto = 4


prob = ModeloUAFLP(n_dptos=n_dptos, areas_dptos=areas, flujo_mats=fl_materiales, 
                   lados_inst=lados_inst, rel_asp_max=max_rel_aspecto)

solucion = np.array([[3, 5, 7, 1, 4, 6, 2], 
                     [0, 0, 1, 0, 0, 0, 1]])

fitness = prob.calcular_fitness(solucion)

print(fitness)

prob.mostrar_layout(solucion)