import numpy as np
from uaflp import ModeloUAFLP
from discrete_gwo import GWODiscreto

# Sportswear manuf
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

# solucion = np.array([[6, 1, 7, 4, 9, 11, 8, 3, 2, 10, 5], # best
#                      [0, 0, 0, 0, 0,  1, 0, 0, 0,  0, 1]])

# solucion = np.array([[1, 7, 9, 4, 10, 5, 6, 8, 3, 2, 11], # chosen
#                      [0, 0, 0, 0,  0, 1, 0, 0, 0, 0,  1]])

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

solucion = np.array([[3, 5, 7, 1, 4, 6, 2], 
                     [0, 0, 1, 0, 0, 0, 1]])


# prob = ModeloUAFLP(n_dptos=n_dptos, areas_dptos=areas, flujo_mats=fl_materiales, 
#                    lados_inst=lados_inst, rel_asp_max=max_rel_aspecto)

# fitness = prob.calcular_fitness(solucion)
# print(fitness)
# prob.mostrar_layout(solucion)

lobo_gris = GWODiscreto(n_dptos=n_dptos, areas_dptos=areas, flujo_mats=fl_materiales, 
                   lados_inst=lados_inst, rel_asp_max=max_rel_aspecto)

print(lobo_gris.calcular_fitness(solucion))

lobo_gris.generar_manada(10)

lobo_gris.identificar_lideres()

lobo_gris.buscar_presa(theta_1=0.2)
lobo_gris.atacar_presa(theta_2=0.4)

print(lobo_gris.fit_lideres)

