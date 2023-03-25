import numpy as np

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
departamentos = np.arange(1, n_dptos + 1)

def fitness_rand():
    fitness = np.random.randint(20, 100)

    return fitness

# Discrete GWO
tam_manada, teta_1, teta_2 = 10, 0.3, 0.5

# Creacion de la manada
manada = []
for i in range(tam_manada):
    lobo = np.random.permutation(departamentos)
    manada.append(lobo)

# Calcular fitness de la manada
fit_manada = []
for lobo in manada:
    fit_manada.append(fitness_rand())

# Identificar lobos lideres
lobos_lideres = [] # alpha, beta, delta
fit_lideres = []

while len(lobos_lideres) < 3:

    pos_min = np.argmin(fit_manada)

    lider = manada[pos_min]
    fit_lider = fit_manada[pos_min]
    manada.pop(pos_min)
    fit_manada.pop(pos_min)

    lobos_lideres.append(lider)
    fit_lideres.append(fit_lider)

# Busqueda de la presa
movimiento = np.ceil(n_dptos * teta_1).astype(int)

corte = np.random.randint(0, n_dptos + 1 - movimiento)
posiciones = np.arange(corte, corte + movimiento)

for ind, lobo in enumerate(manada):
    seccion = lobo[posiciones[0]:posiciones[-1]+1]
    lobo = np.delete(lobo, posiciones)

    if corte < n_dptos/2:
        lobo = np.append(lobo, seccion)
    else:
        lobo = np.append(seccion, lobo)

    manada.pop(ind)
    fit_manada.pop(ind)

    manada.append(lobo)
    fit_manada.append(fitness_rand())

    if fit_manada[-1] < fit_lideres[0]: # Creo que no es necesario para busqueda pero si en ataque
        break

# Ataque a la presa
dist_alpha = n_dptos * teta_2
alpha = lobos_lideres[0]
alpha_fit = fit_lideres[0]

beta = lobos_lideres[1]
beta_fit = fit_lideres[1]

delta = lobos_lideres[-1]
delta_fit = fit_lideres[-1]

for ind, lob in enumerate(manada):

    pass


print(manada)
print(fit_manada)
    

