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

dptos = np.arange(1, n_dptos + 1)

def fitness_rand():
    fitness = np.random.randint(20, 100)

    return fitness

# Parametros del algoritmo
M = 10
theta_1 = 0.3
theta_2 = 0.5

# Iniciar manada
manada = []
fit_manada = []

for i in range(M):
    lobo_dp = np.random.permutation(dptos) # Componente permutacion - departamentos
    lobo_bh = np.random.randint(0, 2, n_dptos-1) # Componente binario - bahias
    lobo_bh = np.append(lobo_bh, 1) # Bahias debe terminar en 1
    lobo = [lobo_dp, lobo_bh]
    fit_lb = np.random.randint(20, 100) # metodo calcular_fitness()
    
    manada.append(lobo)
    fit_manada.append(fit_lb)

print(manada)
print(fit_manada)

# Identificar lideres
lobos_lideres = [] # alpha, beta, delta
fit_lideres = []

while len(lobos_lideres) < 3:
    pos_min = np.argmin(fit_manada) # Identificar el lobo con min fitness
    lider = manada[pos_min]
    fit_lider = fit_manada[pos_min]
    manada.pop(pos_min) # Sacarlo de la manada
    fit_manada.pop(pos_min)
    
    lobos_lideres.append(lider) # Anadir a los lideres
    fit_lideres.append(fit_lider)

print(lobos_lideres)
print(fit_lideres)

# Realizar busqueda
tam_mov = np.floor(n_dptos * theta_1).astype(int) # Tamano del movimiento a realizar (e.g. 2 posiciones)

for ind, lobo in enumerate(manada):
    pos_ini = np.random.randint(0, n_dptos + 1 - tam_mov) # Indice inicial del movimiento (e.g. indice 1)
    pos_fin = np.random.randint(0, n_dptos + 1 - tam_mov) # Indice en donde se movera la posicion (e.g. indice 4)
    pos_mov = np.arange(pos_ini, pos_ini + tam_mov) # Indices del movimiento a realizar (e.g. [1, 2])
    
    # Movimiento para el elemento de departamentos
    dp = lobo[0]
    mov_dp = dp[pos_mov]
    dp = np.delete(dp, pos_mov)
    dp = np.insert(dp, pos_fin, mov_dp)
    
    # Movimiento para el elemento de bahias
    bh = lobo[1]
    mov_bh = bh[pos_mov]
    bh = np.delete(bh, pos_mov)
    bh = np.insert(bh, pos_fin, mov_bh)
    if bh[-1] == 0:
        bh[-1] = 1
    
    new_pos_lb = [dp, bh]
    new_pos_fit = np.random.randint(20, 100) # metodo calcular_fitness()
    
    if new_pos_fit < min(fit_lideres): # Comparar con alpha
        manada[ind] = lobos_lideres[0] # anadir alpha a la manada
        fit_manada[ind] = fit_lideres[0]
        
        lobos_lideres[0] = new_pos_lb
        fit_lideres[0] = new_pos_fit
        break # ?
    elif new_pos_fit < fit_manada[ind]: # Comparar con posicion actual
        manada[ind] = new_pos_lb
        fit_manada[ind] = new_pos_fit

print('===')
print(fit_lideres)
print(lobos_lideres)
print('===')
print(manada)
print(fit_manada)

# Realizar ataque
da = np.floor(n_dptos * theta_2)

def seleccionar_lider():

    rand = np.random.random()
    if rand < 0.5:
        lider = lobos_lideres[0] # alpha
    elif rand < 0.75:
        lider = lobos_lideres[1] # beta
    else:
        lider = lobos_lideres[2] # delta
    
    return lider

def calcular_dist(elem_lobo, elem_lider):
    
    dist = 0
    for ind in range(len(elem_lider)):
        if elem_lider[ind] != elem_lobo[ind]:
            dist += 1
    
    return dist

def reducir_dist(elem_lobo, elem_lider, index):
    
    dp = np.arange(1, n_dptos+1)
    
    for i in dp[index:]:
        lid_val = elem_lider[i-1]
        lb_val = elem_lobo[i-1]
        if lid_val != lb_val:
            p_cambio = np.where(elem_lobo == lid_val)
            elem_lobo[i-1] = lid_val
            elem_lobo[p_cambio] = lb_val
            
            return elem_lobo, i
    
# Prueba de los metodos
lb1 = np.array([4, 6, 5, 7, 3, 2, 1])
lid = seleccionar_lider()
distancia = calcular_dist(lb1, lid[0])
lb, ind = reducir_dist(lb1, lid[0], 0)
print(lb)