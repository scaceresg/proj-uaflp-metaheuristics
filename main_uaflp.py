import os
import sys
import pandas as pd
from alg_lobo_gris import AlgLoboGris
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(base_dir, 'uaflp-instances')

try:
    os.chdir(path)
except FileNotFoundError:
    print(f'Directory: {path} does not exist')
    sys.exit()
except NotADirectoryError:
    print(f'{path} is not a directory')
    sys.exit()

# Valores de los parámetros
val_mnd = 10 # Posibles valores: [6, 8, 10, 12, 15]
val_ld = 3 # Posibles valores: [1, 3]
val_t1 = 0.4 # Posibles valores: [0.2, 0.4, 0.6, 0.8]
val_t2 = 0.4 # Posibles valores: [0.2, 0.4, 0.6, 0.8]
val_t3 = 0.4 # Posibles valores: [0.2, 0.4, 0.6, 0.8]
t_lim = 300 # en segundos
n_rondas = 10 # Numero de veces que el algoritmo corre

# Crear diccionario para guardar los resultados
resultados = {'inst': [],
              'n_dptos': []}

fig = plt.figure(dpi=300)

data_inst = 'inst_O9.txt'

lobo_gris = AlgLoboGris(archivo_datos=data_inst)

rond = 0

while rond < n_rondas:

    resultados[f'UB_{rond}'] = []
    resultados[f'sol_{rond}'] = []

    res = lobo_gris.optimizacion_lobo_gris(tam_manada=val_mnd, n_lideres=val_ld, theta_1=val_t1,
                                           theta_2=val_t2, theta_3=val_t3, tiempo_lim=t_lim)
    
    plt.plot(res['vals_fitness'], label=f'ronda: {rond}')

    ub = res['vals_fitness'][-1]
    sol = res['sols_uaflp'][-1]

    resultados[f'UB_{rond}'].append(ub)
    resultados[f'sol_{rond}'].append(sol)

    print(f'El algoritmo finaliza la ronda {rond} para la inst {data_inst} con UB = {ub}')

    rond += 1    

df = pd.DataFrame(res)
df.set_index('inst', inplace=True)

print(df.head())

plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función fitness')
plt.title('Evolución del valor de la función fitness por ronda')
plt.legend()

plt.show()