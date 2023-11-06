import os
import sys
import pandas as pd
from alg_lobo_gris import AlgLoboGris

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

with open('uaflp_instances.txt') as f:
    data = f.readlines()

# Set of parameters to evaluate
set_params = [[10, 1, 0.4, 0.2, 1], [10, 3, 0.6, 0.4, 0.4], [15, 1, 0.4, 0.2, 1],
              [15, 3, 0.6, 0.4, 0.6], [15, 3, 0.4, 0.2, 0.4], [15, 3, 0.2, 0.4, 0.6]]

t_limite = 600
n_rondas = 10

for ln in data:

    inst = ln.split()[0]

    lobo_gris = AlgLoboGris(archivo_datos=inst)

    resultados = {}
    resultados['set_params'] = []

    for rnd in range(n_rondas):
        resultados[f'UB_{rnd}'] = []

    for ind, params in enumerate(set_params):
        
        print(f'Corriendo set de parámetros {ind} = {params}')
        resultados[f'set_params'].append(f'set_params_{ind}')

        t_manada, n_lideres, theta_1, theta_2, theta_3 = params
        
        rnd = 0
        while rnd < n_rondas:

            lobo_res = lobo_gris.optimizacion_lobo_gris(tam_manada=t_manada, n_lideres=n_lideres, theta_1=theta_1,
                                                        theta_2=theta_2, theta_3=theta_3, tiempo_lim=t_limite)

            resultados[f'UB_{rnd}'].append(lobo_res['vals_fitness'][-1])

            print(f'Finalizando corrida de set de parámetros {ind} para la instancia {inst} en la ronda {rnd}')

            rnd += 1

    df = pd.DataFrame(resultados)
    df.set_index('set_params', inplace=True)    
    df.to_excel(f'resultados-{inst}-set-params.xlsx')

print('Se finalizaron todos los experimentos')