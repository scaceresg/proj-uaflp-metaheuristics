import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
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

with open('uaflp_instances-case.txt') as f:
    data = f.readlines()

# Set of parameters to evaluate
set_params = [[15, 1, 0.4, 0.2, 1], [15, 3, 0.2, 0.4, 0.6]]

t_limite = 600
n_rondas = 10

for ln in data:

    inst = ln.split()[0]

    lobo_gris = AlgLoboGris(archivo_datos=inst)
    
    resultados = {}
    resultados['set_params'] = []

    for rnd in range(n_rondas):
        resultados[f'UB_{rnd}'] = []
        resultados[f'SOL_{rnd}'] = []

    for ind, params in enumerate(set_params):
        
        fig = plt.figure(dpi=400)

        print(f'Corriendo set de parámetros {ind} = {params}')
        resultados[f'set_params'].append(f'set_params_{ind}')

        t_manada, n_lideres, theta_1, theta_2, theta_3 = params
        
        rnd = 0
        while rnd < n_rondas:

            lobo_res = lobo_gris.optimizacion_lobo_gris(tam_manada=t_manada, n_lideres=n_lideres, theta_1=theta_1,
                                                        theta_2=theta_2, theta_3=theta_3, tiempo_lim=t_limite)
            
            plt.plot(lobo_res['vals_fitness'], label=f'round: {rnd}')

            resultados[f'UB_{rnd}'].append(lobo_res['vals_fitness'][-1])
            resultados[f'SOL_{rnd}'].append(lobo_res['sols_uaflp'][-1])

            print(f'Finalizando corrida de set de parámetros {ind} para la instancia {inst} en la ronda {rnd}')

            rnd += 1

        plt.xlabel('Iterations')
        plt.ylabel('Fitness function value')
        plt.title('Fitness function value per round')
        plt.legend()

        plt.savefig(f'plot-application-set-params-{ind}.png')

    df = pd.DataFrame(resultados)
    df.set_index('set_params', inplace=True)    
    df.to_excel(f'resultados-{inst}.xlsx')

print('Se finalizaron todos los experimentos')