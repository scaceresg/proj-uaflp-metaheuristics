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

################## NO MODIFICAR ##################
lst_parametros = ['tam_manada', 'n_lideres', 'theta_1', 'theta_2', 'theta_3']

################## MODIFICAR AL GUSTO (MANTENER EN LISTAS) ##################
lst_vals_params = [[6, 8, 10, 12, 15], [1, 3], [0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]] 
tiempo_limite = 60 # en segundos (MANTENER ENTERO)
n_rondas = 2 # Numero de veces que el algoritmo corre (MANTENER ENTERO) 

# Crear diccionario para guardar los resultados
resultados = {}

################## MODIFICAR ##################
# Nombre del archivo con la instancia de UAFLP (MANTENER EN CADENA DE TEXTO)
data_inst = 'inst_O9.txt'

lobo_gris = AlgLoboGris(archivo_datos=data_inst)

for param, vals_param in zip(lst_parametros, lst_vals_params):
    
    if len(vals_param) < 2:
        raise AttributeError(f'Deben haber al menos dos valores de parametros para el parametro {param}')
    
    resultados[f'{param}'] = []

    for rnd in range(n_rondas):
        resultados[f'UB_{rnd}'] = []
        resultados[f'sol_{rnd}'] = []

    for v_param in vals_param:
        
        print(f'Corriendo experimentos para el parametro {param} = {v_param}')
        resultados[f'{param}'].append(v_param)

        ronda = 0

        while ronda < n_rondas:

            exp = lobo_gris.correr_experimentos(parametro=param, valor_param=v_param, tmp_lim=tiempo_limite)

            resultados[f'UB_{ronda}'].append(exp['UB'])
            resultados[f'sol_{ronda}'].append(exp['solucion'])

            print(f'Finalizando experimentos para el parametro {param} = {v_param} en la ronda {ronda}')
            ronda += 1

    df = pd.DataFrame(resultados)
    df.set_index(f'{param}', inplace=True)    
    df.to_excel(f'resultados-{data_inst}-{param}.xlsx')

print('Se finalizaron todos los experimentos')