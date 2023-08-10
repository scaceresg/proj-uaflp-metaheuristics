import numpy as np
from copy import deepcopy
from time import time
from uaflp import ModeloUAFLP

class AlgLoboGris(ModeloUAFLP):

    # Constructor de la clase (hereda las caracteristicas de la clase ModeloUAFLP)
    def __init__(self, n_dptos: int = None, areas_dptos: np.ndarray = None, flujo_materiales: np.ndarray = None, lados_instalacion: np.ndarray = None, tasa_aspecto_max: int = None, costo_manejo_unit: float = None, nombres_dptos: list = None, mejor_valor: float = None, archivo_datos: str = None) -> None:
        super().__init__(n_dptos, areas_dptos, flujo_materiales, lados_instalacion, tasa_aspecto_max, costo_manejo_unit, nombres_dptos, mejor_valor, archivo_datos)

    # Correr algoritmo de lobo gris
    def optimizacion_lobo_gris(self, tam_manada:int, n_lideres:int, theta_1:float, theta_2:float, theta_3:float=None, tiempo_lim:int=3600):
        
        if tam_manada not in range(n_lideres + 1, 16):
            raise ValueError('Parameter value tam_manada should be greater than n_lideres and lower than 15')
        
        if n_lideres not in {1, 3}:
            raise ValueError('Parameter value n_lideres should be equal to 1 or equal to 3')

        if theta_1 <= 0 or theta_1 >= 1 or theta_2 <= 0 or theta_2 >= 1 or theta_3 <= 0 or theta_3 >= 1:
            raise ValueError('Parameter values theta_1, theta_2 and theta_3 should be greater than 0 and lower than 1')

        if n_lideres == 1 and theta_3 != None:
            raise AttributeError('Parameter theta_3 has no influence when n_lideres is 1 and should be None')

        timeout = time() + tiempo_lim
        lobos_sols = []
        fit_sols = []

        self.generar_manada(tam_manada)
        self.identificar_lideres(n_lideres)
        while time() < timeout:
            
            self.buscar_presa(theta_1)
            self.atacar_presa(n_lideres, theta_2, theta_3)

            lobos_sols.append(self.lobos_lideres[0])
            fit_sols.append(self.fit_lideres[0])

        return {'sols_uaflp': lobos_sols, 'vals_fitness': fit_sols}

    # Inicializar poblacion de la manada
    def generar_manada(self, tam_manada):

        self.manada = [0] * tam_manada
        self.fit_manada = [0] * tam_manada

        for lobo_ind in range(tam_manada):
            lobo_dpts = list(np.random.permutation(self.departamentos))
            lobo_bahs = list(np.random.randint(0, 2, self.n_dptos - 1)) + [1]
            self.manada[lobo_ind] = np.array([lobo_dpts, lobo_bahs])
            self.fit_manada[lobo_ind] = self.calcular_fitness(self.manada[lobo_ind])

    # Identificar lobos lideres ([alpha]) si n_lideres = 1, ([alpha, beta, delta]) si n_lideres = 3
    def identificar_lideres(self, n_lideres):

        self.lobos_lideres = [0] * n_lideres
        self.fit_lideres = [0] * n_lideres

        for ind in range(n_lideres):
            p_min = np.argmin(self.fit_manada)
            self.lobos_lideres[ind] = self.manada[p_min]
            self.fit_lideres[ind] = self.fit_manada[p_min]
            self.manada.pop(p_min)
            self.fit_manada.pop(p_min)

    # Ejecutar comportamiento de busqueda de la presa
    def buscar_presa(self, theta_1):
        
        tam_movimiento = np.floor(self.n_dptos * theta_1).astype(int)
        
        for ind, lobo in enumerate(self.manada):

            lobo_dp = deepcopy(lobo[0])
            lobo_bh = deepcopy(lobo[1])

            # Realizar movimiento en elementos de lobos
            lobo_dp = self.realizar_movimiento(lobo_dp, tam_movimiento)
            lobo_bh = self.realizar_movimiento(lobo_bh, tam_movimiento)

            if lobo_bh[-1] != 1:
                lobo_bh[-1] = 1

            # Calcular funcion fitness del nuevo lobo
            nuevo_lobo = np.array([list(lobo_dp), list(lobo_bh)])
            fit_nuevo_lobo = self.calcular_fitness(nuevo_lobo)

            # Comparar y reemplazar (si aplica) nuevo lobo generado
            if fit_nuevo_lobo < self.fit_lideres[0]:

                self.insertar_nuevo_lider(nuevo_lobo, fit_nuevo_lobo, ind)
                break # Romper busqueda para iniciar ataque

            elif fit_nuevo_lobo < self.fit_manada[ind]:

                # Reemplazar nuevo lobo en manada
                self.manada[ind] = nuevo_lobo
                self.fit_manada[ind] = fit_nuevo_lobo

    # Ejecutar comportamiento de ataque a la presa
    def atacar_presa(self, n_lideres, theta_2, theta_3=None):
        
        da = np.floor(self.n_dptos * theta_2)

        for ind_md, lobo in enumerate(self.manada):

            nuevo_lobo = deepcopy(lobo)
            lider = None
            if n_lideres == 1:
                lider = self.lobos_lideres[0]
            else:
                rand = np.random.random()
                if rand < theta_3:
                    lider = self.lobos_lideres[0]
                elif rand < (theta_3 + (1 - theta_3)/2):
                    lider = self.lobos_lideres[1]
                else:
                    lider = self.lobos_lideres[2]

            # Calcular y reducir distancia con el lider para elemento con dptos
            dist_dptos = self.calcular_dist_lider(nuevo_lobo[0], lider[0])

            ind_dp = 0
            while dist_dptos > da:

                nuevo_lobo[0] = self.reducir_dist_dptos(nuevo_lobo[0], lider[0], ind_dp)
                dist_dptos = self.calcular_dist_lider(nuevo_lobo[0], lider[0])
                ind_dp += 1

            # Calcular y reducir distancia con el lider para elemento de bahias
            dist_bahias = self.calcular_dist_lider(nuevo_lobo[1], lider[1])

            ind_bh = 0
            while dist_bahias > da:

                nuevo_lobo[1] = self.reducir_dist_bahias(nuevo_lobo[1], lider[1], ind_bh)
                dist_bahias = self.calcular_dist_lider(nuevo_lobo[1], lider[1])
                ind_bh += 1

            fit_nuevo_lobo = self.calcular_fitness(nuevo_lobo)

            # Comparar y reemplazar (si aplica) nuevo lobo generado
            if fit_nuevo_lobo < self.fit_lideres[0]:

                 self.insertar_nuevo_lider(nuevo_lobo, fit_nuevo_lobo, ind_md)

            elif fit_nuevo_lobo < self.fit_manada[ind_md]:

                # Reemplazar nuevo lobo en manada
                self.manada[ind_md] = nuevo_lobo
                self.fit_manada[ind_md] = fit_nuevo_lobo

    # Metodo para realizar movimiento en elemento de lobos
    def realizar_movimiento(self, elem_lobo, tam_movimiento):

        pos_inicio, pos_fin = np.random.randint(0, self.n_dptos - tam_movimiento + 1, 2)
        pos_movimiento = np.arange(pos_inicio, pos_inicio + tam_movimiento)
        movimiento = elem_lobo[pos_movimiento]
        elem_lobo = np.delete(elem_lobo, pos_movimiento)
        elem_lobo = np.insert(elem_lobo, pos_fin, movimiento)

        return elem_lobo
    
    # Metodo para reemplazar nuevo lider en lideres y desplazar lobo delta a manada
    def insertar_nuevo_lider(self, nuevo_lobo, fit_nuevo_lobo, ind_manada):
        
        # Agregar nuevo alpha a los lideres
        self.lobos_lideres.insert(0, nuevo_lobo)
        self.fit_lideres.insert(0, fit_nuevo_lobo)

        # Desplazar antiguo lobo delta a la manada
        self.manada[ind_manada] = self.lobos_lideres[-1]
        self.fit_manada[ind_manada] = self.fit_lideres[-1]

        # Eliminar antiguo lobo delta de los lideres
        self.lobos_lideres.pop()
        self.fit_lideres.pop()

    # Metodo para calcular distancia entre lobo en manada y lobo lider
    def calcular_dist_lider(self, elem_lobo, elem_lider):

        distancia = 0
        for ind in range(len(elem_lider)):
            if elem_lobo[ind] != elem_lider[ind]:
                distancia += 1

        return distancia
    
    # Metodo para reducir la distancia entre el lobo en manada y lobo lider (para elemento con dptos)
    def reducir_dist_dptos(self, elem_lobo, elem_lider, ind_actual):

        dpto_lobo = elem_lobo[ind_actual]
        dpto_lider = elem_lider[ind_actual]
        if dpto_lobo != dpto_lider:
            pos_cambio = np.where(elem_lobo == dpto_lider)[0]
            elem_lobo[ind_actual] = dpto_lider
            elem_lobo[pos_cambio] = dpto_lobo

        return elem_lobo
    
    # Metodo para reducir la distancia entre el lobo en manada y lobo lider (para elemento con bahias)
    def reducir_dist_bahias(self, elem_lobo, elem_lider, ind_actual):

        if elem_lobo[ind_actual] != elem_lider[ind_actual]:
            elem_lobo[ind_actual] = elem_lider[ind_actual]

        return elem_lobo