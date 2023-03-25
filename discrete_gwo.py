import numpy as np
from uaflp import ModeloUAFLP

# Algoritmo de optimizacion de lobo gris discreto
class GWODiscreto(ModeloUAFLP):

    # Constructor
    def __init__(self, n_dptos:int, areas_dptos:np.ndarray, flujo_mats:np.ndarray, lados_inst:np.ndarray, 
                 rel_asp_max:int, costo_unit:float = None, nombres_dptos:list = None, archivo_datos:str = None) -> None:
        super().__init__(n_dptos, areas_dptos, flujo_mats, lados_inst, rel_asp_max, costo_unit, nombres_dptos, archivo_datos)

    # Correr algoritmo DGWO
    def correr_dgwo(self, tam_manada:int, theta_1:float, theta_2:float):
        self.inicializar_manada(tam_manada)
        self.identificar_lideres()
        
        for lobo in self.manada:
            self.buscar_presa(theta_1, lobo)

    # Inicializar la manada
    def inicializar_manada(self, tam_manada):
        self.manada = []
        self.fit_manada = []

        for i in range(tam_manada):
            lobo_dptos = np.random.permutation(self.departamentos)
            lobo_bah = np.random.randint(0, 2, self.n_dptos-1)
            lobo_bah = np.append(lobo_bah, 1)
            lobo = [lobo_dptos, lobo_bah]
            fit_lb = self.calcular_fitness(lobo)

            self.manada.append(lobo)
            self.fit_manada.append(fit_lb)

        return self.manada, self.fit_manada
    
    # Identificar lobos lideres: alpha, beta, delta
    def identificar_lideres(self):
        self.lobos_lideres = [] # alpha, beta, delta
        self.fit_lideres = []

        while len(self.lobos_lideres) < 3:
            
            pos_min = np.argmin(self.fit_manada)
            lider = self.manada[pos_min]
            fit_lider = self.fit_manada[pos_min]
            self.manada.pop(pos_min)
            self.fit_manada.pop(pos_min)

            self.lobos_lideres.append(lider)
            self.fit_lideres.append(fit_lider)

        return self.lobos_lideres, self.fit_lideres

    # Busqueda de la presa
    def buscar_presa(self, theta_1, lobo):

        tam_mov = np.floor(self.n_dptos * theta_1).astype(int)
        for elem in lobo:
            pos_inicial = np.random.randint(0, self.n_dptos)

        
        # corte = np.random.randint(0, self.n_dptos + 1 - movimiento)
        # posiciones = np.arange(corte, corte + movimiento)

        # for ind, lobo in enumerate(self.manada):

        #     # Individuo departamentos
        #     seccion = lobo[posiciones[0]:posiciones[-1]+1]
        #     lobo = np.delete(lobo, posiciones)

        #     num_rand = np.random.random(size=1)

        #     if corte < self.n_dptos/2:
        #         lobo = np.append(lobo, seccion)
        #     else:
        #         lobo = np.append(seccion, lobo)

        #     self.manada.pop(ind)
        #     self.fit_manada.pop(ind)
        #     self.manada.append(lobo)
        #     self.fit_manada.append(self.calcular_fitness(lobo))

            # Sebas, estoy en el metodo de buscar_presa. 
            # Me encontre con el problema que `lobo in manada` es una lista de dos elementos [[1, 2, 3..], [0, 1, 0,...1]]
            # Necesito identificar un metodo para realizar el movimiento para ambos elementos, sobretodo el de [0, 1, 0, ...1]