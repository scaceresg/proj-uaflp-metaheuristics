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
        
        
        # Construir el cuerpo del algoritmo: iteraciones o no_mejoras despues de tantas veces
        self.inicializar_manada(tam_manada)
        self.identificar_lideres()
        self.buscar_presa(theta_1)
        self.atacar_presa(theta_2)
        self.reunir_lideres()

        # Construir return
        # p_min = np.argmin(self.fit_lideres)
        # return self.manada[p_min], self.fit_manada[p_min]


    # Inicializar la manada
    def inicializar_manada(self, tam_manada):
        self.manada = []
        self.fit_manada = []

        for i in range(tam_manada):
            lobo_dptos = np.random.permutation(self.departamentos)
            lobo_bah = np.random.randint(0, 2, self.n_dptos-1)
            lobo_bah = np.append(lobo_bah, 1)
            lobo = np.array([list(lobo_dptos), list(lobo_bah)])
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

    # Comportamiento de busqueda de la presa
    def buscar_presa(self, theta_1):
        
        # Tamaño del movimiento a realizar: ej. 2 posiciones
        tam_mov = np.floor(self.n_dptos * theta_1).astype(int)
        
        for ind, lobo in enumerate(self.manada):
            
            # Encontrar índices del movimiento
            pos_mov, pos_fin = self.encontrar_pos(tam_mov)

            # Realizar movimiento para cada elemento de lobo
            lobo_dp = lobo[0]
            lobo_dp = self.realizar_movimiento(lobo_dp, pos_mov, pos_fin)

            lobo_bh = lobo[1]
            lobo_bh = self.realizar_movimiento(lobo_bh, pos_mov, pos_fin)
            if lobo_bh[-1] == 0:
                lobo_bh[-1] = 1 # El elemento de bahías debe terminar en 1
            
            nueva_pos_lobo = np.array([list(lobo_dp), list(lobo_bh)])
            nueva_pos_fit = self.calcular_fitness(nueva_pos_lobo)

            # Comparar lobo con alpha
            if nueva_pos_fit < min(self.fit_lideres):
                self.manada[ind] = self.lobos_lideres[0] # Reemplazar lobo con el alpha en manada
                self.fit_manada[ind] = self.fit_lideres[0]

                self.lobos_lideres[0] = nueva_pos_lobo # Reemplazar alpha con nuevo lobo
                self.fit_lideres[0] = nueva_pos_fit
                break # Salirse del for
            elif nueva_pos_fit < self.fit_manada[ind]:
                self.manada[ind] = nueva_pos_lobo
                self.fit_manada[ind] = nueva_pos_fit

    # Encontrar posiciones de movimiento
    def encontrar_pos(self, tam_movimiento):
        
        pos_inicial, pos_final = np.random.randint(0, self.n_dptos + 1 - tam_movimiento, 2) # Índices inicial (ej. 1) y donde se ubicará el nuevo movimiento (ej. 5)
        pos_movimiento = np.arange(pos_inicial, pos_inicial + tam_movimiento) # Índices del movimiento (ej. [1, 2])

        return pos_movimiento, pos_final

    # Realizar movimiento en el proceso de búsqueda
    def realizar_movimiento(self, elem_lobo, pos_movimiento, pos_final):
        
        movimiento = elem_lobo[pos_movimiento]
        elem_lobo = np.delete(elem_lobo, pos_movimiento)
        elem_lobo = np.insert(elem_lobo, pos_final, movimiento)

        return elem_lobo
    
    # Comportamiento de ataque de la presa
    def atacar_presa(self, theta_2):

        da = np.floor(self.n_dptos * theta_2)

        for ind_md, lobo in enumerate(self.manada):
            
            nuevo_atq_lobo = []
            
            # Seleccionar lobo lider: alpha, beta o delta
            lider = self.seleccionar_lider()

            # Calcular distancia y acercar a presa (lobo lider)
            for ind, elem in enumerate(lobo):
                distancia = self.calcular_dist(elem, lider[ind])
                i = 0
                while distancia > da:
                    elem, i = self.reducir_dist(elem, lider[ind], i)
                    distancia = self.calcular_dist(elem, lider[ind])
                
                nuevo_atq_lobo.append(list(elem))

            nuevo_atq_lobo = np.array(nuevo_atq_lobo)
            nuevo_atq_fit = self.calcular_fitness(nuevo_atq_lobo)

            # Si fitness es mejor que lobo actual: actualizar
            if nuevo_atq_fit < self.fit_manada[ind_md]:
                self.manada[ind_md] = nuevo_atq_lobo
                self.fit_manada[ind_md] = nuevo_atq_fit
        
        return self.manada, self.fit_manada

    # Seleccionar lider para calcular distancia
    def seleccionar_lider(self):

        rand = np.random.random()
        if rand < 0.5:
            lider = self.lobos_lideres[0] # Calcular con alpha (50% prob)
        elif rand < 0.75:
            lider = self.lobos_lideres[1] # Calcular con beta (25% prob)
        else:
            lider = self.lobos_lideres[2] # Calcular con delta (25% prob)
        
        return lider
     
    # Calcular distancia entre elementos de lobos
    def calcular_dist(self, elem_lobo, elem_lider):

        dist = 0
        for ind in range(len(elem_lider)):
            if elem_lider[ind] != elem_lobo[ind]:
                dist += 1

        return dist
    
    # Reducir distancia entre elementos de lobos
    def reducir_dist(self, elem_lobo, elem_lider, index):

        for i in self.departamentos[index:]:
            lider_val = elem_lider[i-1]
            lobo_val = elem_lobo[i-1]
            if lider_val != lobo_val:
                pos_cambio = np.where(elem_lobo == lider_val)
                elem_lobo[i-1] = lider_val
                elem_lobo[pos_cambio] = lobo_val

                return elem_lobo, i
            
    # Reunir lideres a la manada para reiniciar iteracion
    def reunir_lideres(self):
        
        while len(self.lobos_lideres) > 0:
            
            pos_min = np.argmin(self.fit_lideres)
            self.manada.append(self.lobos_lideres[pos_min])
            self.fit_manada.append(self.fit_lideres[pos_min])
        
        return self.manada, self.fit_manada
