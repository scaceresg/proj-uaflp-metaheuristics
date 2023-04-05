import numpy as np
import matplotlib.pyplot as plt

class ModeloUAFLP:

    # Constructor
    def __init__(self, n_dptos:int, areas_dptos:np.ndarray, flujo_mats:np.ndarray, 
                 lados_inst:np.ndarray,  rel_asp_max:int, costo_unit:float=None, 
                 nombres_dptos:list=None, archivo_datos:str=None) -> None:
        if archivo_datos == None:
            self.n_dptos = n_dptos
            self.departamentos = np.arange(1, self.n_dptos + 1)
            self.areas_dptos = areas_dptos
            self.flujo_mats = flujo_mats
            self.lados_inst = lados_inst
            self.rel_asp_max = rel_asp_max
            self.nombres_dptos = nombres_dptos
            self.costo_unit = costo_unit
            if self.costo_unit == None:
                self.costo_unit = np.ones((self.n_dptos, self.n_dptos))
            else:
                self.costo_unit = np.full((self.n_dptos, self.n_dptos), self.costo_unit)
        else:
            pass

    # Decodificar la solucion: sol = [[1, 2, 3], [0, 1, 1]] -> Estructura de bahias flexibles
    def decodificar_sol(self, sol):

        self.bahias = self.identificar_bahias(sol)
        self.lados_dptos, self.centros_dptos = self.calcular_dim_dptos()
        
        return self.lados_dptos, self.centros_dptos

    # Identificar los departamentos dentro de cada bahia
    def identificar_bahias(self, sol):
        bahias = []
        dpts_bahias = []

        for ind, bah in enumerate(sol[1]):
            dpto = sol[0, ind]
            if ind == 0 or bah == 0:
                dpts_bahias.append(dpto)
                if bah == 1:
                    bahias.append(dpts_bahias)
                    dpts_bahias = []
            elif bah == 1:
                dpts_bahias.append(dpto)
                bahias.append(dpts_bahias)
                dpts_bahias = []

        return bahias
    
    # Calcular dimensiones y centroides de departamentos
    def calcular_dim_dptos(self):
        lados_dptos = []
        centros_dptos = []

        cont_ancho = 0
        for bah in self.bahias:
            dim_dpto = []
            x_y_dpto = []
            area_bahia = 0

            for dp in bah:
                area_bahia += self.areas_dptos[dp-1]

            ancho_bahia = area_bahia / self.lados_inst[1]
            
            cont_largo = 0
            for dp in bah:
                largo_dpto = self.areas_dptos[dp-1] / ancho_bahia
                dim_dpto.append([ancho_bahia, largo_dpto])
                centro_x = cont_ancho + ancho_bahia / 2
                centro_y = cont_largo + largo_dpto / 2
                x_y_dpto.append([centro_x, centro_y])
                cont_largo += largo_dpto
            
            lados_dptos.append(dim_dpto)
            centros_dptos.append(x_y_dpto)
            cont_ancho += ancho_bahia
        
        return lados_dptos, centros_dptos
    
    # Calcular costo de manejo de materials = sum(costos_unit * flujo_mats * distancias)
    def calcular_mhc(self, sol):

        self.lados_dptos, self.centros_dptos = self.decodificar_sol(sol)

        # Aplanar lista de centroides
        dptos = sol[0]
        centros = [0] * self.n_dptos
        cnts = [cn for dpts in self.centros_dptos for cn in dpts]

        for cn, dp in zip(cnts, dptos):
            centros[dp-1] = cn
        
        rect_distancias = self.calcular_distancias(centros)

        mhc = np.sum(self.costo_unit * rect_distancias * self.flujo_mats)
        
        return mhc

    # Calcular las distancias rectilineas entre departamentos
    def calcular_distancias(self, centroides):
        
        distancias = np.zeros((self.n_dptos, self.n_dptos))

        for i in self.departamentos:
            for j in self.departamentos[i:]:
                rect_dist_x = np.abs(centroides[i-1][0] - centroides[j-1][0])
                rect_dist_y = np.abs(centroides[i-1][1] - centroides[j-1][1])

                distancias[j-1, i-1] = rect_dist_x + rect_dist_y

        return distancias

    # Calcular fitness: fitness = MHC + Dinf^k * MHC
    def calcular_fitness(self, sol):
        
        costo_total_manejo = self.calcular_mhc(sol)
        k_param = 3 # Sugerido por Tate and Smith

        # Aplanar lista de lados
        dptos = sol[0]
        lados = [0] * self.n_dptos
        lds = [ld for dpts in self.lados_dptos for ld in dpts]

        for ld, dp in zip(lds, dptos):
            lados[dp-1] = ld

        no_factibles = 0
        for i in dptos:
            tasa_aspecto = max(lados[i-1][0], lados[i-1][1])/min(lados[i-1][0], lados[i-1][1])

            if tasa_aspecto > self.rel_asp_max:
                no_factibles += 1

        fitness = costo_total_manejo + costo_total_manejo * (no_factibles ** k_param)

        return round(fitness, ndigits=2)
    
    def mostrar_layout(self, sol):
        
        if self.nombres_dptos == None:
            self.nombres_dptos = []
            for d in range(1, self.n_dptos + 1):
                self.nombres_dptos.append(f'Dpto_{d}')

        self.lados_dptos, self.centros_dptos = self.decodificar_sol(sol)

        if self.lados_inst[0] > self.lados_inst[1]:
            fig = plt.figure(dpi=300, figsize=(6, 4))
        else:
            fig = plt.figure(dpi=300, figsize=(4, 6))

        plt.rcParams.update({'font.size': 6})
        ax = fig.add_subplot(111)
        ax.set_xlim([0, self.lados_inst[0]])
        ax.set_ylim([0, self.lados_inst[1]])
        ax.set_xticks([0, self.lados_inst[0]])
        ax.set_yticks([0, self.lados_inst[1]])

        esq_x = 0
        for ind, bah in enumerate(self.bahias):
            lds_bah = self.lados_dptos[ind]
            cnt_bah = self.centros_dptos[ind]

            esq_y = 0
            for inx, dpt in enumerate(bah):
                rect = plt.Rectangle((esq_x, esq_y), width=lds_bah[inx][0],
                                     height=lds_bah[inx][1], facecolor='white',
                                     edgecolor='black')
                plt.text(cnt_bah[inx][0], cnt_bah[inx][1], f'{self.nombres_dptos[dpt-1]}',
                         horizontalalignment='center', verticalalignment='top')
                plt.plot(cnt_bah[inx][0], cnt_bah[inx][1], color='black', marker=None,
                         markersize=2)
                ax.add_patch(rect)

                esq_y += lds_bah[inx][1]

            esq_x += lds_bah[inx][0]

        plt.show()