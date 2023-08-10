import os
import sys
import numpy as np
import matplotlib.pyplot as plt

class ModeloUAFLP:
    
    # Constructor de la clase (argumentos requeridos para crear la clase)
    def __init__(self, n_dptos:int=None, areas_dptos:np.ndarray=None, flujo_materiales:np.ndarray=None,
                lados_instalacion:np.ndarray=None, tasa_aspecto_max:int=None, costo_manejo_unit:float=None,
                nombres_dptos:list=None, mejor_valor:float=None, archivo_datos:str=None) -> None:
        
        if archivo_datos == None:
            self.n_dptos = n_dptos
            self.departamentos = np.arange(1, self.n_dptos + 1)
            self.areas_dptos = areas_dptos
            self.flujo_materiales = flujo_materiales
            self.lados_instalacion = lados_instalacion # ancho, largo
            self.tasa_aspecto_max = tasa_aspecto_max
            self.nombres_dptos = nombres_dptos
            self.mejor_val = mejor_valor
            self.costo_manejo_unit = costo_manejo_unit
            
            # Definir matriz de costo de manejo unitario
            if self.costo_manejo_unit == None:
                matriz_costos_dptos = np.ones((self.n_dptos, self.n_dptos))
            else:
                matriz_costos_dptos = np.full((self.n_dptos, self.n_dptos), self.costo_manejo_unit)
        else: # Extraer datos de archivo de datos
            self.obtener_datos(arch_datos=archivo_datos)

    # Método para leer archivo de datos
    def obtener_datos(self, arch_datos:str):

        # Cambiar el directorio
        path = os.path.dirname(os.path.realpath(__file__)) + '\\uaflp-instances'

        try:
            os.chdir(path)
        except FileNotFoundError:
            print(f'Directory: {path} does not exist')
            sys.exit()
        except NotADirectoryError:
            print(f'{path} is not a directory')
            sys.exit()

        # Abrir el archivo de datos
        with open(arch_datos) as f:
            lines = f.readlines()

        # Leer cada linea del archivo
        pr_linea = lines[0].split() # [n_dptos, ancho_inst, largo_inst, tasa_asp, costo_unit, mejor_val]
        seg_linea = lines[1].split() # [areas_dptos 1,..., n]
        ter_linea = lines[2].split() # [nombres_dptos 1,..., n] o ['None']
        fl_mats = [] # lista de listas flujos de materials
        for l in lines[3:]:
            ln = [float(x) for x in l.split()]
            fl_mats.append(ln)

        # Guardar los parametros del modelo UAFLP
        self.n_dptos = int(pr_linea[0])
        self.departamentos = np.arange(1, self.n_dptos + 1)
        self.lados_instalacion = np.array([float(x) for x in pr_linea[1:3]])
        self.tasa_aspecto_max = int(pr_linea[3])
        self.costo_manejo_unit = float(pr_linea[4])
        self.matriz_costos_dptos = np.full((self.n_dptos, self.n_dptos), self.costo_manejo_unit)
        self.mejor_val = float(pr_linea[-1])
        self.areas_dptos = np.array([float(x) for x in seg_linea])

        if len(ter_linea) > 1:
            self.nombres_dptos = [x for x in ter_linea]
        else:
            self.nombres_dptos = None

        self.flujo_materiales = np.array(fl_mats)
        

    # Método para decodificar la solución 
    def decodificar_solucion(self, solucion:np.ndarray):
        self.identificar_bahias(solucion)
        self.calcular_lados_centros()
    
    # Identificar bahías
    def identificar_bahias(self, solucion):
        self.bahias = []
        dpts_bahias = []

        for ind, bah in enumerate(solucion[1]):
            dpto = solucion[0, ind] # Identificar dpto en posicion

            if ind == 0 or bah == 0:
                dpts_bahias.append(dpto) # Añadir dptos en bahía
                if bah == 1:
                    self.bahias.append(dpts_bahias) # Añadir bahía a bahías
                    dpts_bahias = []
            elif bah == 1:
                dpts_bahias.append(dpto) # Añadir dptos en bahía
                self.bahias.append(dpts_bahias) # Añadir bahía a bahías
                dpts_bahias = []
    
    # Calcular dimensiones de lados y centroides de los departamentos
    def calcular_lados_centros(self):
        self.centroides_dptos = [0] * self.n_dptos
        self.lados_dptos = [0] * self.n_dptos

        contador_ancho = 0
        for bah in self.bahias:

            area_bahia = 0
            for dpto in bah:
                area_bahia += self.areas_dptos[dpto-1] # area bahia = suma(areas dptos en bahia)

            ancho_bahia = area_bahia / self.lados_instalacion[1] # ancho bahia = area de bahia / largo de instalacion

            contador_largo = 0
            for dpto in bah:
                largo_dpto = self.areas_dptos[dpto-1] / ancho_bahia # largo dpto = area dpto / ancho bahia
                self.lados_dptos[dpto-1] = [ancho_bahia, largo_dpto] # Nota: ancho bahia == ancho dpto en bahia
                centro_x = contador_ancho + ancho_bahia / 2
                centro_y = contador_largo + largo_dpto / 2
                self.centroides_dptos[dpto-1] = [centro_x, centro_y]
                contador_largo += largo_dpto

            contador_ancho += ancho_bahia

    # Dibujar layout de planta de la solucion
    def dibujar_layout(self, solucion:np.ndarray):
        if self.nombres_dptos == None:
            self.nombres_dptos = [f'Dpto {d}' for d in self.departamentos]

        self.decodificar_solucion(solucion)

        if self.lados_instalacion[0] > self.lados_instalacion[1]:
            fig = plt.figure(dpi=300, figsize=(6, 4))
        else:
            fig = plt.figure(dpi=300, figsize=(4, 6))

        plt.rcParams.update({'font.size': 6})
        ax = fig.add_subplot(111)
        ax.set_xlim([0, self.lados_instalacion[0]])
        ax.set_ylim([0, self.lados_instalacion[1]])
        ax.set_xticks([0, self.lados_instalacion[0]])
        ax.set_yticks([0, self.lados_instalacion[1]])

        esq_x = 0
        for bah in self.bahias:

            esq_y = 0
            for dpt in bah:
                rect = plt.Rectangle((esq_x, esq_y), width=self.lados_dptos[dpt-1][0],
                                     height=self.lados_dptos[dpt-1][1], facecolor='white',
                                     edgecolor='black')
                plt.text(self.centroides_dptos[dpt-1][0], self.centroides_dptos[dpt-1][1],
                         f'{self.nombres_dptos[dpt-1]}', horizontalalignment='center',
                         verticalalignment='top')
                plt.plot(self.centroides_dptos[dpt-1][0], self.centroides_dptos[dpt-1][1],
                         color='black', marker=None, markersize=2)
                ax.add_patch(rect)

                esq_y += self.lados_dptos[dpt-1][1]

            esq_x += self.lados_dptos[dpt-1][0]

        plt.show()

    # Calcular la función fitness
    def calcular_fitness(self, solucion:np.ndarray):
        self.decodificar_solucion(solucion)
        self.calcular_distancias()
        self.calcular_mhc()

        k = 3

        contador_no_factibles = 0
        for i in solucion[0]:
            tasa_aspecto_dpto = max(self.lados_dptos[i-1][0], self.lados_dptos[i-1][1]) / min(self.lados_dptos[i-1][0], self.lados_dptos[i-1][1])

            if tasa_aspecto_dpto > self.tasa_aspecto_max:
                contador_no_factibles += 1

        fitness = self.mhc + self.mhc * (contador_no_factibles ** k)

        return round(fitness, ndigits=2)

    # Calcular las distancias rectilíneas entre departamentos
    def calcular_distancias(self):
        self.distancias_dptos = np.zeros((self.n_dptos, self.n_dptos))

        for i in self.departamentos:
            for j in self.departamentos[i:]:
                dist_rectil_x = np.abs(self.centroides_dptos[i-1][0] - self.centroides_dptos[j-1][0])
                dist_rectil_y = np.abs(self.centroides_dptos[i-1][1] - self.centroides_dptos[j-1][1])

                self.distancias_dptos[j-1, i-1] = dist_rectil_x + dist_rectil_y

    # Calcular el costo total de manejo de materiales entre departamentos
    def calcular_mhc(self):
        self.matriz_costos_dptos = None

        if self.costo_manejo_unit == None:
            self.matriz_costos_dptos = np.ones((self.n_dptos, self.n_dptos))
        else:
            self.matriz_costos_dptos = np.full((self.n_dptos, self.n_dptos), self.costo_manejo_unit)

        self.mhc = np.sum(self.matriz_costos_dptos * self.distancias_dptos * self.flujo_materiales)