"""Creado Por
Eder Mauricio Abello Rodriguez"""


class Suma(object):
    """Clase generadora"""

    def __init__(self):
        pass

    @staticmethod
    def suma_var(var1, var2):
        """Suma entre dos variables"""
        print('Sumando variables')
        result = var1 + var2
        return result

    @staticmethod
    def resta_var(var1, var2):
        """Resta entre dos variables"""
        print('Restando variables')
        result = var1 - var2
        return result
