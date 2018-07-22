"""
Funciones Fibonacdci
Creado por: Eder Mauricio Abello
"""


def fib(number):
    """Fibonacci function"""
    var1, var2 = 0, 1
    while var1 < number:
        print(var1,)
        var1, var2 = var2, var1 + var2


def fib2(number):
    """Return a list containing"""
    result = []
    var1, var2 = 0, 1
    while var1 < number:
        result.append(var1)
        var1, var2 = var2, var1+var2
    return result
