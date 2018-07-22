"""
Modulo principal
Creado por: Eder Mauricio Abello
"""
import sys
import fibo
import classes


def main():
    """Funcion Principal"""


def test4():
    """Funcion Principal"""
    item = classes.MyClass()
    item.example()
    print(item.real)
    print(item.imagine)


def test3():
    """Funcion Principal"""
    try:
        number = int(input('Please enter a number'))
        fibo.fib(number)
    except ValueError:
        print('Input is not a number')


def test2():
    """Funcion Principal"""
    if len(sys.argv) < 2:
        print('Usage: script number')
    else:
        number = int(sys.argv[1])
        print('Number value: ', number)

        fibo.fib(number)


def test():
    """test"""
    number = int(input("Please enter an integer: "))

    if number < 0:
        number = 0
        print('Negative changed to zero')
    elif number == 0:
        print('Zero')
    elif number == 1:
        print('Single')
    else:
        print('More')

    words = ['cat', 'window', 'defenestrate']
    for word in words:
        print(word, len(word))

    for number in range(2, 10):
        for ximp in range(2, number):
            if number % ximp == 0:
                print(number, 'equals', ximp, '*', number/ximp)
                break
        else:
            print(number, 'is a prime number')

    fibo.fib(2000)

    f100 = fibo.fib2(100)
    print(f100)


main()
