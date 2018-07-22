import numpy as np

def main():
    print('Manual numpy example')
    print('Creating numpy array')

    x = np.array([2, 1, 3, 4])
    print('Checking type of x')
    print(str(type(x)))

    print('Creating multidimensional array')
    y = np.array([[1, 2, 3], [4, 5, 6]])
    print('Checking type of y')
    print(str(type(y)))

    print('Printing y')
    print(y)

    z = np.array([1, 2, 3, 4, 5, 6])

    print('Printing z shape')
    print(z.shape)

    print('Printing z')
    print(z)


if __name__ == '__main__':
    main()