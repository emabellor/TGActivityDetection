import numpy as np


def main():
    print('Initializing main function')
    print('Int to bytes')

    elem = 213541200214
    bytes = elem.to_bytes(length=8, byteorder='little')
    print(elem.to_bytes(length=8, byteorder='little'))

    array = [0xAE, 0x1B, 0x99, 0xBE, 0x1C, 0x00, 0x00, 0x00]
    num = int.from_bytes(bytes, byteorder='little')
    print(num)


if __name__ == '__main__':
    main()
