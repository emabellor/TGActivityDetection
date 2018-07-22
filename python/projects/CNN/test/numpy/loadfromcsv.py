import numpy as np
from datetime import datetime

str2date = lambda x: datetime.strptime(x.decode("utf-8"), '%Y-%m-%d')

def main():
    print('Init main function')
    path = '/home/mauricio/CSV/yahoo.csv'
    array = np.genfromtxt(path, delimiter=',', skip_header=1, converters={0: str2date})  # Lines to skip in header!
    print(array)
    print('Done!')


if __name__ == '__main__':
    main()
