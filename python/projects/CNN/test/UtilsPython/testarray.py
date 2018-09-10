def main():
    print('Initializing main function')

    # Initialize array - 10 elems
    array = [0 for x in range(10)]

    # Set zero index
    array[0] = 0

    # Set custom index
    array[2] = 0

    # Add value to custom index
    array[4] += 1

    print('Array: {0}'.format(array))

    # Array division
    # Divide all elements by one number
    array = [x/4 for x in array]
    print('New array: {0}'.format(array))

    print('Done!')


if __name__ == '__main__':
    main()
