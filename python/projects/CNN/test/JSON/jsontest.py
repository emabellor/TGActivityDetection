import json


def main():
    print('Initializing main function')
    a = 3
    b = '1234'

    json_string = json.dumps({'a': a, 'b': b})
    print(json_string)

    print('Len JSON string')
    print(len(json_string))

    print('Generating object')
    json_obj = {'a': a, 'b': b}  # python dictionary -> Could be useful in some elements!

    print(json_obj)
    print('Done!')


if __name__ == '__main__':
    main()
