def dict_upper(dict_arg):
    for verb in dict_arg.items():
        if not all([isinstance(i, str) for i in verb]):
            print('Unable to upper Dictionary. Non-str entry.')
            return dict_arg
    return dict((k.upper(), v.upper()) for k,v in dict_arg.items())

def list_upper(list_arg):
    for e in list_arg:
        if not isinstance(e, str):
            print('Unable to upper List. Non-str element.')
            return list_arg
    return [e.upper() for e in list_arg]

def str_upper(str_arg):
    return str_arg.upper()

dict_func = {'list': list_upper,
             'dict': dict_upper,
             'str' : str_upper
}

support_upper = ['str', 'dict', 'list']