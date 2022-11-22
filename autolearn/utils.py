import covalent as ct

@ct.electron(executor='local')
def prepare_dict(my_dict):
    return dict(my_dict)
