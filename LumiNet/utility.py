import pickle

def SaveObj(obj, folder, name):
    if '.pkl' in name:
        with open(folder + name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(folder + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def LoadObj(folder, name):
    if '.pkl' in name:
        with open(folder + name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(folder + name + '.pkl', 'rb') as f:
            return pickle.load(f)
def recur_items(dictionary,nest):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key,nest)
            yield from recur_items(value,nest+1)
        else:
            yield (key,nest)
def print_dic(dic,nest=0):
    for key,nest in recur_items(dic,0):
        print("\t"*nest,key)
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
