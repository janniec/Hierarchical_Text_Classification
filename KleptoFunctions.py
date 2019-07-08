import klepto
import os
import pandas as pd
import numpy as np


def chewing_file(num_sections, data, filename, foldername): 
    '''
    LIST, DATAFRAMES, & DICTONARIES ONLY.
    When 'swallowing' a pickle causes your notebook to choke, chew it up into batches with klepto.
    batches object into num sections and saves batchs into folders within foldername
    each folder will be named accordingly, "K_"+filename+"_1", "K_"+filename+"_2", ...
    '''
    d = klepto.archives.dir_archive('%s' % foldername, cached=True, serialized=True)
    sections = int(len(data)/num_sections)
    filterByKey = lambda keys: {x: data[x] for x in keys} #for dictionary
    
    if type(data) == list:
        for num in range(num_sections-1):
            new_name = filename + "_%s" % (num+1)
            d[new_name] = data[(num*sections):((num+1)*sections)]
        d[filename + "_%s" % (num_sections)] = data[(num_sections-1)*sections:]
    elif type(data) == pd.DataFrame:
        for num in range(num_sections-1):
            new_name = filename + "_%s" % (num+1)
            d[new_name] = data.iloc[(num*sections):((num+1)*sections)]
        d[filename + "_%s" % (num_sections)] = data.iloc[(num_sections-1)*sections:]
    elif type(data) == dict:
        for num in range(num_sections-1):
            new_name = filename + "_%s" % (num+1)
            subkeys = sorted(list(data.keys()))[(num*sections):((num+1)*sections)]
            d[new_name] = filterByKey(subkeys)
        subkeys = sorted(list(data.keys()))[(num_sections-1)*sections:]
        d[filename + "_%s" % (num_sections)] = filterByKey(subkeys) 
    else: 
        d[filename] = data
        print("data is a", type(data))
        
    d.dump()
    d.clear()
    
def puking_file(filename, foldername): 
    '''
    LIST, DATAFRAMES, & DICTONARIES ONLY.
    Pukes/returns the pieces/batches of the object previously saved.  
    Auto-detects number of folders within foldername that CONTAIN the filename, as such...
    "K_"+filename+"_1", "K_"+filename+"_2", ... 
    (CAUTION: do not name your files too similarly when you batch save)
    reforms your object with data from the batched folders and returns  
    '''
    folder = os.listdir(foldername)
    files = sorted([s for s in folder if s[2:-2] == filename])
    
    d = klepto.archives.dir_archive('%s' % foldername, cached=True, serialized=True)
    for file in files:
        d.load(file[2:])
#     print(d.keys())
    
    if type(d[filename+"_1"]) == pd.DataFrame:
        df = []
        for key in sorted(d.keys()):
            df.append(d[str(key)])
        file = pd.concat(df)
    elif type(d[filename+"_1"]) == list:
        file = []
        for key in sorted(d.keys()):
            file.extend(d[str(key)])
    elif type(d[filename+"_1"]) == dict:
        file = {}
        for sub_dict in sorted(d.keys()):
            for key in d[sub_dict].keys():
                file[key] = d[sub_dict][key]
    else:
        file = d[filename]
        print("data is not a DF, list, or dict")
    
    print('Filename:', filename, 
          '\n# of Folders:', len(d.keys()),
          '\nType:', type(file), 
          '\nLen:', len(file))
    
    d.clear()
    return file
