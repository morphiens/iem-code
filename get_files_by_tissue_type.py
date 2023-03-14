import glob
import json
dataPath="../../Data/Morphle/data/slides/uploaded/*/meta/scan_params.json"
files = glob.glob(dataPath)

files_by_speciment={}
for _file in files:
    print(_file)    
    with open(_file) as f:
        _dict=json.load(f)
        specimentType=_dict["specimenType"]
        if specimentType not in files_by_speciment:
            files_by_speciment[specimentType] = []
        files_by_speciment[_dict["specimenType"]].append(_file)
with open("slides_by_tissue_type.json","w") as f:
    json.dump(files_by_speciment,f,indent=4)

for values in files_by_speciment:
    print(len(values))