import json
import os
import logging
input_images_pattern="../../Data/Output/*.png"
output_images_dir="../../Data/ByTissueType"

scan_params_template="../../Data/Morphle/data/slides/uploaded/{}/meta/scan_params.json"
import glob
morphle_id_to_tissue_type_map={}


def getMorphleId(img):
    mid=img[:img.index(".jpg")]
    mid=mid[:mid.rindex("_")]
    mid=mid[mid.rindex("/")+1:]
    return mid
    pass
def getTissueType(img):
    mid=getMorphleId(img)
    if mid  not in morphle_id_to_tissue_type_map:
        scan_params=scan_params_template.format(mid)
        with open(scan_params) as f:
            _dict=json.load(f)
        morphle_id_to_tissue_type_map[mid]=_dict["specimenType"]


    return morphle_id_to_tissue_type_map[mid]
    
for _file in glob.glob(input_images_pattern):
    try:
        tissue_type=getTissueType(_file)
        _dir=os.path.join(output_images_dir,tissue_type)
        os.makedirs(_dir,exist_ok=True)
        os.symlink(os.path.abspath(_file),os.path.join(_dir,os.path.basename(_file)))
    except:
        logging.exception("Failed to process %s",_file)
    
    # break
    