from pydantic import BaseModel
from typing import Dict, List,Set
import torch

ATTRIBUTE_DICT = {'V1':["A1","B1","C1","D1"],'V2':["A2","B2","C2","D2"],'V3':["A3","B3"]}

class Attribute_info(BaseModel):
    v: str = ''
    q: str = ''
    h: str = ''

def build_combined_atrributes():
    feature_info_list =[]
    for  v, q_set  in ATTRIBUTE_DICT.items():
        _feature_info_list = [] #每个v
        for q in q_set:
            feature_info = Attribute_info(v=v,q=q)
            _feature_info_list.append(feature_info) #每个v下所有的q特征
        
        new_feature_info_list = []
        for _feature_info in _feature_info_list:
            new_feature_info = [_feature_info]
            new_feature_info_list.append(new_feature_info) #单独特征
            for feature_info in feature_info_list: #构建交叉特征     
                new_feature_info = [_feature_info]
                new_feature_info.extend(feature_info)      
                new_feature_info_list.append(new_feature_info) 
                    
        feature_info_list.extend(new_feature_info_list)
    return feature_info_list