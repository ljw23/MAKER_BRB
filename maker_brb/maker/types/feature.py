# -*- encoding: utf-8 -*-
'''
@File    :   feature.py
@Time    :   2021/06/23 00:39:06
@Author  :   liujunwen 
@Version :   1.0
@Contact :   596951616@qq.com
'''

# here put the import lib
from typing import Dict, List,Set
from pydantic import BaseModel
from copy import  deepcopy
from pprint import pprint

class Feature_info(BaseModel):
    v: str
    q: str

    def __str__(self):
        return self.v+self.q

class  Feature(BaseModel):
    feature_id: int
    feature_info: List[Feature_info]

def find_feature(feature_info:List[Feature_info], feature_list:List[Feature]) -> Feature:
    for feature in feature_list:
        if feature_info == feature.Feature_info:
            return feature


class Feature_builder:
    def __init__(self):
        self.get_h_set()
        self.get_features()
        self.build_feature_list()

    def get_h_set(self):
        self.h_set = {1,0}
    
    def get_features(self):
        self.features =  {'V1':["A1","B1","C1","D1"],'V2':["A2","B2","C2","D2"],'V3':["A3","B3"]}

    def build_feature_list(self):
        '''
        构建特征以及所有交特征
        '''
        self.feature_list:List[Feature]= []

        feature_id = 0
        for  v, q_set  in self.features.items():
            _feature_info_list = []
            for q in q_set:
                feature_info = Feature_info(v=v,q=q)
                _feature_info_list.append(feature_info) #每个v下所有的q特征
            
            new_feature_list = []
            for _feature_info in _feature_info_list:
                new_feature_list.append(Feature(feature_id=feature_id, feature_info=[_feature_info])) #单独特征
                feature_id += 1
                for _feature in self.feature_list: #构建交叉特征
                    new_feature_info = [_feature_info]
                    new_feature_info.extend(_feature.feature_info)
                    new_feature = Feature(feature_id=feature_id, feature_info=sorted(new_feature_info,key=lambda x:str(x)))
                    new_feature_list.append(new_feature)
                    feature_id += 1
            self.feature_list.extend(new_feature_list)


if __name__ == '__main__':
    feature_builder =  Feature_builder()
    pprint(feature_builder.feature_list)
                