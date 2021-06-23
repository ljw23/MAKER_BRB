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
    h: str
    feature_info: List[Feature_info]

    def __str__(self):
        return '_'.join([str(_feature_info) for _feature_info in self.feature_info])

class Feature_builder:
    def __init__(self):
        self.get_h_set()
        self.get_features()
        self.build_feature_list()

    def get_h_set(self):
        self.h_set = {'1','0'}
    
    def get_features(self):
        self.features =  {'V1':["A1","B1","C1","D1"],'V2':["A2","B2","C2","D2"],'V3':["A3","B3"]}

    def build_feature_list(self):
        '''
        构建特征以及所有交特征
        '''
        self.feature_list:List[Feature]= []  
        
        
        feature_info_list =[]
        for  v, q_set  in self.features.items():
            _feature_info_list = [] #每个v
            for q in q_set:
                feature_info = Feature_info(v=v,q=q)
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
    

        feature_id = 0
        for h in self.h_set:
            for feature_info in feature_info_list:
                self.feature_list.append(Feature(feature_id=feature_id, h='all',feature_info=sorted(feature_info,key=lambda x:str(x)))) #无假设纯特征
                self.feature_list.append(Feature(feature_id=feature_id, h=h,feature_info=sorted(feature_info,key=lambda x:str(x))))
                feature_id += 2
        
    def find_feature_id(self,h=None, feature_info:List[Feature_info]=None) -> int:
        for feature in self.feature_list:
            if feature_info == feature.feature_info and (not h or h==feature.h):
                return feature.feature_id
    
    def find_feature(self, feature_id:int) -> Feature:
        for feature in self.feature_list:
            if feature_id == feature.feature_id:
                return feature
    
    @property
    def len_feature_list(self):
        return len(self.feature_list)

    def find_v_features(self,v_list:List[str]=None,h:str=None,level:int=0)->List[Feature]:
        '''
        根据v特征列表以及层级查询特征
        '''
        if not v_list and not level:
            return
        if not level:
            level = len(v_list)
        v_features = []
        for feature in self.feature_list:
            if len(feature.feature_info) != level:
                continue
            if not v_list:
                v_features.append(feature)
                continue
            v_list_in_feature = [feature_info.v for feature_info in feature.feature_info]
            if v_list_in_feature == v_list and (not h or h==feature.h):
                v_features.append(feature)
        return v_features

    def find_single_feature(self):
        return find_v_features(level=1)

    def find_multi_feature(self):
        level_set = set(range(len(self.features)))
        level_set.remove(0)
        multi_features = []
        for level in level_set:
            multi_features.extend(self.find_v_features(level=level+1))
        return multi_features


if __name__ == '__main__':
    feature_builder =  Feature_builder()
    # pprint(feature_builder.feature_list)
    # pprint(feature_builder.len_feature_list)
    # pprint(feature_builder.find_v_features(v_list=['V1','V3']))
    # print(len(feature_builder.find_v_features(v_list=['V1','V3'])))
    # print('level:')
    # pprint(feature_builder.find_v_features(level=2))
    pprint(feature_builder.find_feature_id(h='1',feature_info=[Feature_info(v='V1', q='B1')]))
