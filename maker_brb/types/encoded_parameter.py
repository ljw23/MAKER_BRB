from pydantic import BaseModel
from typing import Dict, List,Set
from pprint import pprint


ATTRIBUTE_DICT = {'V1':["A1","B1","C1","D1"],'V2':["A2","B2","C2","D2"],'V3':["A3","B3"]}

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

class Attribute_info(BaseModel):
    v: str = ''
    q: str = ''
    h: str = ''

class Encoded_Parameter(BaseModel):
    parameter_id :int  = -1 #参数张量的id
    attribute_combination: List[Attribute_info] = None
    value: float = 1.0

class Encoded_Parameters(BaseModel):

    parameter_list: List[Encoded_Parameter]

    def __init(self):
        super(Encoded_Parameters, self).__init__()

    def get(self,**kwargs)->float:
        raise NotImplementedError

    def from_id_get_value(self,parameter_id :int)->float:
        for parameter in self.parameter_list:
            if parameter_i == parameter.parameter_id:
                return parameter.value

class W_k_rules(Encoded_Parameters):
    def __init__(self):
        super(W_k_rules,self).__init__()
        attribute_list = build_combined_atrributes()
        self.build_parameter_list(attribute_list)
    
    def build_parameter_list(self,attribute_list):
        self.parameter_list = []
        for i,attribute in enumerate(attribute_list):
            self.parameter_list.append(Encoded_Parameter(parameter_id=i,
                                        attribute_combination=attribute,
                                        value=1.0))

    def get(self, k:int):
        return self.from_id_get_tensor(parameter_id=k)

class alpha_v_q(Encoded_Parameters):
    def get(self,v:int, q:int):
        for parameter in self.parameter_list:
            attribute = parameter.attribute_combination[0]
            if v==attribute.v and q == attribute.q:
                return self.from_id_get_tensor(parameter.parameter_id)



if __name__ == '__main__':
    pprint(build_combined_atrributes())
    pprint(len(build_combined_atrributes()))
    w_k_rules = W_k_rules()
    print(w_k_rules.get(k=1))

   
