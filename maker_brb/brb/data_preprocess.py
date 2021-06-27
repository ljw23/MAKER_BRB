from typing import List, Dict
import numpy as np
from copy import deepcopy

'''
attribute_dict:Dict 特征属性的字典 {A_i: A_v,i }
{A1:[0,1,2], A2:[1.1, 1,5, 3.0, 4.0]}
维护一个位置字典{A1:[121,122,123]}
input_x 与特征属性字典格式一致
转化为 alpha_origin  {A1:[0,1,0], A2:[0.5, 0.5, 0, 0]}
alpha然后转成[0,1,0,0.5,0.5,0,0,...]

k_dict : [k, num_A] 描述第k条规则，每个属性取第几个类别
return alpha_k: [:, k, num_A]
'''


class Attribute_builder:
    def __init__(self, attributes: Dict):
        self.attributes = attributes
        self.num_A = len(attributes)

        self.build_attribute_index_map()
        self.num_k = self.get_rules_num()
        self.build_k_rule_map()

    def build_attribute_index_map(self):
        '''
        #构建特征位置字典
        {A1:[121,122,123]}
        '''
        self.attribute_index_map = {}
        i = 0
        for attr, value_list in self.attributes.items():
            self.attribute_index_map[attr] = []
            for value in value_list:
                self.attribute_index_map[attr].append(i)
                i += 1
        self.num_attribute_values = i

    def transform_input_dict(self, input_x_dict:Dict)->List[float]:
        '''
        input_x_dict: {A1:0.7,A2:0.8}
        return input_x: [num_attribute_values], [0,0.2,0.8,0,1.0,0,0]
        '''
        input_x = self.num_attribute_values*[0.]

        for attr, input_Ax in input_x_dict.items():
            attribute_ref_values = self.attributes[attr]
            alpha_list = self.get_attribute_alpha(attribute_ref_values=attribute_ref_values,input_Ax=input_Ax)
            for i, alpha in enumerate(alpha_list):
                input_x_index = self.attribute_index_map[attr][i]
                input_x[input_x_index] = alpha
        
        return input_x
                
    def get_attribute_alpha(self, attribute_ref_values: List[float],
                            input_Ax: float) ->List[float]:
        '''
        实现论文公式(2)
        attribute_values: [1.0,1.5,2.0]
        input_Ax: 1.8
        attribute_dist: [0, 0.8, 0.2]
        '''
        num_a_v = len(attribute_ref_values)
        ret = num_a_v*[0.0]

        upper_index = -1
        lower_index = -1
        for i, v in enumerate(attribute_ref_values):
            if input_Ax <=v:
                upper_index = i
                break
            if input_Ax >=v:
                lower_index = i

        if upper_index < 0: #input_Ax没有上限
            ret[-1] = 1.0
            return ret

        if lower_index < 0: #input_Ax没有下限
            ret[0] = 1.0
            return ret

        assert lower_index+1 == upper_index

        ret[lower_index] = (attribute_ref_values[upper_index]-input_Ax)/(attribute_ref_values[upper_index]-attribute_ref_values[lower_index])
        ret[upper_index] = 1-ret[lower_index]
        return ret
    
    #对于规则转换计算

    def get_rules_num(self):
        '''
        计算规则总数量，每个属性的取值类别的乘积
        '''
        num_k = 1
        for attr, value_list in self.attributes.items():
            num_k *= len(value_list)
        return num_k

    def build_k_rule_map(self):
        '''
        建立规则的映射表，每个属性取一个属性值
        [num_k, num_A]
        '''
        def add_attr(rule_map, value_list):
            ret = []
            for i,value in enumerate(value_list):
                _rule_map = deepcopy(rule_map)
                for rule in _rule_map:
                    rule.append(i)
                ret.extend(_rule_map)
            return ret

        self.k_rule_map = [[]]
        self.attr_id_map = [] #属性对应的id，对应k_rule_map里的位置顺序

        for attr, value_list in self.attributes.items():
            self.attr_id_map.append(attr)
            self.k_rule_map = add_attr(self.k_rule_map, value_list)
        self.k_rule_map = np.array(self.k_rule_map)

    def transform_input_x(self, input_x):
        '''
        input_x是已经转换为1维的向量
        现在要将input_x转换为alpha [num_k, num_A]
        1. 先根据 k_rule_map的列id判断是哪一个Attribute
        2. 再根据k_rule_map的值判断Attribute_ref_value的id
        3.从attribute_index_map字典中获取此Attribute_ref_value在input_x中取第几个值
        '''
        alpha_matrix = np.zeros_like(self.k_rule_map, dtype=float)
        for k, k_rule in enumerate(self.k_rule_map):
            for attribute_id, attribute_ref_id_in_attr in enumerate(k_rule):
                attr = self.attr_id_map[attribute_id]
                attribute_ref_id_in_inputx  =self.attribute_index_map[attr][attribute_ref_id_in_attr]
                _alpha_value = input_x[attribute_ref_id_in_inputx]
                alpha_matrix[k,attribute_id] = _alpha_value
        return alpha_matrix

    def transform(self, input_dict):
        input_x = self.transform_input_dict(input_dict)
        return self.transform_input_x(input_x)


