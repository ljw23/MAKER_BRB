# import torch.nn as nn
# from torch.nn import Linear
# from torch.nn.parameter import Parameter, UninitializedParameter
# from torch import init
# from typing import  Dict,List,Set,Iterable
# from torch import Tensor
# from .types.feature import Feature_builder
# from .types.instance import Instance

# class MAKER(nn.Module):
#     def __init__(self, basic_p:Dict,feature_builder:Feature_builder):
#         '''
#         {'V1':["A1","B1","C1","D1"],'V2':["A2","B2","C2","D2"],'V3':["A3","B3"]}
#         '''
#         factory_kwargs = {'device':device, 'dtype':dtype}
#         super(MAKER, self).__init__()
#         self.basic_p = basic_p
#         self.feature_builder = feature_builder
#         single_features = feature_builder.find_v_features(level=1)
#         self.features_num = len(single_features)
#         w_p_mass = {}
#         w_r = {}
#         for p_mass_feature in single_features:
#             self.w_p_mass[p_mass_feature.feature_id] = Parameter[torch.Tensor(1)]
#             self.w_r[p_mass_feature.feature_id] = Parameter[torch.Tensor(1)]

#         multi_features = feature_builder.find_multi_feature()
#         for multi_feature in multi_features:
#             self.w_gama[multi_feature.feature_id] = Parameter[torch.Tensor(1)]

#     def forward(self,batch_instance:List[Instance]):

#         m_hvq = {}
#         for k, v in self.w_p_mass:
#             m_hvq[k] = v*self.basic_p[k]

#         r = 0
#         for h in self.feature_builder.h_set:
