from .encoded_parameter import Encoded_Parameters

class Alpha_v_q(Encoded_Parameters):
    def get(self,v:int, q:int):
        for parameter in self.parameter_list:
            attribute = parameter.attribute_combination[0]
            if v==attribute.v and q == attribute.q:
                return self.from_id_get_tensor(parameter.parameter_id)
