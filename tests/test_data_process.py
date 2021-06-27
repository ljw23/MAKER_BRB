from maker_brb.brb.data_preprocess import Attribute_builder
from pprint import pprint

ATTRIBUTE_DICT = {
    "A1": [1.0, 2.0, 3.0, 4.0],
    "A2": [10.0, 20.0],
    "A3": [1.5, 2.5, 3.5, 4.8]
}
INPUT_DICT = {"A1": 2.5, "A2": 12.0, "A3": 4.8}


def test_Attribute_builder(attribute_dict=ATTRIBUTE_DICT):
    attribute_builder = Attribute_builder(ATTRIBUTE_DICT)

    input_x = attribute_builder.transform(INPUT_DICT)

    print('inpu_x:')
    pprint(input_x)

    print('k_rule_map:')
    pprint(attribute_builder.k_rule_map)
    pprint(attribute_builder.k_rule_map.shape)


if __name__ == '__main__':
    test_Attribute_builder()
