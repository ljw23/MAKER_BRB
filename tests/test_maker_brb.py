from maker_brb import __version__
from maker_brb.types.alpha_v_q import *
from maker_brb.types.encoded_parameter import *
from maker_brb.types.w_k_rules import *
from maker_brb.types.attrbute import *


def test_version():
    assert __version__ == '0.1.0'


def test_types():
    pprint(build_combined_atrributes())
    pprint(len(build_combined_atrributes()))
    w_k_rules = W_k_rules()
    print(w_k_rules.get(k=1))


if __name__ == '__main__':
    test_types()
