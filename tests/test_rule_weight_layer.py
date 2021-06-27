from maker_brb.brb.rule_weight_layer import Rule_weight_layer
import torch


def test_weight_layer():
    num_A = 10
    num_k = 20
    batchsize = 8
    alpha = torch.rand(batchsize, num_k, num_A)

    rule_weight_layer = Rule_weight_layer(num_k, num_A)
    output = rule_weight_layer(alpha)

    print(output)


if __name__ == '__main__':
    test_weight_layer()
