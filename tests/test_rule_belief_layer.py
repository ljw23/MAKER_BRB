from maker_brb.brb.rule_belief_layer import Rule_belief_layer
import torch

def test_weight_layer():
    num_h = 3
    num_k = 20
    batch_size =8
    W_rule_act = torch.rand(batch_size,num_k)

    rule_belief_layer = Rule_belief_layer(num_k, num_h)
    output = rule_belief_layer(W_rule_act)

    print(output)
    print(output.shape)

if __name__ == '__main__':
    test_weight_layer()


