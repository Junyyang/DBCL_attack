import math
import torch
import torch.nn as nn
from .Sketch import Sketch

# cpu = torch.device("cpu")
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = cpu

from src.conf import Args

args = Args()

device = args.device


class SketchLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, hash_idx=None, rand_sgn=None, training=True, q=2, sketchmat=None):
        if training:
            # input_features = weight.shape[-1]

            # sketching the input and weight matrices
            # hash_idx, rand_sgn = Sketch.rand_hashing(input_features, q)
            if args.sketchtype == 'count':
                input_sketch = Sketch.countsketch(input.to(device), hash_idx, rand_sgn).to(device)
                weight_sketch = Sketch.countsketch(weight.to(device), hash_idx, rand_sgn).to(device)
                output = input_sketch.mm(weight_sketch.t())

                ctx.save_for_backward(input_sketch, weight_sketch, bias, hash_idx, rand_sgn)
        
            else:
                input_sketch = Sketch.gaussiansketch(input.to(device), sketchmat).to(device)
                weight_sketch = Sketch.gaussiansketch(weight.to(device), sketchmat).to(device)

                output = input_sketch.mm(weight_sketch.t())

                ctx.save_for_backward(input_sketch, weight_sketch, bias, sketchmat)
        else:
            output = input.mm(weight.t())

        output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if args.sketchtype == 'count':
            input_sketch, weight_sketch, bias, hash_idx, rand_sgn = ctx.saved_tensors
        else:
            input_sketch, weight_sketch, bias, sketchmat = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = grad_training = None

        if ctx.needs_input_grad[0]:
            grad_input_tmp = grad_output.to(device).mm(weight_sketch.to(device))
            if args.sketchtype == 'count':
                grad_input = Sketch.transpose_countsketch(grad_input_tmp.to(device), hash_idx, rand_sgn).to(device)
            else:
                grad_input = Sketch.transpose_gaussiansketch(grad_input_tmp.to(device), sketchmat).to(device)

        if ctx.needs_input_grad[1]:
            grad_weight_tmp = grad_output.to(device).t().mm(input_sketch.to(device))
            if args.sketchtype == 'count':
                grad_weight = Sketch.transpose_countsketch(grad_weight_tmp.to(device), hash_idx, rand_sgn).to(device)
            else:
                grad_weight = Sketch.transpose_gaussiansketch(grad_weight_tmp.to(device), sketchmat).to(device)

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0).to(device)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class SketchLinear(nn.Module):
    def __init__(self, input_features, output_features, q=2):
        super(SketchLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.q = q

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)

        bound = 1 / math.sqrt(input_features)
        scaling = math.sqrt(3.0)
        self.weight.data.uniform_(-bound * scaling, bound * scaling)
        self.bias.data.uniform_(-bound, bound)

    def forward(self, input, hash_idx=None, rand_sgn=None, sketchmat=None):
        return SketchLinearFunction.apply(input, self.weight, self.bias, hash_idx, rand_sgn, self.training, self.q, sketchmat)

    def extra_repr(self):
        return 'input_features={}, output_features={}, weight={}, bias={}'.format(
            self.input_features, self.output_features, self.weight, self.bias
        )