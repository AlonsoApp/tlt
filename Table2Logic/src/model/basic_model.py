import torch.nn as nn
from intermediate_representation.lf_parser import ASTTree


class BasicModel(nn.Module):

    def __init__(self):
        super(BasicModel, self).__init__()
        pass

    def padding_sketch(self, sketch):
        """
        Padding the sketch with leaf actions (C, I and V) where necessary.
        While we still don't know the id_c of the leaf actions, we know based on the grammar exactly, where to insert one.
        @param sketch:
        @return:
        """
        tree = ASTTree.from_action_list(sketch, padding=True)
        return tree.to_action_list()

