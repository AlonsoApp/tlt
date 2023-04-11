import copy

from intermediate_representation.lf_grammar import Stat
from intermediate_representation.lf_parser import ASTTree


class ActionInfo(object):
    """sufficient statistics for making a prediction of an action at a time step"""

    def __init__(self, action=None):
        self.t = 0
        self.score = 0
        self.parent_t = -1
        self.action = action
        self.frontier_prod = None
        self.frontier_field = None

        # for GenToken actions only
        self.copy_from_src = False
        self.src_token_position = -1


class Beams(object):
    def __init__(self, is_sketch=False):
        self.actions = []
        self.action_infos = []
        self.inputs = []
        self.score = 0.
        self.t = 0
        self.is_sketch = is_sketch
        self.sketch_step = 0
        self.sketch_attention_history = list()

    def get_availableClass(self):
        """
        return next possible action class.
        :return:
        """

        # TODO: it could be update by speed
        # return the available class using rule
        # FIXME: now should change for these 11: "Filter 1 ROOT",
        def check_type(lists):
            for s in lists:
                if type(s) == int:
                    return False
            return True

        stack = [Stat]
        for action in self.actions:
            infer_action = action.get_next_action(is_sketch=self.is_sketch)
            infer_action.reverse()
            if stack[-1] is type(action):
                stack.pop()
                # check if the are non-terminal
                if check_type(infer_action):
                    stack.extend(infer_action)
            else:
                raise RuntimeError("Not the right action")

        result = stack[-1] if len(stack) > 0 else None

        return result

    def apply_action(self, action):
        # TODO: not finish implement yet
        self.t += 1
        self.actions.append(action)

    def clone_and_apply_action(self, action):
        new_hyp = self.copy()
        new_hyp.apply_action(action)

        return new_hyp

    def clone_and_apply_action_info(self, action_info):
        action = action_info.action
        action.score = action_info.score
        new_hyp = self.clone_and_apply_action(action)
        new_hyp.action_infos.append(action_info)
        new_hyp.sketch_step = self.sketch_step
        new_hyp.sketch_attention_history = copy.copy(self.sketch_attention_history)

        return new_hyp

    def copy(self):
        new_hyp = Beams(is_sketch=self.is_sketch)
        # if self.tree:
        #     new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.score = self.score
        new_hyp.t = self.t
        new_hyp.sketch_step = self.sketch_step
        new_hyp.sketch_attention_history = copy.copy(self.sketch_attention_history)

        return new_hyp

    @property
    def completed(self):
        return True if self.get_availableClass() is None else False

    @property
    def is_valid(self):
        actions = self.actions
        return self.check_sel_valid(actions)

    def check_sel_valid(self, actions):
        # All hypotheses should be valid at this point. We could comment this to save computing time
        #tree = ASTTree.from_action_list(actions)
        #return tree.is_valid
        return True

