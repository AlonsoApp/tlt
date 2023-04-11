from execution.execute_generator import Node

class EvalNode(Node):
	def __init__(self, Node):
		super().__init__(Node.full_table, Node.dict_in)

	def _get_var_name(self, vars, func):
		"""
		Gets a variable name that hasn't been used yet
		The variable is the first character of the function name for example "a"
		if "a" is already in use it returns "a2", "a3" and so on
		"""
		c = func[0]
		if c in vars:
			vars[c]+=1
			return "{}{}".format(c, vars[c])
		else:
			vars[c]=1
			return c

	def to_amr(self, vars=None):
		"""
		Converts this Logic Form into an AMR like string
		All branches are :ARG parameters
		"""
		if vars is None:
			vars = {}
		args = []
		for i, child in enumerate(self.child_list):
			c_node_type, c_node = child
			if c_node_type == "text_node":
				var = self._get_var_name(vars, c_node)
				args.append(":ARG{i} ({var} / {fn})".format(i=i, var=var, fn=c_node))
			elif c_node_type == "func_node":
				tmp = EvalNode(c_node).to_amr(vars)
				args.append(":ARG{i} {tmp}".format(i=i, tmp=tmp))
		args_str = " ".join(args)
		if len(args)>0:
			args_str = " "+args_str
		var = self._get_var_name(vars, self.func)
		amr = "({var} / {fn}{args})".format(var=var, fn=self.func, args=args_str)
		return amr