import re
import ast
import warnings
from csv import reader

class ComplexityAnalyzer(ast.NodeVisitor):
	#NodeVisitor leszármazott osztály, összegyűjti az AST-ből a szükséges paramétereket
	def __init__(self):
		self.current_depth = 0
		self.max_depth = 0
		self.nesting_nodes = (ast.For, ast.While)

		self.variables = set()
		self.max_dimension = 0

	def visit_Name(self, node):
		if isinstance(node.ctx, ast.Store):
			self.variables.add(node.id)
		self.generic_visit(node)

	def visit_Subscript(self, node):
		dim_depth = 1
		current_node = node.value

		while isinstance(current_node, ast.Subscript):
			dim_depth += 1
			current_node = current_node.value

		if dim_depth > self.max_dimension:
			self.max_dimension = dim_depth
		self.generic_visit(node)

	def generic_visit(self, node):
		is_nesting_node = isinstance(node, self.nesting_nodes)
		if is_nesting_node:
			self.current_depth += 1
			if self.current_depth > self.max_depth:
				self.max_depth = self.current_depth
		super().generic_visit(node)
		if is_nesting_node:
			self.current_depth -= 1

def obtain_deep_params(code):
	#Visszatér a kódrészlet AST szerinti paraméterezésével (szótár objektum formájában)
	try:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", SyntaxWarning)
			tree = ast.parse(code)
		visitor = ComplexityAnalyzer()
		visitor.visit(tree)
		return {
			"unique_variables": len(visitor.variables),
			"max_dimension": visitor.max_dimension,
			"max_nesting_depth": visitor.max_depth
		}
	except SyntaxError:
		pass

def featurize(py_codes):
	#Elkészíti a kódrészlet-lista teljes vektorizálását
	base = []
	for code in py_codes:
		loop_regexp = r"^\s*(for\s+.*?\s+in\s+.*?|while\s+.*?)\s*:"
		matches = re.finditer(loop_regexp, code, re.M)
		loop_count = 0
		for match in matches:
			loop_count+=1
		features = obtain_deep_params(code)
		num_lines = len([line for line in code.split('\n')])

		if features != None:
			features.update({"loop_count": loop_count, "num_lines": num_lines})
		else:
			features = {
				"unique_variables": 0,
				"max_dimension": 0,
				"max_nesting_depth": 0,
				"loop_count": loop_count,
				"num_lines": num_lines
			}
		base.append([nums for nums in features.values()])
	return base
