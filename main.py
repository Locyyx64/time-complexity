import numpy as np
import matplotlib.pyplot as plt
import itertools
from parser import ComplexityAnalyzer, featurize
from concurrent.futures import ProcessPoolExecutor
from math import exp, ceil
from csv import reader
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

class LinearSVM:
	#Lineáris szupport vektor gép, subgradient descent megközelítéssel
	def __init__(self, lambda_param=0.01, n_iters=1000):
		self.lambda_param = lambda_param
		self.iters = n_iters
		self.w = None
		self.b = None
		self.lr = None
	
	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.w = np.zeros(n_features)
		self.b = 0

        # Subgradient descent (vectorized)
		for _ in range(self.iters):
			i = 0
			for idx, x_i in enumerate(X):
				i+=1
				self.lr = 1/(self.lambda_param*i)
				condition = y[idx] * (np.dot(x_i, self.w) - self.b)
				if condition >= 1:
					## normal subgradient descent
					self.w -= self.lr * (self.lambda_param*self.w)
				else:
					##misclassified subgradient
					self.w -= self.lr * (self.lambda_param*self.w - np.dot(x_i, y[idx]))
					self.b -= self.lr * y[idx]
	def decision_function(self, X):
		return np.dot(X, self.w) - self.b

class MulticlassSVM:
	#Többosztályos SVM aggregátor, létrehozza és kezeli az összes osztály SVM-jét
	def __init__(self, n_classes=7, lambda_param=0.001, n_iters=1000):
		self.n_classes = n_classes
		self.classifiers = []
		self.lambda_param = lambda_param
		self.iters = n_iters
		self.classifiers = []
	
	def fit_one(self, iter, X, y):
		print(f"Training the SVM for class {iter}...")
		y_binary = np.where(y == iter, 1, -1)
		svm = LinearSVM(lambda_param=self.lambda_param, n_iters=self.iters)
		svm.fit(X, y_binary)
		print(f"Class {iter} training finished!")
		return svm

	def fit(self, X, y):
		with ProcessPoolExecutor(max_workers=7) as executor:
			self.classifiers = list(executor.map(self.fit_one, range(7), itertools.repeat(X), itertools.repeat(y)))
	
	def predict(self, X):
		predicted = []
		for sample in X:
			all_predictions = [cl.decision_function(sample) for cl in self.classifiers]
			#print(all_predictions)
			predicted.append(np.argmax(all_predictions))
		return predicted


## BATCH GENERATOR!!!
def train_test_split(X, y, test_size=0.2, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
        
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions

#def generate_confusion_matrix(y_true, y_pred, n_classes=7):
#	matrix = np.zeros((n_classes, n_classes), dtype=int)
#    
#	for true_label, pred_label in zip(y_true, y_pred):
#    	matrix[true_label, pred_label] += 1
#        
#	return matrix

if __name__ == "__main__":
	print("Loading codes...")
	with open("../data/python_raw.csv", 'r') as file:
		csv_file = reader(file)
		csv_file = [line[1:] for line in csv_file]
	csv_file.pop(0)
	py_codes = [entry[0] for entry in csv_file]
	complexities = [entry[1] for entry in csv_file]

	compl = ["constant", "logn", "linear", "nlogn", "quadratic", "cubic", "np"]
	
	for i in range(len(complexities)):
		complexities[i] = compl.index(complexities[i])
	#print(complexities)
	print("Featurizing codes...")
	base = featurize(py_codes)
	#print(len(base))
	#print(len(complexities))
	
	##-----------ONE VS. REST SVM --------------#

	X = np.array(base, dtype=float)
	y = np.array(complexities, dtype=int)

	use_embed = True
	#Változtasd 'graphcodebert'-re ha azt szeretnéd használni
	model = "unixcoder"
	file = f"{model}_features.npy"
	if use_embed == True:	
		if not Path(file).exists():
			print(f"\t\t{file} not found! You need to run transform.py first (modify last part to {model})!")
		else:
			print(f"Loading {model} features...")
			X = np.load(file)

	#Előfeldolgozás (standardizálás)
	print("Scaling data...")
	X_mean = np.mean(X, axis=0)
	X_std = np.std(X, axis=0)
	X_scaled = (X - X_mean) / (X_std + 1e-8)

	#print(X_scaled)
	
	#Train batch és test batch létrehozása (sztochasztikusan) a megadott arányban
	print("Splitting data...")
	
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
	print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.\n")

	#Lineáris SVM inicializálása, tanítása, érvényesítése (kikommentelt, mert a tanítás sok időt vesz igénybe, 
	#és miután nyilvánvaló lett hogy az adathalmaz nem lineárisan szétválasztható, felesleges mindig futtatni)
#	model = MulticlassSVM(n_classes=7, lambda_param=0.5, n_iters=1000)
#	model.fit(X_train, y_train)
# 
#	print("\nMaking predictions on test set...")#	
#	y_pred = model.predict(X_test)

#	accuracy = accuracy_score(y_test, y_pred)
#	cm1 = confusion_matrix(y_test, y_pred)

#	print(f"Linear accuracy: {accuracy * 100:.2f}%")


	#Nemlineáris SVM inicializálása, tanítása, GridSearch alkalmazása
	#Gamma: 0.5 alapból, 'scale' a transzformer feature-ök használatakor
	#param_grid = {
	#	"kernel": ["rbf"],
	#	"C": [1, 5, 10, 20, 40, 50, 60, 70, 90, 100],
	#	"class_weight": ['balanced'],
	#		"gamma": ['scale', 'auto', 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
	#}

	non_linear = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")

	#grid_search = GridSearchCV(
	#	estimator=non_linear,
	#	param_grid=param_grid,
	#	cv=5,
	#	scoring='accuracy',
	#	n_jobs=-1,
	#	verbose=2
	#)

	#grid_search.fit(X_train, y_train)
	#print(f"Best params: {grid_search.best_params_}")
	non_linear.fit(X_train, y_train)
	y_pred1_test = non_linear.predict(X_test)
	print("Non-linear accuracy: ",  non_linear.score(X_test, y_test))

	#Random Forest inicializálása, tanítása, szintén GridSearch
	rf = RandomForestClassifier(
		bootstrap=True,
		n_estimators=200,
		min_samples_leaf=2,
		max_depth=10
		)
	
	#param_grid = {
    #	'n_estimators': [100, 200],
    #	'max_depth': [None, 10, 20],
    #	'min_samples_split': [2, 5],
    #	'min_samples_leaf': [1, 2],
    #	'bootstrap': [True, False]
	#}

	#grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
	#grid_search.fit(X_unscaled_train, y_train)
	
	#print("Best Parameters:", grid_search.best_params_)
	#print("Best Estimator:", grid_search.best_estimator_)
	
	rf.fit(X_train, y_train)
	y_pred = rf.predict(X_test)
	cm3 = confusion_matrix(y_test, y_pred)
	print(f"RF accuracy: {accuracy_score(y_test, y_pred)}")

	#Többosztályos XGBoost alkalmazása
	xgb_model = XGBClassifier(
		num_class=7,
		eval_metric='mlogloss',
		max_depth=5,
		learning_rate=0.04,
		objective='multi:softmax',
		n_estimators=120,
	)
	xgb_model.fit(X_train, y_train)
	y_pred = xgb_model.predict(X_test)
	print("XGBoost accuracy: ", accuracy_score(y_test, y_pred))
	cm4 = confusion_matrix(y_test, y_pred)

	#Confusion mátrixok
	fig, axs = plt.subplots(1,3)

	#cm2_train = confusion_matrix(y_train, y_pred1_train)
	cm2_test = confusion_matrix(y_test, y_pred1_test)

	# Miután a lineáris SVM nem fut let minden alkalommal, nem jelenítjük meg a confusion mátrixát
	#disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=compl)
	#disp1.plot(ax=axs[0,0],cmap=plt.cm.Blues)
	#axs[0, 0].set_title("Linear SVM")

	disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2_test, display_labels=compl)
	disp2.plot(ax=axs[0],cmap=plt.cm.Blues, colorbar=False)
	axs[0].set_title("RBF kernel")

	disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=compl)
	disp3.plot(ax=axs[1],cmap=plt.cm.Blues, colorbar=False)
	axs[1].set_title("Random forest")

	disp4 = ConfusionMatrixDisplay(confusion_matrix=cm4, display_labels=compl)
	disp4.plot(ax=axs[2],cmap=plt.cm.Blues, colorbar=False)
	axs[2].set_title("XGBoost")

	plt.tight_layout()
	plt.show()
	


