import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from csv import reader

def transform(codes, model_code="graphcodebert"):
	#A megadott open modell szerint vektorizálja a kódrészleteket, és lementi egy numpy fájlba
	model_name = f"microsoft/{model_code}-base"

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModel.from_pretrained(model_name)

	device = torch.device("cpu")
	model.to(device)
	model.eval()

	all_embeddings = []
	with torch.no_grad():
		for i, code in enumerate(codes):
			inputs = tokenizer(
				code,
				padding=True,
				truncation=True,
				max_length=512,
				return_tensors="pt"
			).to(device)
			outputs = model(**inputs)
			#print("Processing code snippet...")

			embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
			print(f"Processed: {i}/{len(codes)}")
			all_embeddings.append(embedding)
	embd_matrix = np.vstack(all_embeddings)
	np.save(f"{model_code}_features.npy", embd_matrix)
	print("Embeddings saved.")

	return embd_matrix

if __name__ == "__main__":
	with open("../data/python_raw.csv", 'r') as file:
		csv_file = reader(file)
		csv_file = [line[1:] for line in csv_file]
	csv_file.pop(0)
	py_codes = [entry[0] for entry in csv_file]
	complexities = [entry[1] for entry in csv_file]

	test_embeddings = transform(py_codes, model_code="unixcoder")
	#print(f"Shape of output: {test_embeddings.shape}")
	#print(test_embeddings_code)
