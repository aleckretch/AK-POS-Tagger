import random

class MEMM:

	DEFAULT_MODEL_PATH = "memm-model.txt" # the default path of the best model to be used
	DEFAULT_MIN_TOKEN_OCCURRENCES = 2 # the default minimum amount of occurrences for a token to appear to be considered by the model
	DEFAULT_MIN_FEATURE_OCCURRENCES = 30 # the default minimum amount of occurrences for a feature to appear to be considered by the model
	DEFAULT_TO_LOWERCASE = True # the default of whether or not to convert all tokens to their lowercase form for the model
	DEFAULT_MAX_EPOCHS = 10 # the default number of epochs for training the perceptron

	def __init__(self, model_path=DEFAULT_MODEL_PATH):
		self.reset_vars()
		if model_path is not None:
			with open(model_path) as model:
				self.load_model(model)

	def reset_vars(self):
		self.feature_dictionary = {}
		self.token_dictionary = {}
		self.token_and_tag_vectors = {} # the vectors for each token as a given tag in association with a given feature set
		self.unknown_token_and_tag_vectors = {}
		self.feature_count = 0

	def load_model(self, model):
		IS_KNOWN_TOKENS = 1
		IS_FEATURES = 2
		IS_TOKEN = 3
		IS_UNKNOWN = 4
		current_state = 0
		current_token = ""
		current_tag = ""
		for line in model:
			if len(line) > 2:
				if "TOKENS:\tTOKENS" in line:
					current_state = IS_KNOWN_TOKENS
				elif "FEATURES:\tFEATURES" in line:
					current_state = IS_FEATURES
				elif "TOKEN:\t" in line:
					current_state = IS_TOKEN
					current_token = line.split("TOKEN:\t")[1].strip()
					self.token_and_tag_vectors[current_token] = {}
				elif "TAG:\t" in line:
					current_tag = line.split("TAG:\t")[1].strip()
					if current_state == IS_TOKEN:
						self.token_and_tag_vectors[current_token][current_tag] = {}
					elif current_state == IS_UNKNOWN:
						self.unknown_token_and_tag_vectors[current_tag] = {}
				elif "BEST:\t" in line:
					vector_text = line.split("BEST:\t")[1].strip().split(" ")
					current_best_vector = [] # reset current best vector
					for number in vector_text:
						current_best_vector.append(int(number))
					if current_state == IS_TOKEN:
						self.token_and_tag_vectors[current_token][current_tag]["best"] = current_best_vector
					elif current_state == IS_UNKNOWN:
						self.unknown_token_and_tag_vectors[current_tag]["best"] = current_best_vector
				elif "BIAS:\t" in line:
					current_bias = int(line.split("BIAS:\t")[1].strip())
					if current_state == IS_TOKEN:
						self.token_and_tag_vectors[current_token][current_tag]["bias"] = current_bias
					elif current_state == IS_UNKNOWN:
						self.unknown_token_and_tag_vectors[current_tag]["bias"] = current_bias
				elif "UNKNOWN:\tUNKNOWN" in line:
					current_state = IS_UNKNOWN
				else:
					if current_state == IS_KNOWN_TOKENS:
						known_tokens = line.split(" ")
						i = 0
						for known_token in known_tokens:
							self.token_dictionary[known_token.strip()] = i
							i += 1
					elif current_state == IS_FEATURES:
						all_features = line.split(" ")
						i = 0
						for feature in all_features:
							self.feature_dictionary[feature.strip()] = i
							i += 1
						self.feature_count = i

	def get_pos_tags(self, sentence, to_lowercase=DEFAULT_TO_LOWERCASE):
		tag_predictions = [] # return array
		i = 0
		while i < len(sentence):
			token_minus_2 = None
			token_minus_1 = None
			token_plus_1 = None
			token_plus_2 = None
			if i >= 1:
				token_minus_1 = sentence[i-1]
				if to_lowercase:
					token_minus_1 = token_minus_1.lower()
				if i >= 2:
					token_minus_2 = sentence[i-2]
					if to_lowercase:
						token_minus_2 = token_minus_2.lower()
			if i+1 < len(sentence):
				token_plus_1 = sentence[i+1]
				if to_lowercase:
					token_plus_1 = token_plus_1.lower()
				if i+2 < len(sentence):
					token_plus_2 = sentence[i+2]
					if to_lowercase:
						token_plus_2 = token_plus_2.lower()
			token = sentence[i]
			if to_lowercase:
				token = token.lower()
			vector_minus_2 = self.empty_vector(self.feature_count)
			if token_minus_2 is not None and token_minus_2 in self.feature_dictionary:
				vector_minus_2[self.feature_dictionary[token_minus_2]] = 1
			vector_minus_1 = self.empty_vector(self.feature_count)
			if token_minus_1 is not None and token_minus_1 in self.feature_dictionary:
				vector_minus_1[self.feature_dictionary[token_minus_1]] = 1
			vector_token = self.empty_vector(self.feature_count)
			if token in self.feature_dictionary:
				vector_token[self.feature_dictionary[token]] = 1
			vector_plus_1 = self.empty_vector(self.feature_count)
			if token_plus_1 is not None and token_plus_1 in self.feature_dictionary:
				vector_plus_1[self.feature_dictionary[token_plus_1]] = 1
			vector_plus_2 = self.empty_vector(self.feature_count)
			if token_plus_2 is not None and token_plus_2 in self.feature_dictionary:
				vector_plus_2[self.feature_dictionary[token_plus_2]] = 1
			current_vector = vector_minus_2 + vector_minus_1 + vector_token + vector_plus_1 + vector_plus_2 # total vector is concatenation of vectors for window
			current_token_and_tag_vectors = {}
			if token not in self.token_and_tag_vectors:
				current_token_and_tag_vectors = self.unknown_token_and_tag_vectors
			else:
				current_token_and_tag_vectors = self.token_and_tag_vectors[token]
			highest_tag_similarity_score = 0
			tag_prediction = ""
			for tag in current_token_and_tag_vectors:
				best_vector = current_token_and_tag_vectors[tag]["best"]
				bias = current_token_and_tag_vectors[tag]["bias"]
				current_tag_similarity_score = self.get_similarity_score(best_vector, current_vector)-bias
				if len(tag_prediction) == 0 or current_tag_similarity_score > highest_tag_similarity_score:
					highest_tag_similarity_score = current_tag_similarity_score
					tag_prediction = tag
			tag_predictions.append(tag_prediction)
			i += 1
		return tag_predictions

	# development function
	def set_model(self, data, minimum_for_token=DEFAULT_MIN_TOKEN_OCCURRENCES, minimum_for_feature=DEFAULT_MIN_FEATURE_OCCURRENCES, to_lowercase=DEFAULT_TO_LOWERCASE, max_epochs=DEFAULT_MAX_EPOCHS):
		self.reset_vars()
		self.build_token_dictionary(data, minimum_for_token, to_lowercase)
		data.seek(0)
		self.build_feature_dictionary(data, minimum_for_feature, to_lowercase)
		data.seek(0)
		lines = [] # convert the data lines into an array of lines
		for line in data:
			lines.append(line)
		i = 0
		while i < len(lines):
			token_minus_2 = None
			token_minus_1 = None
			token_plus_1 = None
			token_plus_2 = None
			if i >= 1:
				line_minus_1 = lines[i-1]
				token_minus_1 = line_minus_1.split("\t")[0].strip()
				if to_lowercase:
					token_minus_1 = token_minus_1.lower()
				if i >= 2:
					line_minus_2 = lines[i-2]
					token_minus_2 = line_minus_2.split("\t")[0].strip()
					if to_lowercase:
						token_minus_2 = token_minus_2.lower()
			if i+1 < len(lines):
				line_plus_1 = lines[i+1]
				token_plus_1 = line_plus_1.split("\t")[0].strip()
				if to_lowercase:
					token_plus_1 = token_plus_1.lower()
				if i+2 < len(lines):
					line_plus_2 = lines[i+2]
					token_plus_2 = line_plus_2.split("\t")[0].strip()
					if to_lowercase:
						token_plus_2 = token_plus_2.lower()
			line = lines[i]
			if "\t" in line and len(line) > 2:
				token_and_tag = line.split("\t")
				token = token_and_tag[0].strip()
				if to_lowercase:
					token = token.lower()
				tag = token_and_tag[1].strip()
				vector_minus_2 = self.empty_vector(self.feature_count)
				if token_minus_2 is not None and token_minus_2 in self.feature_dictionary:
					vector_minus_2[self.feature_dictionary[token_minus_2]] = 1
				vector_minus_1 = self.empty_vector(self.feature_count)
				if token_minus_1 is not None and token_minus_1 in self.feature_dictionary:
					vector_minus_1[self.feature_dictionary[token_minus_1]] = 1
				vector_token = self.empty_vector(self.feature_count)
				if token in self.feature_dictionary:
					vector_token[self.feature_dictionary[token]] = 1
				vector_plus_1 = self.empty_vector(self.feature_count)
				if token_plus_1 is not None and token_plus_1 in self.feature_dictionary:
					vector_plus_1[self.feature_dictionary[token_plus_1]] = 1
				vector_plus_2 = self.empty_vector(self.feature_count)
				if token_plus_2 is not None and token_plus_2 in self.feature_dictionary:
					vector_plus_2[self.feature_dictionary[token_plus_2]] = 1
				vector = vector_minus_2 + vector_minus_1 + vector_token + vector_plus_1 + vector_plus_2 # total vector is concatenation of vectors for window
				if token in self.token_dictionary: # token was seen enough in data to include as a known token
					if token in self.token_and_tag_vectors:
						if tag in self.token_and_tag_vectors[token]:
							self.token_and_tag_vectors[token][tag]["vectors"].append(vector)
						else:
							self.token_and_tag_vectors[token][tag] = {"vectors": [vector], "best": self.empty_vector(self.feature_count*5), "bias": 0}
					else:
						self.token_and_tag_vectors[token] = {tag: {"vectors": [vector], "best": self.empty_vector(self.feature_count*5), "bias": 0}}
				else: # treat as unknown
					if tag in self.unknown_token_and_tag_vectors:
						self.unknown_token_and_tag_vectors[tag]["vectors"].append(vector)
					else:
						self.unknown_token_and_tag_vectors[tag] = {"vectors": [vector], "best": self.empty_vector(self.feature_count*5), "bias": 0}
			i += 1
		self.set_best_vectors(max_epochs)

	def build_token_dictionary(self, data, minimum_for_token, to_lowercase):
		token_counts = {}
		for line in data:
			if "\t" in line and len(line) > 2:
				token_and_tag = line.split("\t")
				token = token_and_tag[0].strip()
				if to_lowercase:
					token = token.lower()
				if token in token_counts:
					token_counts[token] += 1
				else:
					token_counts[token] = 1
		i = 0
		for token in token_counts:
			if token_counts[token] > minimum_for_token:
				self.token_dictionary[token] = i
				i += 1

	def build_feature_dictionary(self, data, minimum_for_feature, to_lowercase):
		feature_counts = {}
		for line in data:
			if "\t" in line and len(line) > 2:
				token_and_tag = line.split("\t")
				token = token_and_tag[0].strip()
				if to_lowercase:
					token = token.lower()
				if token in feature_counts:
					feature_counts[token] += 1
				else:
					feature_counts[token] = 1
		i = 0
		for feature in feature_counts:
			if feature_counts[feature] > minimum_for_feature:
				self.feature_dictionary[feature] = i
				i += 1
		self.feature_count = i

	# development function
	def save_model(self, save_path):
		known_tokens_string = "TOKENS:\tTOKENS\n"
		for known_token in self.token_dictionary.keys():
			known_tokens_string += "%s " % known_token
		features_string = "FEATURES:\tFEATURES\n"
		for feature in self.feature_dictionary.keys():
			features_string += "%s " % feature
		token_string = ""
		for token in self.token_and_tag_vectors:
			token_string += "TOKEN:\t%s\n" % token
			for tag in self.token_and_tag_vectors[token]:
				vector_string = ""
				for number in self.token_and_tag_vectors[token][tag]["best"]:
					vector_string += "%d " % number
				token_string += "TAG:\t%s\nBEST:\t%s\nBIAS:\t%d\n" % (tag, vector_string, self.token_and_tag_vectors[token][tag]["bias"])
		unknown_token_string = "UNKNOWN:\tUNKNOWN\n"
		for tag in self.unknown_token_and_tag_vectors:
			vector_string = ""
			for number in self.unknown_token_and_tag_vectors[tag]["best"]:
				vector_string += "%d " % number
			unknown_token_string += "TAG:\t%s\nBEST:\t%s\nBIAS:\t%d\n" % (tag, vector_string.strip(), self.unknown_token_and_tag_vectors[tag]["bias"])
		model_string = "%s\n\n\n%s\n\n\n%s\n\n\n%s" % (known_tokens_string.strip(), features_string.strip(), token_string.strip(), unknown_token_string.strip())
		# save the model
		with open(save_path, "w") as model_file:
			model_file.write(model_string)

	def set_best_vectors(self, max_epochs):
		# rearrange the token and tag vector data for the perceptron
		for token in self.token_and_tag_vectors:
			tag_vectors = []
			current_best_tag_vectors = {}
			for tag in self.token_and_tag_vectors[token]:
				for vector in self.token_and_tag_vectors[token][tag]["vectors"]:
					tag_vectors.append({"tag": tag, "vector": vector})
				current_best_tag_vectors[tag] = {"best": self.token_and_tag_vectors[token][tag]["best"], "bias": self.token_and_tag_vectors[token][tag]["bias"]}
			updated_best_tag_vectors = self.perceptron_best_vectors(tag_vectors, current_best_tag_vectors, max_epochs)
			# save the best vectors for the appropriate token
			for tag in self.token_and_tag_vectors[token]:
				self.token_and_tag_vectors[token][tag]["best"] = updated_best_tag_vectors[tag]["best"]
				self.token_and_tag_vectors[token][tag]["bias"] = updated_best_tag_vectors[tag]["bias"]
		# do the same for the unknown tokens
		unknown_tag_vectors = []
		current_best_unknown_tag_vectors = {}
		for tag in self.unknown_token_and_tag_vectors:
			for vector in self.unknown_token_and_tag_vectors[tag]["vectors"]:
				self.unknown_tag_vectors.append({"tag": tag, "vector": vector})
			self.current_best_unknown_tag_vectors[tag] = {"best": self.unknown_token_and_tag_vectors[tag]["best"], "bias": self.unknown_token_and_tag_vectors[tag]["bias"]}
		updated_best_unknown_tag_vectors = self.perceptron_best_vectors(unknown_tag_vectors, current_best_unknown_tag_vectors, max_epochs)
		for tag in self.unknown_token_and_tag_vectors:
			self.unknown_token_and_tag_vectors[tag]["best"] = updated_best_unknown_tag_vectors[tag]["best"]
			self.unknown_token_and_tag_vectors[tag]["bias"] = updated_best_unknown_tag_vectors[tag]["bias"]

	def empty_vector(self, size): # build a vector of all 0s for perceptron
		vector = []
		i = 0
		while i < size:
			vector.append(0)
			i += 1
		return vector

	def perceptron_best_vectors(self, tag_vectors, best_tag_vectors, max_epochs):
		has_converged = False
		i = 0
		while has_converged is False and i < max_epochs:
			has_converged = True # it has converged until proven otherwise
			random.shuffle(tag_vectors) # shuffle the data each epoch
			for tag_vector in tag_vectors: # run the perceptron through each of the shuffled data vectors for a given token
				test_tag = tag_vector["tag"]
				test_vector = tag_vector["vector"]
				for tag in best_tag_vectors: # compare the data vector of the given token to the best vector of each possible tag for that token
					best_vector = best_tag_vectors[tag]["best"]
					bias = best_tag_vectors[tag]["bias"]
					should_predict_tag = (tag == test_tag)
					does_predict_tag = self.perceptron_decision(best_vector, test_vector, bias)
					if should_predict_tag and does_predict_tag: # made correct prediction
						continue
					elif should_predict_tag and not does_predict_tag: # predicted it was tag, but should not have done so
						has_converged = False
						best_tag_vectors[tag]["bias"] = bias-1
						best_tag_vectors[tag]["best"] = self.add_vectors(best_vector, test_vector)
					elif not should_predict_tag and does_predict_tag: # predicted it was not the tag, but it was
						has_converged = False
						best_tag_vectors[tag]["bias"] = bias+1
						best_tag_vectors[tag]["best"] = self.subtract_vectors(best_vector, test_vector)
			i += 1
		return best_tag_vectors

	def perceptron_decision(self, w, x, bias): # returns true if the cosine similarity score is greater than the bias
		if self.get_similarity_score(w, x) > bias:
			return True
		else:
			return False

	def get_similarity_score(self, w, x): # cosine similarity score
		score = float(0.0)
		i = 0
		while i < len(w):
			add_amount = float(w[i]) * float(x[i])
			score += add_amount
			i += 1
		return score

	def add_vectors(self, w, x): # add two vectors
		new_vector = list(w)
		i = 0
		while i < len(new_vector):
			new_vector[i] += x[i]
			i += 1
		return new_vector

	def subtract_vectors(self, w, x): # subtract two vectors
		new_vector = list(w)
		i = 0
		while i < len(new_vector):
			new_vector[i] -= x[i]
			i += 1
		return new_vector