class MM:

	DEFAULT_MODEL_PATH = "mm-model.txt" # the default path of the best model to be used
	DEFAULT_MIN_TOKEN_OCCURRENCES = 2 # the default minimum amount of occurrences for a token to appear to be considered by the model
	DEFAULT_TO_LOWERCASE = False # the default of whether or not to convert all tokens to their lowercase form for the model
	SMOOTHING_SUFFIXES = ["acy", "al", "ance", "ence", "dom", "er", "or", "ism", "ist", "ity", "ty", "ment", "ness", "ship", "ation", "ition", "sion", "tion", "ion", "ate", "en", "ify", "fy", "ize", "ise", "able", "ible", "ial", "al", "esque", "ful", "ic", "ical", "ious", "eous", "ous", "ish", "ative", "itive", "ive", "less", "ing", "est", "ly", "y", "ed", "es", "s"] # suffixes to check for in the training set to be used for unknown words with the same suffix in testing
	
	def __init__(self, model_path=DEFAULT_MODEL_PATH):
		self.reset_vars()
		if model_path is not None:
			with open(model_path) as model:
				self.load_model(model)

	def reset_vars(self):
		self.token_as_tag_likelihood = {} # the counts of occurrences of all tags of which a token is seen
		self.suffixed_token_as_tag_likelihood = {} # the counts of occurrences of all tags of which a token ending in a certain suffix is seen
		self.hyphenated_token_as_tag_likelihood = {"total":0} # the counts of occurrences of all tags of which a token with a hyphen is seen
		self.capitalized_token_as_tag_likelihood = {"total":0} # the counts of occurrences of all tags of which a token starting with a capital character is seen
		self.unknown_token_as_tag_likelihood = {} # the counts of occurrences of all tags of which an unknown token is seen
		self.tag_to_tag_likelihood = {} # the counts of tags following a given tag

	def load_model(self, model):
		IS_TOKEN = 1
		IS_SUFFIX = 2
		IS_HYPHEN = 3
		IS_CAPITALIZED = 4
		IS_UNKNOWN = 5
		IS_TAG = 6
		current_state = 0
		current_key = ""
		for line in model:
			if len(line) > 0:
				if "TOKEN:\t" in line:
					current_state = IS_TOKEN
					current_key = line.split("TOKEN:\t")[1].strip()
				elif "SUFFIX:\t" in line:
					current_state = IS_SUFFIX
					current_key = line.split("SUFFIX:\t")[1].strip()
				elif "HYPHEN:\tHYPHEN" in line:
					current_state = IS_HYPHEN
				elif "CAPITALIZED:\tCAPITALIZED" in line:
					current_state = IS_CAPITALIZED
				elif "UNKNOWN:\tUNKNOWN" in line:
					current_state = IS_UNKNOWN
				elif "TAG:\t" in line:
					current_state = IS_TAG
					current_key = line.split("TAG:\t")[1].strip()
				elif "total " in line: # handle data
					data = line.split("\t")
					tag_and_count_dict = {}
					for datum in data:
						tag_and_count = datum.split(" ")
						if len(tag_and_count) > 1:
							tag = tag_and_count[0].strip()
							count = int(tag_and_count[1].strip())
							tag_and_count_dict[tag] = count
					if current_state == IS_TOKEN:
						self.token_as_tag_likelihood[current_key] = tag_and_count_dict
					elif current_state == IS_SUFFIX:
						self.suffixed_token_as_tag_likelihood[current_key] = tag_and_count_dict
					elif current_state == IS_HYPHEN:
						self.hyphenated_token_as_tag_likelihood = tag_and_count_dict
					elif current_state == IS_CAPITALIZED:
						self.capitalized_token_as_tag_likelihood = tag_and_count_dict
					elif current_state == IS_UNKNOWN:
						self.unknown_token_as_tag_likelihood = tag_and_count_dict
					elif current_state == IS_TAG:
						self.tag_to_tag_likelihood[current_key] = tag_and_count_dict

	def get_pos_tags(self, sentence, to_lowercase=DEFAULT_TO_LOWERCASE):
		tag_predictions = [] # return array
		prev_tag_prediction = "" # each prediction tag is based in part on previous tag
		for token in sentence:
			if to_lowercase:
				token = token.lower()
			is_unknown = False
			current_token_as_tag_likelihood = {}
			if token not in self.token_as_tag_likelihood:
				has_suffix = False
				for suffix in self.suffixed_token_as_tag_likelihood:
					if suffix != "total" and token.endswith(suffix):
						has_suffix = True
						current_token_as_tag_likelihood = self.suffixed_token_as_tag_likelihood[suffix]
						break # as soon as the suffix is found stop searching to save time and avoid using a smaller subsuffix of the found suffix
				if has_suffix is False:
					current_token_as_tag_likelihood = self.unknown_token_as_tag_likelihood
				is_unknown = True
			else:
				current_token_as_tag_likelihood = self.token_as_tag_likelihood[token]
			token_total = int(current_token_as_tag_likelihood["total"])
			highest_probability = float(0)
			tag_prediction = ""
			for tag in current_token_as_tag_likelihood:
				if tag != "total":
					current_probability = float(int(current_token_as_tag_likelihood[tag])/token_total)
					if len(prev_tag_prediction) > 0:
						prev_tag_total = int(self.tag_to_tag_likelihood[prev_tag_prediction]["total"])
						if tag in self.tag_to_tag_likelihood[prev_tag_prediction]:
							current_probability *= float(int(self.tag_to_tag_likelihood[prev_tag_prediction][tag])/prev_tag_total)
						else:
							current_probability = 0
						# if the word is unknown but contains a hyphen, the hyphenated probabilities should be considered
						if is_unknown and "-" in token:
							if tag in self.hyphenated_token_as_tag_likelihood:
								current_probability *= float(int(self.hyphenated_token_as_tag_likelihood[tag])/int(self.hyphenated_token_as_tag_likelihood["total"]))
							else:
								current_probability = 0
						# if the word is unknown but starts with a capital, the capital probabilities should be considered
						if is_unknown and token[0].isupper():
							if tag in self.capitalized_token_as_tag_likelihood:
								current_probability *= float(int(self.capitalized_token_as_tag_likelihood[tag])/int(self.capitalized_token_as_tag_likelihood["total"]))
							else:
								current_probability = 0
					if current_probability >= highest_probability:
						highest_probability = current_probability
						tag_prediction = tag
			prev_tag_prediction = tag_prediction # the current prediction becomes the previous prediction for the next prediction
			tag_predictions.append(tag_prediction) # add the prediction to the return array
		return tag_predictions

	# development function
	def set_model(self, data, minimum=DEFAULT_MIN_TOKEN_OCCURRENCES, to_lowercase=DEFAULT_TO_LOWERCASE):
		self.reset_vars()
		prev_tag = ""
		for line in data:
			if "\t" in line and len(line) > 2:
				token_and_tag = line.split("\t")
				token = token_and_tag[0].strip()
				if to_lowercase:
					token = token.lower()
				tag = token_and_tag[1].strip()
				# update token-tag likelihood
				if token in self.token_as_tag_likelihood:
					self.token_as_tag_likelihood[token]["total"] += 1
					if tag in self.token_as_tag_likelihood[token]:
						self.token_as_tag_likelihood[token][tag] += 1
					else:
						self.token_as_tag_likelihood[token][tag] = 1
				else:
					self.token_as_tag_likelihood[token] = {"total": 1, tag: 1}
				# check if token contains a certain suffix
				for suffix in self.SMOOTHING_SUFFIXES:
					if token.endswith(suffix):
						if suffix in self.suffixed_token_as_tag_likelihood:
							self.suffixed_token_as_tag_likelihood[suffix]["total"] += 1
							if tag in self.suffixed_token_as_tag_likelihood[suffix]:
								self.suffixed_token_as_tag_likelihood[suffix][tag] += 1
							else:
								self.suffixed_token_as_tag_likelihood[suffix][tag] = 1
						else:
							self.suffixed_token_as_tag_likelihood[suffix] = {"total": 1, tag: 1}
						break # break to avoid adding adding a token under multiple suffixes (i.e. "ity" and "ty")
				# check if token contains a hyphen
				if "-" in token:
					self.hyphenated_token_as_tag_likelihood["total"] += 1
					if tag in self.hyphenated_token_as_tag_likelihood:
						self.hyphenated_token_as_tag_likelihood[tag] += 1
					else:
						self.hyphenated_token_as_tag_likelihood[tag] = 1
				# check if token begins with capital letter
				if token[0].isupper():
					self.capitalized_token_as_tag_likelihood["total"] += 1
					if tag in self.capitalized_token_as_tag_likelihood:
						self.capitalized_token_as_tag_likelihood[tag] += 1
					else:
						self.capitalized_token_as_tag_likelihood[tag] = 1
				# update tag-tag likelihood
				if len(prev_tag) > 0:
					if prev_tag in self.tag_to_tag_likelihood:
						self.tag_to_tag_likelihood[prev_tag]["total"] += 1
						if tag in self.tag_to_tag_likelihood[prev_tag]:
							self.tag_to_tag_likelihood[prev_tag][tag] += 1
						else:
							self.tag_to_tag_likelihood[prev_tag][tag] = 1
					else:
						self.tag_to_tag_likelihood[prev_tag] = {"total": 1, tag: 1}
				prev_tag = tag
			else:
				prev_tag = ""
		# remove tokens seen less than the minimum required
		if minimum > 1:
			remove_token_keys = []
			for key in self.token_as_tag_likelihood.keys():
				if self.token_as_tag_likelihood[key]["total"] < minimum:
					remove_token_keys.append(key)
			# for all the keys to be removed, store as one "unknown" probability
			for key in remove_token_keys:
				for subkey in self.token_as_tag_likelihood[key]:
					if subkey in self.unknown_token_as_tag_likelihood:
						self.unknown_token_as_tag_likelihood[subkey] += self.token_as_tag_likelihood[key][subkey]
					else:
						self.unknown_token_as_tag_likelihood[subkey] = self.token_as_tag_likelihood[key][subkey]
				self.token_as_tag_likelihood.pop(key)
			# do the same for the suffixed tokens
			remove_suffixed_token_keys = []
			for key in self.suffixed_token_as_tag_likelihood.keys():
				if self.suffixed_token_as_tag_likelihood[key]["total"] < minimum:
					remove_suffixed_token_keys.append(key)
			for key in remove_suffixed_token_keys:
				self.suffixed_token_as_tag_likelihood.pop(key)

	# development function
	def save_model(self, save_path):
		token_as_tag_likelihood_string = ""
		for key in self.token_as_tag_likelihood:
			token_as_tag_likelihood_string += "TOKEN:\t%s\n" % key
			for subkey in self.token_as_tag_likelihood[key]:
				token_as_tag_likelihood_string += "%s %d\t" % (subkey, self.token_as_tag_likelihood[key][subkey])
			token_as_tag_likelihood_string = "%s\n" % token_as_tag_likelihood_string.strip()
		suffixed_token_as_tag_likelihood_string = ""
		for key in self.suffixed_token_as_tag_likelihood:
			suffixed_token_as_tag_likelihood_string += "SUFFIX:\t%s\n" % key
			for subkey in self.suffixed_token_as_tag_likelihood[key]:
				suffixed_token_as_tag_likelihood_string += "%s %d\t" % (subkey, self.suffixed_token_as_tag_likelihood[key][subkey])
			suffixed_token_as_tag_likelihood_string = "%s\n" % suffixed_token_as_tag_likelihood_string.strip()
		hyphenated_token_as_tag_likelihood_string = "HYPHEN:\tHYPHEN\n"
		for key in self.hyphenated_token_as_tag_likelihood:
			hyphenated_token_as_tag_likelihood_string += "%s %d\t" % (key, self.hyphenated_token_as_tag_likelihood[key])
		capitalized_token_as_tag_likelihood_string = "CAPITALIZED:\tCAPITALIZED\n"
		for key in self.capitalized_token_as_tag_likelihood:
			capitalized_token_as_tag_likelihood_string += "%s %d\t" % (key, self.capitalized_token_as_tag_likelihood[key])
		unknown_token_as_tag_likelihood_string = "UNKNOWN:\tUNKNOWN\n"
		for key in self.unknown_token_as_tag_likelihood:
			unknown_token_as_tag_likelihood_string += "%s %d\t" % (key, self.unknown_token_as_tag_likelihood[key])
		unknown_token_as_tag_likelihood_string = "%s\n" % unknown_token_as_tag_likelihood_string.strip()
		tag_to_tag_likelihood_string = ""
		for key in self.tag_to_tag_likelihood:
			tag_to_tag_likelihood_string += "TAG:\t%s\n" % key
			for subkey in self.tag_to_tag_likelihood[key]:
				tag_to_tag_likelihood_string += "%s %d\t" % (subkey, self.tag_to_tag_likelihood[key][subkey])
			tag_to_tag_likelihood_string = "%s\n" % tag_to_tag_likelihood_string.strip()
		model_string = "%s\n\n\n%s\n\n\n%s\n\n\n%s\n\n\n%s\n\n\n%s" % (token_as_tag_likelihood_string.strip(), suffixed_token_as_tag_likelihood_string.strip(), hyphenated_token_as_tag_likelihood_string.strip(), capitalized_token_as_tag_likelihood_string.strip(), unknown_token_as_tag_likelihood_string.strip(), tag_to_tag_likelihood_string.strip())
		# save the model
		with open(save_path, "w") as model_file:
			model_file.write(model_string)