class MM:

	DEFAULT_MODEL_PATH = "mm-model.txt" # the default path of the best model to be used
	DEFAULT_MIN_TOKEN_OCCURRENCES = 2 # the default minimum amount of occurrences for a token to appear to be considered by the model
	DEFAULT_MIN_TAG_TO_TOKEN_OCCURRENCES = 100 # the default minimum amount of occurrences for a token to appear with a tag before to be considered by the model
	DEFAULT_TO_LOWERCASE = False # the default of whether or not to convert all tokens to their lowercase form for the model
	SMOOTHING_SUFFIXES = ["acy", "al", "ance", "ence", "dom", "er", "or", "ism", "ist", "ity", "ty", "ment", "ness", "ship", "ation", "ition", "sion", "tion", "ion", "ate", "en", "ify", "fy", "ize", "ise", "able", "ible", "ial", "esque", "ful", "ic", "ical", "ious", "eous", "ous", "ish", "ative", "itive", "ive", "less", "ing", "est", "ly", "y", "ed", "es", "s"] # suffixes to check for in the training set to be used for unknown words with the same suffix in testing
	
	def __init__(self, model_path=DEFAULT_MODEL_PATH):
		self.reset_vars()
		if model_path is not None:
			with open(model_path) as model:
				self.load_model(model)

	def reset_vars(self):
		self.token_as_tag_likelihood = {} # the counts of occurrences of all tags of which a token is seen
		self.suffixed_token_as_tag_likelihood = {} # the counts of occurrences of all tags of which a token ending in a certain suffix is seen
		self.number_token_as_tag_likelihood = {"total":0} # the counts of occurrences of all tags of which a token contains a number
		self.hyphenated_token_as_tag_likelihood = {"total":0} # the counts of occurrences of all tags of which a token with a hyphen is seen
		self.capitalized_token_as_tag_likelihood = {"total":0} # the counts of occurrences of all tags of which a token starting with a capital character is seen
		self.unknown_token_as_tag_likelihood = {} # the counts of occurrences of all tags of which an unknown token is seen
		self.tag_to_tag_likelihood = {} # the counts of tags following a given tag
		self.tag_to_tag_to_tag_likelihood = {} # the counts of tags following a given tag following a given tag
		self.bigram_tokens_as_tags_likelihood = {} # the counts of occurrences of all tags of which a bigram is seen

	def load_model(self, model):
		IS_TOKEN = 1
		IS_SUFFIX = 2
		IS_NUMBER = 3
		IS_HYPHEN = 4
		IS_CAPITALIZED = 5
		IS_UNKNOWN = 6
		IS_TAG = 7
		IS_TRAG = 8
		IS_BIGRAM = 9
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
				elif "NUMBER:\tNUMBER" in line:
					current_state = IS_NUMBER
				elif "HYPHEN:\tHYPHEN" in line:
					current_state = IS_HYPHEN
				elif "CAPITALIZED:\tCAPITALIZED" in line:
					current_state = IS_CAPITALIZED
				elif "UNKNOWN:\tUNKNOWN" in line:
					current_state = IS_UNKNOWN
				elif "TAG:\t" in line:
					current_state = IS_TAG
					current_key = line.split("TAG:\t")[1].strip()
				elif "TRAG:\t" in line:
					current_state = IS_TRAG
					current_key = line.split("TRAG:\t")[1].strip()
				elif "BIGRAM:\t" in line:
					current_state = IS_BIGRAM
					current_key = line.split("BIGRAM:\t")[1].strip()
				elif "total " in line: # handle data
					data = line.split("\t")
					if current_state == IS_TRAG or current_state == IS_BIGRAM:
						bigram_tags_and_count_dict = {}
						for datum in data:
							bigram_tags_and_count = datum.split(" ")
							if len(bigram_tags_and_count) > 2:
								first_tag = bigram_tags_and_count[0].strip()
								second_tag = bigram_tags_and_count[1].strip()
								count = int(bigram_tags_and_count[2].strip())
								bigram_tags = "%s %s" % (first_tag, second_tag)
								bigram_tags_and_count_dict[bigram_tags] = count
							elif len(bigram_tags_and_count) > 1:
								total = bigram_tags_and_count[0].strip()
								count = int(bigram_tags_and_count[1].strip())
								bigram_tags_and_count_dict[total] = count
						if current_state == IS_TRAG:
							self.tag_to_tag_to_tag_likelihood[current_key] = bigram_tags_and_count_dict
						elif current_state == IS_BIGRAM:
							self.bigram_tokens_as_tags_likelihood[current_key] = bigram_tags_and_count_dict
					else:
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
						elif current_state == IS_NUMBER:
							self.number_token_as_tag_likelihood = tag_and_count_dict
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
		for i, token in enumerate(sentence):
			prev_tag_prediction = None
			if len(tag_predictions) > 0:
				prev_tag_prediction = tag_predictions[len(tag_predictions)-1]
			two_prev_tag_prediction = None
			if len(tag_predictions) > 1:
				two_prev_tag_prediction = tag_predictions[len(tag_predictions)-2]
			prev_token = None
			if i > 0:
				prev_token = sentence[i-1]
			next_token = None
			if i+1 < len(sentence):
				next_token = sentence[i+1]
			highest_probability = float(0)
			tag_prediction = ""
			tag_likelihoods = self.get_pos_tag_likelihoods_for_token(token, prev_token, next_token, prev_tag_prediction, two_prev_tag_prediction, to_lowercase=to_lowercase)
			for tag in tag_likelihoods:
				tag_likelihood = tag_likelihoods[tag]
				if i+1 < len(sentence):
					next_token = sentence[i+1]
					third_token = None
					if i+2 < len(sentence):
						third_token = sentence[i+2]
					next_tag_likelihoods = self.get_pos_tag_likelihoods_for_token(next_token, token, third_token, tag, prev_tag_prediction, to_lowercase=to_lowercase)
					for next_tag in next_tag_likelihoods:
						next_tag_likelihood = next_tag_likelihoods[next_tag]
						if i+2 < len(sentence):
							third_token = sentence[i+2]
							fourth_token = None
							if i+3 < len(sentence):
								fourth_token = sentence[i+3]
							third_tag_likelihoods = self.get_pos_tag_likelihoods_for_token(third_token, next_token, fourth_token, next_tag, tag, to_lowercase=to_lowercase)
							for third_tag in third_tag_likelihoods:
								third_tag_likelihood = third_tag_likelihoods[third_tag]
								if third_tag_likelihood*next_tag_likelihood*tag_likelihood >= highest_probability:
									highest_probability = third_tag_likelihood*next_tag_likelihood*tag_likelihood
									tag_prediction = tag
						elif next_tag_likelihood*tag_likelihood >= highest_probability:
							highest_probability = next_tag_likelihood*tag_likelihood
							tag_prediction = tag
				elif tag_likelihood >= highest_probability:
					highest_probability = tag_likelihood
					tag_prediction = tag
			tag_predictions.append(tag_prediction) # add the prediction to the return array
		return tag_predictions

	def get_pos_tag_likelihoods_for_token(self, token, prev_token, next_token, prev_tag, two_prev_tag, to_lowercase=DEFAULT_TO_LOWERCASE):
		pos_tag_likelihoods = {}
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
		for tag in current_token_as_tag_likelihood:
			if tag != "total":
				current_probability = float(int(current_token_as_tag_likelihood[tag])/token_total)
				if prev_tag is not None and len(prev_tag) > 0:
					if two_prev_tag is not None and len(two_prev_tag) > 0 and "%s %s" % (two_prev_tag, prev_tag) in self.tag_to_tag_to_tag_likelihood:
						prev_tags = "%s %s" % (two_prev_tag, prev_tag)
						prev_tags_total = int(self.tag_to_tag_to_tag_likelihood[prev_tags]["total"])
						if tag in self.tag_to_tag_to_tag_likelihood[prev_tags]:
							current_probability *= float(int(self.tag_to_tag_to_tag_likelihood[prev_tags][tag])/prev_tags_total)
						else:
							current_probability = 0
					else:
						prev_tag_total = int(self.tag_to_tag_likelihood[prev_tag]["total"])
						if tag in self.tag_to_tag_likelihood[prev_tag]:
							current_probability *= float(int(self.tag_to_tag_likelihood[prev_tag][tag])/prev_tag_total)
						else:
							current_probability = 0
					# if the word is unknown but contains a number, the number probabilities should be considered
					if is_unknown and self.is_number(token):
						if tag in self.number_token_as_tag_likelihood:
							current_probability *= float(int(self.number_token_as_tag_likelihood[tag])/int(self.number_token_as_tag_likelihood["total"]))
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
				# check if the previous token and current token form a known bigram
				if prev_token is not None and len(prev_token) > 0 and prev_tag is not None and len(prev_tag) > 0:
					bigram_tokens = "%s %s" % (prev_token, token)
					if bigram_tokens in self.bigram_tokens_as_tags_likelihood:
						bigram_tokens_total = int(self.bigram_tokens_as_tags_likelihood[bigram_tokens]["total"])
						bigram_tags = "%s %s" % (prev_tag, tag)
						if bigram_tags in self.bigram_tokens_as_tags_likelihood[bigram_tokens]:
							current_probability *= float(int(self.bigram_tokens_as_tags_likelihood[bigram_tokens][bigram_tags])/int(bigram_tokens_total))
						else:
							current_probability = 0
				# check if the token and the next token form a known bigram
				if next_token is not None and len(next_token) > 0:
					bigram_tokens = "%s %s" % (token, next_token)
					if bigram_tokens in self.bigram_tokens_as_tags_likelihood:
						bigram_tokens_total = int(self.bigram_tokens_as_tags_likelihood[bigram_tokens]["total"])
						bigram_first_tag_same_total = 0
						for bigram_tags in self.bigram_tokens_as_tags_likelihood[bigram_tokens]: # since the next token tag is unknown, check all of the tag bigrams where the first tag is the checked tag
							if " " in bigram_tags:
								first_tag = bigram_tags.split(" ")[0].strip()
								if first_tag == tag:
									bigram_first_tag_same_total += self.bigram_tokens_as_tags_likelihood[bigram_tokens][bigram_tags]
						if bigram_first_tag_same_total > 0:
							current_probability *= float(int(bigram_first_tag_same_total)/int(bigram_tokens_total))
						else:
							current_probability = 0
				if current_probability > 0:
					pos_tag_likelihoods[tag] = current_probability
		if len(pos_tag_likelihoods) == 0:
			for tag in current_token_as_tag_likelihood: # if all the tag likelihoods were reset to 0, figure out a tie-breaker
				if tag != "total":
					current_probability = float(int(current_token_as_tag_likelihood[tag])/int(current_token_as_tag_likelihood["total"]))
					if prev_tag is not None and len(prev_tag) > 0:
						prev_tag_total = int(self.tag_to_tag_likelihood[prev_tag]["total"])
						if tag in self.tag_to_tag_likelihood[prev_tag]:
							current_probability *= float(int(self.tag_to_tag_likelihood[prev_tag][tag])/prev_tag_total)
						else:
							current_probability = 0
					pos_tag_likelihoods[tag] = current_probability
		return pos_tag_likelihoods

	# development function
	def set_model(self, data, minimum=DEFAULT_MIN_TOKEN_OCCURRENCES, to_lowercase=DEFAULT_TO_LOWERCASE):
		self.reset_vars()
		two_prev_tag = ""
		prev_tag = ""
		prev_token = ""
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
				# check if token contains a number
				if self.is_number(token):
					self.number_token_as_tag_likelihood["total"] += 1
					if tag in self.number_token_as_tag_likelihood:
						self.number_token_as_tag_likelihood[tag] += 1
					else:
						self.number_token_as_tag_likelihood[tag] = 1
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
				if prev_tag is not None and len(prev_tag) > 0:
					if prev_tag in self.tag_to_tag_likelihood:
						self.tag_to_tag_likelihood[prev_tag]["total"] += 1
						if tag in self.tag_to_tag_likelihood[prev_tag]:
							self.tag_to_tag_likelihood[prev_tag][tag] += 1
						else:
							self.tag_to_tag_likelihood[prev_tag][tag] = 1
					else:
						self.tag_to_tag_likelihood[prev_tag] = {"total": 1, tag: 1}
				# update tag-tag-tag likelihood
				if prev_tag is not None and len(prev_tag) > 0 and two_prev_tag is not None and len(two_prev_tag) > 0:
					prev_tags = "%s %s" % (two_prev_tag, prev_tag)
					if prev_tags in self.tag_to_tag_to_tag_likelihood:
						self.tag_to_tag_to_tag_likelihood[prev_tags]["total"] += 1
						if tag in self.tag_to_tag_to_tag_likelihood[prev_tags]:
							self.tag_to_tag_to_tag_likelihood[prev_tags][tag] += 1
						else:
							self.tag_to_tag_to_tag_likelihood[prev_tags][tag] = 1
					else:
						self.tag_to_tag_to_tag_likelihood[prev_tags] = {"total": 1, tag: 1}
				# update bigram tags likelihood
				if prev_tag is not None and len(prev_tag) > 0 and prev_token is not None and len(prev_token) > 0:
					bigram_tokens = "%s %s" % (prev_token, token)
					bigram_tags = "%s %s" % (prev_tag, tag)
					if bigram_tokens in self.bigram_tokens_as_tags_likelihood:
						self.bigram_tokens_as_tags_likelihood[bigram_tokens]["total"] += 1
						if bigram_tags in self.bigram_tokens_as_tags_likelihood[bigram_tokens]:
							self.bigram_tokens_as_tags_likelihood[bigram_tokens][bigram_tags] += 1
						else:
							self.bigram_tokens_as_tags_likelihood[bigram_tokens][bigram_tags] = 1
					else:
						self.bigram_tokens_as_tags_likelihood[bigram_tokens] = {"total": 1, bigram_tags: 1}
				two_prev_tag = prev_tag
				prev_tag = tag
				prev_token = token
			else:
				two_prev_tag = ""
				prev_tag = ""
				prev_token = ""
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
			# do the same for the bigram tokens
			remove_bigram_tokens_keys = []
			for key in self.bigram_tokens_as_tags_likelihood.keys():
				if self.bigram_tokens_as_tags_likelihood[key]["total"] < minimum:
					remove_bigram_tokens_keys.append(key)
			for key in remove_bigram_tokens_keys:
				self.bigram_tokens_as_tags_likelihood.pop(key)
			# do the same for tag to tag to tags
			remove_tag_to_tag_to_tag_keys = []
			for key in self.tag_to_tag_to_tag_likelihood.keys():
				if self.tag_to_tag_to_tag_likelihood[key]["total"] < minimum:
					remove_tag_to_tag_to_tag_keys.append(key)
			for key in remove_tag_to_tag_to_tag_keys:
				self.tag_to_tag_to_tag_likelihood.pop(key)

	def is_number(self, string):
		for char in string:
			if char.isdigit() or char in ['-', ':', ',', "."]:
				continue
			else:
				return False
		return True

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
		number_token_as_tag_likelihood_string = "NUMBER:\tNUMBER\n"
		for key in self.number_token_as_tag_likelihood:
			number_token_as_tag_likelihood_string += "%s %d\t" % (key, self.number_token_as_tag_likelihood[key])
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
		tag_to_tag_to_tag_likelihood_string = ""
		for key in self.tag_to_tag_to_tag_likelihood:
			tag_to_tag_to_tag_likelihood_string += "TRAG:\t%s\n" % key # TRAG = trigram tags (prevent including "TAG" in key title)
			for subkey in self.tag_to_tag_to_tag_likelihood[key]:
				tag_to_tag_to_tag_likelihood_string += "%s %d\t" % (subkey, self.tag_to_tag_to_tag_likelihood[key][subkey])
			tag_to_tag_to_tag_likelihood_string = "%s\n" % tag_to_tag_to_tag_likelihood_string.strip()
		bigram_tokens_as_tags_likelihood_string = ""
		for key in self.bigram_tokens_as_tags_likelihood:
			bigram_tokens_as_tags_likelihood_string += "BIGRAM:\t%s\n" % key
			for subkey in self.bigram_tokens_as_tags_likelihood[key]:
				bigram_tokens_as_tags_likelihood_string += "%s %d\t" % (subkey, self.bigram_tokens_as_tags_likelihood[key][subkey])
			bigram_tokens_as_tags_likelihood_string = "%s\n" % bigram_tokens_as_tags_likelihood_string.strip()
		model_string = "%s\n\n\n%s\n\n\n%s\n\n\n%s\n\n\n%s\n\n\n%s\n\n\n%s\n\n\n%s\n\n\n%s" % (token_as_tag_likelihood_string.strip(), suffixed_token_as_tag_likelihood_string.strip(), number_token_as_tag_likelihood_string.strip(), hyphenated_token_as_tag_likelihood_string.strip(), capitalized_token_as_tag_likelihood_string.strip(), unknown_token_as_tag_likelihood_string.strip(), tag_to_tag_likelihood_string.strip(), tag_to_tag_to_tag_likelihood_string.strip(), bigram_tokens_as_tags_likelihood_string.strip())
		# save the model
		with open(save_path, "w") as model_file:
			model_file.write(model_string)