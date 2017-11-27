import argparse
import mm
import memm

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", help="the input file to be analyzed")
	parser.add_argument("-tr", "--train", action="store_true", help="trains the learning algorithm")
	parser.add_argument("-te", "--test", action="store_true", help="tests the learning algorithm")
	parser.add_argument("-tu", "--tune", action="store_true", help="tunes the learning algorithm")
	parser.add_argument("-m", "--model", help="the pre-trained model to be used for testing")
	parser.add_argument("-s", "--save", action="store_true", help="save the model after training")
	parser.add_argument("-min", "--minimum", type=int, help="tunes the minimum number of token occurrences to be considered by the model")
	parser.add_argument("-low", "--lowercase", action="store_true", help="tunes whether or not to convert all tokens to their lowercase form for the model")
	parser.add_argument("-e", "--epochs", type=int, help="tunes the maximum number of epochs to be used in training the maximum entropy markov model perceptron")
	parser.add_argument("-fe", "--features", type=int, help="tunes the minimum number of feature occurrences to be considered by the maximum entropy markov model")
	parser.add_argument("-mm", "--mm", action="store_true", help="run using visible markov model")
	parser.add_argument("-memm", "--memm", action="store_true", help="run using maximum entropy markov model with features as unigrams in a +/- 2 unigram window of tokens")
	args = parser.parse_args()
	if args.file:
		if args.train is False and args.test is False and args.tune is False:
			print("you must select an argument -tr, -te, or -tu (see --help for help)")
		elif args.mm is False and args.memm is False:
			print("you must select an argument -mm or -memm (see --help for help)")
		elif args.train and args.test is False and args.tune is False: # train the input
			if args.mm:
				m1 = mm.MM(model_path=None)
				with open(args.file) as data:
					m1.set_model(data)
					if args.save:
						m1.save_model("m1-data.txt")
					data.seek(0)
					test_accuracy_mm(m1, data, to_lowercase=m1.DEFAULT_TO_LOWERCASE)
			elif args.memm:
				m2 = memm.MEMM(model_path=None)
				with open(args.file) as data:
					m2.set_model(data)
					if args.save:
						m2.save_model("m2-data.txt")
					data.seek(0)
					test_accuracy_memm(m2, data, to_lowercase=m2.DEFAULT_TO_LOWERCASE)
		elif args.train is False and args.test and args.tune is False: # test the input
			if args.model:
				if args.mm:
					m1 = mm.MM(model_path=args.model)
					with open(args.file) as data:
						test_accuracy_mm(m1, data, to_lowercase=m1.DEFAULT_TO_LOWERCASE)
				elif args.memm:
					m2 = memm.MEMM(model_path=args.model)
					with open(args.file) as data:
						test_accuracy_memm(m2, data, to_lowercase=m2.DEFAULT_TO_LOWERCASE)
			else:
				print("you must select a pre-trained model to be tested using the argument -m (see --help for help)")
		elif args.train is False and args.test is False and args.tune: # tune the input
			with open(args.file) as data:
				if args.mm:
					m1 = mm.MM(model_path=None)
					minimum = m1.DEFAULT_MIN_TOKEN_OCCURRENCES
					if args.minimum:
						minimum = args.minimum
					if minimum < 2:
						print("the minimum number of token occurrences to be considered by the model must be at least 2")
					else:
						to_lowercase = False
						if args.lowercase:
							to_lowercase = True
						m1.set_model(data, minimum=minimum, to_lowercase=to_lowercase)
						data.seek(0)
						test_accuracy_mm(m1, data, to_lowercase=m1.DEFAULT_TO_LOWERCASE)
				elif args.memm:
					m2 = memm.MEMM(model_path=None)
					minimum = m2.DEFAULT_MIN_TOKEN_OCCURRENCES
					if args.minimum:
						minimum = args.minimum
					if minimum < 2:
						print("the minimum number of token occurrences to be considered by the model must be at least 2")
					else:
						to_lowercase = False
						if args.lowercase:
							to_lowercase = True
						max_epochs = m2.DEFAULT_MAX_EPOCHS
						if args.epochs > 0:
							max_epochs = args.epochs
						minimum_for_feature = m2.DEFAULT_MIN_FEATURE_OCCURRENCES
						if args.features > 0:
							minimum_for_feature = args.features
						m2.set_model(data, minimum_for_token=minimum, minimum_for_feature=minimum_for_feature, to_lowercase=to_lowercase, max_epochs=max_epochs)
						data.seek(0)
						test_accuracy_memm(m2, data, to_lowercase=m2.DEFAULT_TO_LOWERCASE)
		else:
			print("you must select only one argument -tr, -te, or -tu (see --help for help)")
	else:
		print("you must select a file to be analyzed using the argument -f (see --help for help)")

def test_accuracy_mm(model, data, to_lowercase):
	tag_predictions = []
	sentence = [] # model analyzes one sentence at a time
	# get tag predictions first
	for line in data:
		if "\t" in line and len(line) > 2:
			token_and_tag = line.split("\t")
			token = token_and_tag[0].strip()
			if to_lowercase:
				token = token.lower()
			sentence.append(token)
		else:
			tag_predictions.extend(model.get_pos_tags(sentence, to_lowercase=to_lowercase))
			sentence = [] # reset sentence
	if len(sentence) > 0: # handle last sentence if data file does not end in new line
		tag_predictions.extend(model.get_pos_tags(sentence, to_lowercase=to_lowercase))
	data.seek(0)
	overall_correct = 0 # track correct predictions during testing
	overall_incorrect = 0 # track incorrect predictions during testing
	unknown_correct = 0 # track correct predictions for unknown tokens during testing
	unknown_incorrect = 0 # track incorrect predictions for unknown tokens during testing
	# analyze the predictions to determine how many correct vs. incorrect
	i = 0
	for line in data:
		if "\t" in line and len(line) > 2:
			token_and_tag = line.split("\t")
			token = token_and_tag[0].strip()
			if to_lowercase:
				token = token.lower()
			actual_tag = token_and_tag[1].strip()
			predicted_tag = tag_predictions[i]
			if actual_tag == predicted_tag:
				overall_correct += 1
				if token not in model.token_as_tag_likelihood:
					unknown_correct += 1
			else:
				overall_incorrect += 1
				if token not in model.token_as_tag_likelihood:
					unknown_incorrect += 1
			i += 1
	print("Overall correct: %d (%.3f)" % (overall_correct, float(overall_correct/(overall_incorrect+overall_correct))))
	print("Overall incorrect: %d (%.3f)" % (overall_incorrect, float(overall_incorrect/(overall_incorrect+overall_correct))))
	print("Unknown correct: %d (%.3f)" % (unknown_correct, float(unknown_correct/(unknown_incorrect+unknown_correct))))
	print("Unknown incorrect: %d (%.3f)" % (unknown_incorrect, float(unknown_incorrect/(unknown_incorrect+unknown_correct))))

def test_accuracy_memm(model, data, to_lowercase):
	tag_predictions = []
	sentence = [] # model analyzes one sentence at a time
	# get tag predictions first
	for line in data:
		if "\t" in line and len(line) > 2:
			token_and_tag = line.split("\t")
			token = token_and_tag[0].strip()
			if to_lowercase:
				token = token.lower()
			sentence.append(token)
		else:
			tag_predictions.extend(model.get_pos_tags(sentence, to_lowercase=to_lowercase))
			sentence = [] # reset sentence
	if len(sentence) > 0: # handle last sentence if data file does not end in new line
		tag_predictions.extend(model.get_pos_tags(sentence, to_lowercase=to_lowercase))
	data.seek(0)
	overall_correct = 0 # track correct predictions during testing
	overall_incorrect = 0 # track incorrect predictions during testing
	unknown_correct = 0 # track correct predictions for unknown tokens during testing
	unknown_incorrect = 0 # track incorrect predictions for unknown tokens during testing
	# analyze the predictions to determine how many correct vs. incorrect
	i = 0
	for line in data:
		if "\t" in line and len(line) > 2:
			token_and_tag = line.split("\t")
			token = token_and_tag[0].strip()
			if to_lowercase:
				token = token.lower()
			actual_tag = token_and_tag[1].strip()
			predicted_tag = tag_predictions[i]
			if actual_tag == predicted_tag:
				overall_correct += 1
				if token not in model.token_dictionary:
					unknown_correct += 1
			else:
				overall_incorrect += 1
				if token not in model.token_dictionary:
					unknown_incorrect += 1
			i += 1
	print("Overall correct: %d (%.3f)" % (overall_correct, float(overall_correct/(overall_incorrect+overall_correct))))
	print("Overall incorrect: %d (%.3f)" % (overall_incorrect, float(overall_incorrect/(overall_incorrect+overall_correct))))
	print("Unknown correct: %d (%.3f)" % (unknown_correct, float(unknown_correct/(unknown_incorrect+unknown_correct))))
	print("Unknown incorrect: %d (%.3f)" % (unknown_incorrect, float(unknown_incorrect/(unknown_incorrect+unknown_correct))))

parse_args()