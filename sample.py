import mm
import memm

def main():
	tokens = ["I", "enjoy", "eating", "sweet", "apples", "dipped", "in", "warm", "honey", "."]
	raw_tokens_string = ""
	for token in tokens:
		raw_tokens_string += "%s " % token
	raw_tokens_string = raw_tokens_string.strip()
	print("Raw tokens: %s" % raw_tokens_string)
	m1 = mm.MM()
	tags_1 = m1.get_pos_tags(tokens)
	tokens_and_tags_string_1 = ""
	i = 0
	while i < len(tokens):
		token = tokens[i]
		tag = tags_1[i]
		tokens_and_tags_string_1 += "%s (%s) " % (token, tag)
		i += 1
	tokens_and_tags_string_1 = tokens_and_tags_string_1.strip()
	print("Tokens and POS tags from MM: %s" % tokens_and_tags_string_1)
	m2 = memm.MEMM()
	tags_2 = m2.get_pos_tags(tokens)
	tokens_and_tags_string_2 = ""
	i = 0
	while i < len(tokens):
		token = tokens[i]
		tag = tags_2[i]
		tokens_and_tags_string_2 += "%s (%s) " % (token, tag)
		i += 1
	tokens_and_tags_string_2 = tokens_and_tags_string_2.strip()
	print("Tokens and POS tags from MEMM: %s" % tokens_and_tags_string_2)

main()