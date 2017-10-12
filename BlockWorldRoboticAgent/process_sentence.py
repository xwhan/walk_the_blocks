import nltk

f = open('training_instructions.txt')
lens = []
for line in f:
	token_s = nltk.word_tokenize(line)
	lens.append(len(token_s))

print max(lens)