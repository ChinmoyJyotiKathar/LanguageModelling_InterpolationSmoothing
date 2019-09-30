import nltk
from q1_utils import compute_lamdas,find_perplexity


for i in range(5):
	train_set = './datasets/train'+str(i)+'.txt'
	test_set = './datasets/test'+str(i)+'.txt'
	validation_set = './datasets/validation'+str(i)+'.txt'

	
	corpus= open(train_set,'r').read()
	test_corpus= open(test_set,'r').read()

	word_tokens_train = nltk.word_tokenize(corpus)

	word_tokens_test = list(nltk.word_tokenize(test_corpus))
	sentences = nltk.sent_tokenize(test_corpus)

	unigrams = list(word_tokens_train)
	bigrams = list(nltk.bigrams(word_tokens_train,pad_left=True, pad_right=True))
	trigrams = list(nltk.trigrams(word_tokens_train,pad_left=True, pad_right=True))


	uni_freq_dist = nltk.FreqDist(unigrams)
	bi_freq_dist = nltk.FreqDist(bigrams)
	tri_freq_dist = nltk.FreqDist(trigrams)

	#lamda_values = [0.33,0.33,0.33]
	#remove the following to obtain computed lambda values
	lamda_values = compute_lamdas(validation_set,uni_freq_dist, bi_freq_dist, tri_freq_dist,len(unigrams))
	print(lamda_values)
	perplexity  = find_perplexity(sentences, lamda_values,
									uni_freq_dist, bi_freq_dist,
									tri_freq_dist, len(word_tokens_test), len(unigrams))
	print('perplexity for dataset pair ',i,': ',perplexity)
