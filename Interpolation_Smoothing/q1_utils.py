import nltk
import math
import numpy



# Interplotation
# q(w|u, v) = λ1 × qML(w|u, v) + λ2 × qML(w|v) + λ3 × qML(w) where
# λ1 ≥ 0, λ2 ≥ 0, λ3 ≥ 0 and λ1 + λ2 + λ3 = 1
#qML(w|u, v) = c(u, v, w)/c(u, v)
#qML(w|v) = c(v, w)/c(v)
#qML(w) = c(w)/c()

#lambdas are calculated by maximizing the following on test set
# SUM over all u,v,w:
#	 {  c'(u, v, w)*log (λ1 × qML(w|u, v) + λ2 × qML(w|v) + λ3 × qML(w))  }
# where c'(u, v, w) is number of time trigram cvw appears in test set

# Also
#qML(w|u, v) = c(u, v, w)/c(u, v)
#qML(w|v) = c(v, w)/c(v)
#qML(w) = c(w)/c()
def compute_lamdas(path_to_validation_corpus,uni_freq_dist, bi_freq_dist, tri_freq_dist,C):
	validation_corpus = open(path_to_validation_corpus,'r').read()

	validation_word_tokens = nltk.word_tokenize(validation_corpus)

	validation_trigrams = list(nltk.trigrams(validation_word_tokens))
	validation_tri_freq_dist = nltk.FreqDist(validation_trigrams)

	lambda_values = []

	max_log_likelihood = -99999999
	for lambda1 in range(0,100,5):
		for lambda2 in range(0,100-lambda1,5):
			lambda3 = max(0,1-lambda1/100-lambda2/100)
			
			log_likelihood = 0
			for ngram,freq in validation_tri_freq_dist.items():
				qML_sum = 0
				if tri_freq_dist[ngram]:
					qML_sum = (lambda1/100) * tri_freq_dist[ngram]/bi_freq_dist[ngram[:2]]
			
				if bi_freq_dist[ngram[1:]]:
					qML_sum += (lambda2/100) * bi_freq_dist[ngram[1:]]/uni_freq_dist[ngram[1]]
			
				if uni_freq_dist[ngram[2]]:
					qML_sum += lambda3 * uni_freq_dist[ngram[2]]/C
				else:
					qML_sum += lambda3 * 100/C

				log_qML_sum = math.log(qML_sum)
				log_likelihood += freq*log_qML_sum

			if max_log_likelihood < log_likelihood:
				lambda_values = [lambda1/100,lambda2/100,lambda3]
				max_log_likelihood = log_likelihood

	return lambda_values



#l = 1/M * SUM { log p(xi) }
#preplexity = 2^-l
# M = total number of words in test corpus
# or M = sum(ni), ni is length of ith sentence
# p(xi) is the probability of x^i'th sentence is the product of all q(w|u,v)


# Interplotation
# q(w|u, v) = λ1 × qML(w|u, v) + λ2 × qML(w|v) + λ3 × qML(w) where
# λ1 ≥ 0, λ2 ≥ 0, λ3 ≥ 0 and λ1 + λ2 + λ3 = 1
#qML(w|u, v) = c(u, v, w)/c(u, v)
#qML(w|v) = c(v, w)/c(v)
#qML(w) = c(w)/c()


def find_Sentece_Probability(sentence,lambda_values, uni_freq_dist, bi_freq_dist, tri_freq_dist,C):


	lambda1 = lambda_values[0]
	lambda2 = lambda_values[1]
	lambda3 = lambda_values[2]

	word_tokens = nltk.word_tokenize(sentence)
	trigrams = nltk.trigrams(word_tokens)
	log_prob_sentece = 0

	for ngram in trigrams:
		qML_sum = 0
		if tri_freq_dist[ngram]:
			qML_sum = lambda1 * tri_freq_dist[ngram]/bi_freq_dist[ngram[:2]]
			
		if bi_freq_dist[ngram[1:]]:
			qML_sum += lambda2 * bi_freq_dist[ngram[1:]]/uni_freq_dist[ngram[1]]
			
		qML_sum += lambda3 * uni_freq_dist[ngram[2]]/C
		if qML_sum:
			log_prob_sentece += math.log(qML_sum,2)

	return log_prob_sentece

def find_perplexity(sentences, lambda_values, uni_freq_dist, bi_freq_dist, tri_freq_dist,M,C):
	
	sum_log_prob_sentece = 0
	count = 0
	total = len(list(sentences))
	for sentence in sentences:
		log_prob_sentece = find_Sentece_Probability(sentence,lambda_values, uni_freq_dist, bi_freq_dist, tri_freq_dist,C)

		#log_prob_sentece = math.log(prob_sentece,2)
		sum_log_prob_sentece += log_prob_sentece
		if count%1000 == 0:
			print("Percent completed: ",count/total)
		count+=1
	print("sum_log_prob_sentece is: ",sum_log_prob_sentece)
	print("M is:",M)
	l = 1/M * sum_log_prob_sentece
	print("l is: ",l)
	preplexity = 2**(-l)
	return preplexity

