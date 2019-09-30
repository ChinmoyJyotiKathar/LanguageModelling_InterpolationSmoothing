from random import shuffle
import nltk

def create_datasets(path_to_master_textfile):
	for j in range(5):
		with open(path_to_master_textfile,'r') as rf:
			corpus = rf.read()
			lines = nltk.sent_tokenize(corpus)
			shuffle(lines)
			with open('datasets/train'+str(j)+'.txt','w') as wf1:
				for i in range( int(0.8*len(lines)) ):
					wf1.write(lines[i]+'\n')
			with open('datasets/validation'+str(j)+'.txt','w') as wf2:
				for i in range( int(0.8*len(lines))+1,int(0.9*len(lines))):
					wf2.write(lines[i])
			with open('datasets/test'+str(j)+'.txt','w') as wf3:
				for i in range( int(0.9*len(lines))+1, len(lines)):
					wf3.write(lines[i])

create_datasets('./../xab2')