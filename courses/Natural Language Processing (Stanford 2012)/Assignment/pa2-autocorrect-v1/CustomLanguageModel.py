import collections, math

class CustomLanguageModel:

  def __init__(self, corpus, lambda1=0.2, lambda2=0.45):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.trigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.lambda1 = lambda1	
    self.lambda2 = lambda2	
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
      prev_token1, prev_token2 = None, None
      for datum in sentence.data:  
        token = datum.word
        self.unigramCounts[token] += 1
        self.total += 1  
        if prev_token2:    
           self.bigramCounts[prev_token2, token] += 1
        if prev_token1:    
           self.trigramCounts[prev_token1, prev_token2, token] += 1
        prev_token1, prev_token2 = prev_token2, token

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0 
    lambda1	= self.lambda1
    lambda2	= self.lambda2
    lambda3 = 1. - lambda1 - lambda2
    prev_token1, prev_token2 = None, None
    for token in sentence:
        count3 = self.trigramCounts[prev_token1,prev_token2,token]
        count2 = self.bigramCounts[prev_token1,prev_token2]
        score3 = math.log(count3 + 1) - math.log(count2 + len(self.bigramCounts))
        count2 = self.bigramCounts[prev_token2,token]
        count1 = self.unigramCounts[prev_token2]
        score2 = math.log(count2 + 1) - math.log(count1 + len(self.unigramCounts))
        count = self.unigramCounts[token]
        score1 = math.log(count + 1) - math.log(self.total + len(self.unigramCounts))
        score += lambda3*score3 + lambda2*score2 + lambda1*score1
        prev_token1, prev_token2 = prev_token2, token
        
    return score
