import collections, math

class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.trigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
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
    prev_token1, prev_token2 = None, None
    alpha = 0.4 #1
    for token in sentence:
      if prev_token2 == None:
          count = self.unigramCounts[token]
          score += math.log(count + 1)
          score -= math.log(self.total + len(self.unigramCounts))
      elif prev_token1 == None:
          count2 = self.bigramCounts[prev_token2,token]
          count1 = self.unigramCounts[prev_token2]
          if count2 > 0:
			  score += math.log(count2)
			  score -= math.log(count1)
          else:
			  count = self.unigramCounts[token]
			  score += alpha*math.log(count + 1)
			  score -= alpha*math.log(self.total + len(self.unigramCounts))
      else:
          count3 = self.trigramCounts[prev_token1,prev_token2,token]
          count2 = self.bigramCounts[prev_token1,prev_token2]
          if count3 > 0:
			  score += math.log(count3)
			  score -= math.log(count2)
          else:
			  count2 = self.bigramCounts[prev_token2,token]
			  count1 = self.unigramCounts[prev_token2]
			  if count2 > 0:
				  score += alpha*math.log(count2)
				  score -= alpha*math.log(count1)
			  else:
				  count = self.unigramCounts[token]
				  score += alpha*math.log(count + 1)
				  score -= alpha*math.log(self.total + len(self.unigramCounts))
      prev_token1, prev_token2 = prev_token2, token
        
    return score