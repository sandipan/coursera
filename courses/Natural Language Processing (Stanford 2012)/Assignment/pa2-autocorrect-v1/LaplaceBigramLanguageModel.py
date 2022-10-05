import collections, math

class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
      prev_token = None
      for datum in sentence.data:  
        token = datum.word
        self.unigramCounts[token] += 1
        self.total += 1  
        if prev_token:    
           self.bigramCounts[prev_token, token] += 1
        prev_token = token

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0 
    prev_token = None
    for token in sentence:
      if prev_token == None:
          count = self.unigramCounts[token]
          score += math.log(count + 1)
          score -= math.log(self.total + len(self.unigramCounts))
      else:
          count2 = self.bigramCounts[prev_token,token]
          count1 = self.unigramCounts[prev_token]
          score += math.log(count2 + 1)
          score -= math.log(count1 + len(self.unigramCounts))
      prev_token = token
        
    return score