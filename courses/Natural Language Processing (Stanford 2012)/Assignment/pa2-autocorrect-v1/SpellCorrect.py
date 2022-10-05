import math
from Datum import Datum
from Sentence import Sentence
from HolbrookCorpus import HolbrookCorpus
from UniformLanguageModel import UniformLanguageModel
from UnigramLanguageModel import UnigramLanguageModel
from StupidBackoffLanguageModel import StupidBackoffLanguageModel
from LaplaceUnigramLanguageModel import LaplaceUnigramLanguageModel
from LaplaceBigramLanguageModel import LaplaceBigramLanguageModel
from CustomLanguageModel import CustomLanguageModel
from EditModel import EditModel
from SpellingResult import SpellingResult
import types

# Modified version of Peter Norvig's spelling corrector
"""Spelling Corrector.

Copyright 2007 Peter Norvig. 
Open source code under MIT license: http://www.opensource.org/licenses/mit-license.php
"""

import re, collections

class SpellCorrect:
  """Holds edit model, language model, corpus. trains"""
  

  def __init__(self, lm, corpus):
    """initializes the language model."""
    self.languageModel = lm
    self.editModel = EditModel('../data/count_1edit.txt', corpus)


  def evaluate(self, corpus):  
    """Tests this speller on a corpus, returns a SpellingResult"""
    numCorrect = 0
    numTotal = 0
    testData = corpus.generateTestCases()
    for sentence in testData:
      out = ''
      if sentence.isEmpty():
        continue
      errorSentence = sentence.getErrorSentence()
      hypothesis = self.correctSentence(errorSentence)
      #print ' '.join(errorSentence) + '\t\t' + ' '.join(hypothesis) + '\t\t' + str(sentence.isCorrection(hypothesis))
      if sentence.isCorrection(hypothesis):
        numCorrect += 1
      numTotal += 1
    return SpellingResult(numCorrect, numTotal)

  def correctSentence(self, sentence):
    """Takes a list of words, returns a corrected list of words."""
    if len(sentence) == 0:
      return []
    argmax_i = 0
    argmax_w = sentence[0]
    maxscore = float('-inf')
    maxlm = float('-inf')
    maxedit = float('-inf')

    # skip start and end tokens
    for i in range(1, len(sentence) - 1):
      word = sentence[i]
      editProbs = self.editModel.editProbabilities(word)
      for alternative, editscore in editProbs.iteritems():
        if alternative == word:
          continue
        sentence[i] = alternative
        lmscore = self.languageModel.score(sentence)
        if editscore != 0:
          editscore = math.log(editscore)
        else:
          editscore = float('-inf')
        score = lmscore + editscore
        if score >= maxscore:
          maxscore = score
          maxlm = lmscore
          maxedit = editscore
          argmax_i = i
          argmax_w = alternative

      sentence[i] = word # restores sentence to original state before moving on
    argmax = list(sentence) # copy it
    argmax[argmax_i] = argmax_w # correct it
    return argmax


  def correctCorpus(self, corpus): 
    """Corrects a whole corpus, returns a JSON representation of the output."""
    string_list = [] # we will join these with commas,  bookended with []
    sentences = corpus.corpus
    for sentence in sentences:
      uncorrected = sentence.getErrorSentence()
      corrected = self.correctSentence(uncorrected) # List<String>
      word_list = '["%s"]' % '","'.join(corrected)
      string_list.append(word_list)
    output = '[%s]' % ','.join(string_list)
    return output


#import plotly.plotly as py
#import plotly.graph_objs as go
#import numpy as np
#from plotly.graph_objs import *
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#from plotly.graph_objs import Scatter, Figure, Layout
#import plotly
import seaborn as sns
       
def main():
  """Trains all of the language models and tests them on the dev data. Change devPath if you
     wish to do things like test on the training data."""
  '''
  #plotly.offline.init_notebook_mode() 
  #iplot([{"x": [1, 2, 3], "y": [3, 1, 6]}])
  
  #x = np.random.randn(2000)
  #y = np.random.randn(2000)
  #iplot([Histogram2dContour(x=x, y=y, contours=Contours(coloring='heatmap')),
  #     Scatter(x=x, y=y, mode='markers', marker=Marker(color='white', size=3, opacity=0.3))], show_link=False)
	   
  data = [go.Bar(
            x=['giraffes', 'orangutans', 'monkeys'],
            y=[20, 14, 23]
    )]

  py.iplot(data, filename='basic-bar') 
  '''
  sns.set_style("whitegrid")
  tips = sns.load_dataset("tips")
  ax = sns.barplot(x=["Uniform","Unigram", "LaplaceUnigram", "LaplaceBigram", "StupidBackoff", "Interpolation"], y=[0.065817, 0.061571, 0.110403, 0.135881, 0.180467, 0.193206])
  sns.plt.show()
	 
  trainPath = '../data/holbrook-tagged-train.dat'
  trainingCorpus = HolbrookCorpus(trainPath)

  devPath = '../data/holbrook-tagged-dev.dat'
  devCorpus = HolbrookCorpus(devPath)

  print 'Uniform Language Model: '
  uniformLM = UniformLanguageModel(trainingCorpus)
  uniformSpell = SpellCorrect(uniformLM, trainingCorpus)
  uniformOutcome = uniformSpell.evaluate(devCorpus) 
  print str(uniformOutcome)

  print 'Unigram Language Model: '
  unigramLM = UnigramLanguageModel(trainingCorpus)
  unigramSpell = SpellCorrect(unigramLM, trainingCorpus)
  unigramOutcome = unigramSpell.evaluate(devCorpus) 
  print str(unigramOutcome)

  print 'Laplace Unigram Language Model: ' 
  laplaceUnigramLM = LaplaceUnigramLanguageModel(trainingCorpus)
  laplaceUnigramSpell = SpellCorrect(laplaceUnigramLM, trainingCorpus)
  laplaceUnigramOutcome = laplaceUnigramSpell.evaluate(devCorpus)
  print str(laplaceUnigramOutcome)

  print 'Laplace Bigram Language Model: '
  laplaceBigramLM = LaplaceBigramLanguageModel(trainingCorpus)
  laplaceBigramSpell = SpellCorrect(laplaceBigramLM, trainingCorpus)
  laplaceBigramOutcome = laplaceBigramSpell.evaluate(devCorpus)
  print str(laplaceBigramOutcome)

  print 'Stupid Backoff Language Model: '  
  sbLM = StupidBackoffLanguageModel(trainingCorpus)
  sbSpell = SpellCorrect(sbLM, trainingCorpus)
  sbOutcome = sbSpell.evaluate(devCorpus)
  print str(sbOutcome)

  print 'Custom Language Model: '
  customLM = CustomLanguageModel(trainingCorpus, 0.17, 0.39)
  customSpell = SpellCorrect(customLM, trainingCorpus)
  customOutcome = customSpell.evaluate(devCorpus)
  print str(customOutcome)

  import matplotlib.pyplot as plt
  import numpy as np

  n = 10 #20 #10
  x = np.linspace(0,0.5,n)
  y = np.linspace(0,0.5,n)

  X, Y = np.meshgrid(x,y)
  Z = np.zeros((n,n))
  
  '''
  for i in range(n):
   for j in range(n):
     customLM = CustomLanguageModel(trainingCorpus, X[i,j], Y[i,j])
     customSpell = SpellCorrect(customLM, trainingCorpus)
     customOutcome = customSpell.evaluate(devCorpus)
     print X[i,j], Y[i,j], str(customOutcome)
     Z[i,j] = customOutcome.getAccuracy()
  '''

  Z = np.array([[0.09129512,0.15286624,0.16348195,0.15498938,0.16135881,0.15923567,0.1507431,0.14437367,0.14225053,0.14225053],[0.13375796,0.17409766,0.16772824,0.16348195,0.16772824,0.15711253,0.1507431,0.14649682,0.14437367,0.14225053],[0.14225053,0.17834395,0.16985138,0.16772824,0.17197452,0.16348195,0.16348195,0.15286624,0.14861996,0.14437367],[0.14437367,0.18259023,0.18259023,0.16985138,0.17622081,0.16985138,0.16135881,0.15923567,0.15498938,0.15286624],[0.1507431,0.18471338,0.18895966,0.18259023,0.17834395,0.17409766,0.16985138,0.1656051,0.16348195,0.15286624],[0.1507431,0.17622081,0.1910828,0.18471338,0.18259023,0.16772824,0.1656051,0.1656051,0.15711253,0.1507431],[0.1507431,0.18046709,0.18895966,0.18259023,0.18683652,0.18046709,0.16985138,0.16135881,0.15498938,0.14861996],[0.1507431,0.17834395,0.1910828,0.19320594,0.18471338,0.18046709,0.17197452,0.15923567,0.15498938,0.14649682],[0.14649682,0.17834395,0.18471338,0.17834395,0.1910828,0.17834395,0.17409766,0.1656051,0.15498938,0.14437367],[0.14437367,0.17409766,0.18471338,0.18683652,0.18683652,0.17834395,0.17197452,0.16772824,0.15711253,0.14861996]])
	   
  #print Z 
  #plt.pcolor(X, Y, Z)
  plt.imshow(Z.T, cmap=plt.cm.RdBu)
  plt.colorbar()
  
  for i in range(n):
    for j in range(n):
       plt.text(i, j, round(Z[i,j],2), va='center', ha='center', fontsize=10, color='black')

  plt.xticks(range(n), map(lambda z:round(z,2), x))
  plt.yticks(range(n), map(lambda z:round(z,2), y))
  
  plt.show()
  
if __name__ == "__main__":
    main()
