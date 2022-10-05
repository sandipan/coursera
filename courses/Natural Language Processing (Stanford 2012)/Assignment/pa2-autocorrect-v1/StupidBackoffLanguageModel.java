import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StupidBackoffLanguageModel implements LanguageModel {

  protected Map<String, Integer> wordFreqMap; // map of words that occur in training with their corresponding frequencies
  protected Map<String, Integer> wordPairFreqMap; // map of word pairs that occur in training with their corresponding frequencies
  int totWordType = 0;	
  int totWordFreq = 0; // total # of words in the training corpus

  /** Initialize your data structures in the constructor. */
  public StupidBackoffLanguageModel(HolbrookCorpus corpus) {
    wordFreqMap = new HashMap<String, Integer>();
    wordPairFreqMap = new HashMap<String, Integer>();
    train(corpus);
  }

  /** Takes a corpus and trains your language model. 
    * Compute any counts or other corpus statistics in this function.
    */
  public void train(HolbrookCorpus corpus) {
	for(Sentence sentence : corpus.getData()) { // iterate over sentences
      String word = null, prevWord = null, wordPair = null;
	  for(Datum datum : sentence) { // iterate over words
        prevWord = word;
		word = datum.getWord(); // get the actual word
		Integer freq = wordFreqMap.get(word);
		int f = freq == null ? 1 : freq.intValue() + 1;
        wordFreqMap.put(word, f);
		if (prevWord != null) {
			wordPair = prevWord + "," + word;
			freq = wordPairFreqMap.get(wordPair);
			f = freq == null ? 1 : freq.intValue() + 1;
			wordPairFreqMap.put(wordPair, f);
		}
      }
    }
	totWordType = wordFreqMap.size();
	for (Map.Entry<String, Integer> entry : wordFreqMap.entrySet()) { // iterate over map
		totWordFreq += entry.getValue().intValue();
	}
  }

  /** Takes a list of strings as argument and returns the log-probability of the 
    * sentence using your language model. Use whatever data you computed in train() here.
    */
  public double score(List<String> sentence) {
    double score = 0.0;
	String prevWord = null;
    int fpair, f, fprev;
	fpair = f = fprev = 0;
	double probability = 0.0;
	for(String word : sentence) { // iterate over words in the sentence
	  Integer freq = wordFreqMap.get(word);
	  f = freq == null ? 0 : freq.intValue();
	  if (prevWord != null) {
		  String wordPair = prevWord + "," + word;
		  Integer freqPair = wordPairFreqMap.get(wordPair), freqPrev = wordFreqMap.get(prevWord);
		  fpair = freqPair == null ? 0 : freqPair.intValue();
		  fprev = freqPrev == null ? 0 : freqPrev.intValue();
		  probability = Math.log(fpair > 0 ? (fpair * 1.0) / fprev : (f + 1.0) / (totWordFreq + totWordType));
	  }
	  else {
	  	  probability = Math.log((f + 1.0) / (totWordFreq + totWordType));
 	  }
	  prevWord = word;
      score += probability;
   }
    return score;
  }
}
