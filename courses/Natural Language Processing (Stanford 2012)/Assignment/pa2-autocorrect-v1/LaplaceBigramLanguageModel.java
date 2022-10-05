import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LaplaceBigramLanguageModel implements LanguageModel {
  
  protected Map<String, Integer> wordFreqMap; // map of words that occur in training with their corresponding frequencies
  protected Map<String, Integer> wordPairFreqMap; // map of word pairs that occur in training with their corresponding frequencies
  int totWordType = 0;	
  
  /** Initialize your data structures in the constructor. */
  public LaplaceBigramLanguageModel(HolbrookCorpus corpus) {
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
  }


  /** Takes a list of strings as argument and returns the log-probability of the 
    * sentence using your language model. Use whatever data you computed in train() here.
    */
  public double score(List<String> sentence) {
    double score = 0.0;
	String prevWord = null;
    for(String word : sentence) { // iterate over words in the sentence
	  if (prevWord != null) {
		  String wordPair = prevWord + "," + word;
		  Integer freqPair = wordPairFreqMap.get(wordPair), freq = wordFreqMap.get(word);
		  int fp = freqPair == null ? 0 : freqPair.intValue(), f = freq == null ? 0 : freq.intValue(); 
		  double probability = Math.log((fp + 1.0) / (f + totWordType));
		  score += probability;
	  }
	  prevWord = word;
    }
    return score;
  }
}
