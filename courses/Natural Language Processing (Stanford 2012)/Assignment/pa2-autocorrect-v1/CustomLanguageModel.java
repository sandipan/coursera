import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

public class CustomLanguageModel implements LanguageModel {

  protected Map<String, Integer> wordFreqMap; // map of words that occur in training with their corresponding frequencies
  protected Map<String, Integer> wordPairFreqMap; // map of word pairs that occur in training with their corresponding frequencies
  protected Map<String, Integer> wordTrippleFreqMap; // map of word tripples that occur in training with their corresponding frequencies
  int totWordType = 0;	
  int totWordFreq = 0; // total # of words in the training corpus
  int totWordPairType = 0;
  
  /** Initialize your data structures in the constructor. */
  public CustomLanguageModel(HolbrookCorpus corpus) {
    wordFreqMap = new HashMap<String, Integer>();
    wordPairFreqMap = new HashMap<String, Integer>();
    wordTrippleFreqMap = new HashMap<String, Integer>();
	train(corpus);
  }

  /** Takes a corpus and trains your language model (Stupid Backoff with Trigram). 
    * Compute any counts or other corpus statistics in this function.
    */
  public void train(HolbrookCorpus corpus) { 
	for(Sentence sentence : corpus.getData()) { // iterate over sentences
      List<String> words = new ArrayList<String>();
	  String word = null, wordPair = null, wordTripple = null;
	  Integer freq = null;
	  int f = 0;
	  int len = 0;
	  for(Datum datum : sentence) { // iterate over words
        word = datum.getWord(); // get the actual word
		words.add(word); 
		freq = wordFreqMap.get(word);
		f = freq == null ? 1 : freq.intValue() + 1;
        wordFreqMap.put(word, f);
		len = words.size();
		if (len > 3) {
			words.remove(0);
		}
		len = words.size();
		if (len >= 2) {
			wordPair = len == 2 ? (words.get(0) + "," + words.get(1)) : (words.get(1) + "," + words.get(2));
			freq = wordPairFreqMap.get(wordPair);
			f = freq == null ? 1 : freq.intValue() + 1;
			wordPairFreqMap.put(wordPair, f);
		}
		if (len == 3) {
			wordTripple = words.get(0) + "," + words.get(1) + "," + words.get(2);
			freq = wordTrippleFreqMap.get(wordTripple);
			f = freq == null ? 1 : freq.intValue() + 1;
			wordTrippleFreqMap.put(wordTripple, f);
		}
      }
    }
	totWordType = wordFreqMap.size();
	totWordPairType = wordPairFreqMap.size();
	for (Map.Entry<String, Integer> entry : wordFreqMap.entrySet()) { // iterate over map
		totWordFreq += entry.getValue().intValue();
	}
  }

  /** Takes a list of strings as argument and returns the log-probability of the 
    * sentence using your language model. Use whatever data you computed in train() here.
    */
  public double score(List<String> sentence) {
    double score = 0.0;
	List<String> words = new ArrayList<String>();
	String wordPair = null, wordTripple = null;
	Integer freq = null;
	int len = 0;
	int ftripple, fpair, fcur, fprev;
	ftripple = fpair = fcur = fprev = 0;
	double probability = 0.0;
	for(String word : sentence) { // iterate over words in the sentence
	  words.add(word); 
	  len = words.size();
	  if (len > 3) {
		words.remove(0);
	  }
	  len = words.size();
	  if (len == 3) {
		  wordTripple = words.get(0) + "," + words.get(1) + "," + words.get(2);
		  freq = wordTrippleFreqMap.get(wordTripple);
		  ftripple = freq == null ? 0 : freq.intValue();
		  wordPair = words.get(1) + "," + words.get(2);
		  freq = wordPairFreqMap.get(wordPair);
		  fpair = freq == null ? 0 : freq.intValue();
		  freq = wordFreqMap.get(words.get(1));
		  fprev = freq == null ? 0 : freq.intValue();
	  	  freq = wordFreqMap.get(words.get(2));
		  fcur = freq == null ? 0 : freq.intValue();
	  	  probability = Math.log(ftripple > 0 ? (ftripple * 1.0) / fpair :
								 fpair > 0 ? (fpair * 1.0) / fprev : 
								 (fcur + 1.0) / (totWordFreq + totWordType));
	  }
	  else if (len == 2) {
	  	  wordPair = words.get(0) + "," + words.get(1);
		  freq = wordPairFreqMap.get(wordPair);
		  fpair = freq == null ? 0 : freq.intValue();
		  freq = wordFreqMap.get(words.get(0));
		  fprev = freq == null ? 0 : freq.intValue();
	  	  freq = wordFreqMap.get(words.get(1));
		  fcur = freq == null ? 0 : freq.intValue();
	  	  probability = Math.log(fpair > 0 ? (fpair * 1.0) / fprev : 
								 (fcur + 1.0) / (totWordFreq + totWordType));
	  }
	  else {
	  	  freq = wordFreqMap.get(words.get(0));
		  fcur = freq == null ? 0 : freq.intValue();
	      probability = Math.log((fcur + 1.0) / (totWordFreq + totWordType));
	  }	  
	  score += probability;
    }
    return score;
  }
 
}
