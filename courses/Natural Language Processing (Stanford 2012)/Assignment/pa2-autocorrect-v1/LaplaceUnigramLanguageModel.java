import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LaplaceUnigramLanguageModel implements LanguageModel {

  protected Map<String, Integer> wordFreqMap; // map of words that occur in training with their corresponding frequencies
  int totWordType = 0;	
  int totWordFreq = 0; // total # of words in the training corpus

  /** Initialize your data structures in the constructor. */
  public LaplaceUnigramLanguageModel(HolbrookCorpus corpus) {
    wordFreqMap = new HashMap<String, Integer>();
    train(corpus);
  }

  /** Takes a corpus and trains your language model. 
    * Compute any counts or other corpus statistics in this function.
    */
  public void train(HolbrookCorpus corpus) {
	for(Sentence sentence : corpus.getData()) { // iterate over sentences
      for(Datum datum : sentence) { // iterate over words
        String word = datum.getWord(); // get the actual word
		Integer freq = wordFreqMap.get(word);
		int f = freq == null ? 1 : freq.intValue() + 1;
        wordFreqMap.put(word, f);
      }
    }
	totWordType = wordFreqMap.size();
	for (Map.Entry<String, Integer> entry : wordFreqMap.entrySet()) { // iterate over map
		//System.out.println(entry.getKey() + ": " + entry.getValue());
		totWordFreq += entry.getValue().intValue();
	}
	//System.out.println(totWordFreq);
  }

  /** Takes a list of strings as argument and returns the log-probability of the 
    * sentence using your language model. Use whatever data you computed in train() here.
    */
  public double score(List<String> sentence) {
    double score = 0.0;
    for(String word : sentence) { // iterate over words in the sentence
      Integer freq = wordFreqMap.get(word);
	  int f = freq == null ? 0 : freq.intValue(); 
  	  double probability = Math.log((f + 1.0) / (totWordFreq + totWordType));
	  score += probability;
    }
    return score;
  }
}
