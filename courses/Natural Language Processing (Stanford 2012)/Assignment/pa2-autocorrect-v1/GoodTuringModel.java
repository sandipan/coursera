import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GoodTuringModel implements LanguageModel {

  protected Map<String, Integer> wordFreqMap; // map of words that occur in training with their corresponding frequencies
  int N = 0;  // total # of words in the training corpus
  int [] f_N; // frequency of frequencies

  /** Initialize your data structures in the constructor. */
  public GoodTuringModel(HolbrookCorpus corpus) {
    wordFreqMap = new HashMap<String, Integer>();
	train(corpus);
  }

  /** Takes a corpus and trains your language model (Good Turing). 
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
	int maxFreq = 0;
	for (Map.Entry<String, Integer> entry : wordFreqMap.entrySet()) { // iterate over map
		int freq = entry.getValue().intValue();
		N += freq;
		if (freq > maxFreq)
		{
			maxFreq = freq;
		}
	}
	f_N = new int[maxFreq + 1];
	for (int i = 0; i < f_N.length; ++i) {
		f_N[i] = 0;
	}
	for (Map.Entry<String, Integer> entry : wordFreqMap.entrySet()) { // iterate over map
		int freq = entry.getValue().intValue();
		++f_N[freq];
	}
	/*for (int i = 0; i <= maxFreq; ++i) {
		System.out.print(i + ": " + f_N[i] + " ");
	}*/
  }

  /** Takes a list of strings as argument and returns the log-probability of the 
    * sentence using your language model. Use whatever data you computed in train() here.
    */
  public double score(List<String> sentence) {
     double score = 0.0;
    for(String word : sentence) { // iterate over words in the sentence
      Integer freq = wordFreqMap.get(word);
	  int c = freq == null ? 0 : freq.intValue();
	  double c_star = c != 0 ? (c + 1 < f_N.length ? ((c + 1.0) * f_N[c + 1]) / f_N[c] : c) : 0;
  	  double probability = Math.log(c == 0 ? (f_N[1] * 1.0) / N : c_star / N);
	  score += probability;
    }
    return score;
  }
  
}
