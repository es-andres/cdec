package comparer;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Objects;
import java.util.TreeSet;

import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.Unirest;

import common.GeneralTuple;
import common.Globals;
import eu.kyotoproject.kaf.KafAugToken;
import eu.kyotoproject.kaf.KafSaxParser;

public class EvSimilarityVector extends SimilarityVector{
	/*
	 * unless otherwise noted, all features apply to both pairs and chains
	 */
	//features
	//1
	private double lemma_overlap;
	//2
	private double synset_overlap;
	//3
	private double synset_distance;
	//4
	private double participant_overlap;
	//5
	private double time_overlap;
	//6
	private double location_overlap;
	//7 pair
	private double discourse_distance;
	//8 pair
	private double sentence_distance;
	//9
	private double word2vec_similarity;
	//10 chain
	private double relative_chain_start_position;
	//11 chain
	private double relative_chain_end_position;
	//12 chain
	private double relative_sentence_start_position;
	//13 chain
	private double relative_sentence_end_position;
	//14 chain TODO
	private double relative_chain_size;
	//15 
	private double temporal_relation;
	//16
	private double sts;
	//class
	private boolean coreferent;
	
	public EvSimilarityVector() {
		
	}
	
	//1
	@Override
	public double getLemmaOverlap() {
		return lemma_overlap;
	}
	//2
	@Override
	public double getSynsetOverlap() {
		return synset_overlap;
	}
	//3
	@Override
	public double getSynsetDistance() {
		return synset_distance;
	}
	//4
	@Override
	public double getParticipantOverlap() {
		return participant_overlap;
	}
	//5
	@Override
	public double getTimeOverlap() {
		return time_overlap;
	}
	//6
	@Override
	public double getLocationOverlap() {
		return location_overlap;
	}
	//7
	@Override
	public double getDiscourseDistance() {
		return discourse_distance;
	}
	//8
	@Override
	public double getSentenceDistance() {
		return sentence_distance;
	}
	//9
	@Override
	public double getWord2VecSimilarity() {
		return word2vec_similarity;
	}
	//10
	@Override
	public double getRelativeChainStartPosition() {
		return relative_chain_start_position;
	}
	//11
	@Override
	public double getRelativeChainEndPosition() {
		return relative_chain_end_position;
	}
	//12
	@Override
	public double getRelativeSentenceStartPosition() {
		return relative_sentence_start_position;
	}
	//13
	@Override
	public double getRelativeSentenceEndPosition() {
		return relative_sentence_end_position;
	}
	//14
	@Override
	public double getRelativeChainSize() {
		return relative_chain_size;
	}
	//15
	@Override
	public double getTemporalRelation() {
		return temporal_relation;
	}
	//16
	@Override
	public double getSts() {
		return sts;
	}
	//class
	@Override
	public boolean getCorefPrediction() {
		return coreferent;
	}

	public String calculateDocTemplateSimVector(GeneralTuple t1, GeneralTuple t2, boolean useTime) {
		
		HashMap<String,LinkedList<LinkedList<KafAugToken>>> template1 = (HashMap<String,LinkedList<LinkedList<KafAugToken>>>)t1.first;
		LinkedList<String> sentences1 = (LinkedList<String>)t1.second;
		HashMap<String,LinkedList<LinkedList<KafAugToken>>> template2 = (HashMap<String,LinkedList<LinkedList<KafAugToken>>>)t2.first;
		LinkedList<String> sentences2 = (LinkedList<String>)t2.second;
		
		StringBuilder vector = new StringBuilder();
		
		//1
		lemma_overlap = docPairLemmaOverlap(template1,template2);
		vector.append(lemma_overlap + ",");
		//2
		synset_overlap = 0;
		vector.append(synset_overlap + ",");
		//3
		synset_distance = 0;
		vector.append(synset_distance + ",");
		//4
		participant_overlap = 0;
		vector.append(participant_overlap + ",");
		//5 TODO
		time_overlap = 0;
		vector.append(time_overlap + ",");
		//6
		location_overlap = 0;
		vector.append(location_overlap + ",");
		//7 pair
		discourse_distance = 0;
		vector.append(discourse_distance + ",");
		//8 pair
		sentence_distance = 0;
		vector.append(sentence_distance + ",");
		//9
		word2vec_similarity = docPairWord2VecSimilarity(template1,template2);
		vector.append(word2vec_similarity + ",");
		//10 chain
		//-------> Always 0
		relative_chain_start_position = 0;
		vector.append(relative_chain_start_position + ",");
		//11 chain
		//-------> Always 0
		relative_chain_end_position = 0;
		vector.append(relative_chain_end_position + ",");
		//12 chain
		//-------> Always 0
		relative_sentence_start_position = 0;
		vector.append(relative_sentence_start_position + ",");
		//13 chain
		//-------> Always 0
		relative_sentence_end_position = 0;
		vector.append(relative_sentence_end_position + ",");
		//14 chain
		//-------> Always 0
		relative_chain_size = 0;
		vector.append(relative_chain_size + ",");
		if(useTime) {
			//15 
			temporal_relation = 0;
			vector.append(temporal_relation +  ",");
		}
		//16
		sts = docPairSts(sentences1, sentences2);
		vector.append(sts + ",");
		
		//class
		coreferent = docPairCoreferent(template1, template2);
		vector.append(coreferent);
		
		
		
		return vector.toString();
	}
	
	//this is always a pair of events within a document
	public String calculateEventPairSimVector(String m_id1, String m_id2, KafSaxParser parser, boolean useTime) {

		LinkedList<KafAugToken> ev1 = new LinkedList<KafAugToken>();
		LinkedList<KafAugToken> ev2 = new LinkedList<KafAugToken>();
		for(String t_id : parser.goldMIdToTokenSetMap.get(m_id1))
			ev1.add(parser.tIdToAugTokenMap.get(t_id));
		for(String t_id : parser.goldMIdToTokenSetMap.get(m_id2))
			ev2.add(parser.tIdToAugTokenMap.get(t_id));
		

		StringBuilder vector = new StringBuilder();
		//1
		lemma_overlap = eventPairLemmaOverlap(ev1,ev2,parser);
		vector.append(lemma_overlap + ",");
		//2: this is too slow, so not using it
//		synset_overlap = eventPairSynsetOverlap(ev1,ev2,parser);
		synset_overlap = 0;
		vector.append(synset_overlap + ",");
		//3: this is too slow, so not using it
//		synset_distance = eventPairSynsetDistance(ev1,ev2,parser);
		synset_distance = 0;
		vector.append(synset_distance + ",");
		//4
		participant_overlap = eventPairParticipantOverlap(ev1,ev2,parser);
		vector.append(participant_overlap + ",");
		//5 TODO
		time_overlap = eventPairTimeOverlap(ev1,ev2,parser);
		vector.append(time_overlap + ",");
		//6
		location_overlap = eventPairLocationOverlap(ev1,ev2,parser);
		vector.append(location_overlap + ",");
		//7 pair
		discourse_distance = eventPairDiscourseDistance(ev1,ev2,parser);
		vector.append(discourse_distance + ",");
		//8 pair
		sentence_distance = eventPairSentenceDistance(ev1,ev2,parser);
		vector.append(sentence_distance + ",");
		//9
		word2vec_similarity = eventPairWord2VecSimilarity(ev1,ev2,parser);
		vector.append(word2vec_similarity + ",");
		//10 chain
		//-------> Always 0
		relative_chain_start_position = 0;
		vector.append(relative_chain_start_position + ",");
		//11 chain
		//-------> Always 0
		relative_chain_end_position = 0;
		vector.append(relative_chain_end_position + ",");
		//12 chain
		//-------> Always 0
		relative_sentence_start_position = 0;
		vector.append(relative_sentence_start_position + ",");
		//13 chain
		//-------> Always 0
		relative_sentence_end_position = 0;
		vector.append(relative_sentence_end_position + ",");
		//14 chain
		//-------> Always 0
		relative_chain_size = 0;
		vector.append(relative_chain_size + ",");
		if(useTime) {
			//15 
			temporal_relation = eventPairTemporalRelation(ev1,ev2,parser);
			vector.append(temporal_relation +  ",");
		}
		//16
		sts = eventPairSts(ev1,ev2,parser);
		vector.append(sts + ",");
		
		//class
		coreferent = eventPairCoreferent(ev1, ev2);
		vector.append(coreferent);

		return vector.toString();

	}
	
	//this is always a pair of event chains across documents
	public String calculateEventChainSimVector(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		
		StringBuilder vector = new StringBuilder();
		//1
		lemma_overlap = eventChainLemmaOverlap(chain1, chain2);
		vector.append(lemma_overlap + ",");
		//2: too slow, not using it
//		synset_overlap = eventChainSynsetOverlap(chain1,chain1);
		synset_overlap = 0;
		vector.append(synset_overlap + ",");
		//3: too slow, not using it
//		synset_distance = eventChainSynsetDistance(chain1,chain1);
		synset_distance = 0;
		vector.append(synset_distance + ",");
		//4
		participant_overlap = eventChainParticipantOverlap(chain1,chain2);
		vector.append(participant_overlap + ",");
		//5 TODO
		time_overlap = eventChainTimeOverlap(chain1,chain2);
		vector.append(time_overlap + ",");
		//6 
		location_overlap = eventChainLocationOverlap(chain1,chain2);
		vector.append(location_overlap + ",");
		//7 pair
		//-------> Always 0
		discourse_distance = 0;
		vector.append(discourse_distance + ",");
		//8 pair
		//-------> Always 0
		sentence_distance = 0;
		vector.append(sentence_distance + ",");
		//9 
		word2vec_similarity = eventChainWord2VecSimilarity(chain1,chain2);
		vector.append(word2vec_similarity + ",");
		//10 chain
		relative_chain_start_position = relativeChainStartPosition(chain1,chain2);
		vector.append(relative_chain_start_position + ",");
		//11 chain
		relative_chain_end_position = relativeChainEndPosition(chain1,chain2);
		vector.append(relative_chain_end_position + ",");
		//12 chain
		relative_sentence_start_position = relativeChainSentenceStartPosition(chain1,chain2);
		vector.append(relative_sentence_start_position + ",");
		//13 chain
		relative_sentence_end_position = relativeChainSentenceEndPosition(chain1,chain2);
		vector.append(relative_sentence_end_position + ",");
		//14 chain TODO
		relative_chain_size = relativeChainSize(chain1,chain2);
		vector.append(relative_chain_size + ",");

		//16 
		sts = eventChainSts(chain1,chain2);
		vector.append(sts + ",");
		
		//class
		coreferent = eventChainCoreferent(chain1, chain2);
		vector.append(coreferent);

		return vector.toString();
	}
	
	
	// ================================== Aux Methods ============================================
	//============================================================================================
	//============================================================================================
	//============================================================================================

	// #################################### Doc Pairs ###########################################
	//#############################################################################################
	//#############################################################################################
	
	private double docPairLemmaOverlap(HashMap<String,LinkedList<LinkedList<KafAugToken>>> template1, HashMap<String,LinkedList<LinkedList<KafAugToken>>> template2) {
		LinkedList<LinkedList<KafAugToken>> lemmas1 = template1.get("action");
		LinkedList<LinkedList<KafAugToken>> lemmas2 = template2.get("action");
		
		int overlap = 0;
		int c1_size = 0;
		int c2_size = 0;
		for(LinkedList<KafAugToken> tokenSet : lemmas1)
			c1_size += tokenSet.size();
		for(LinkedList<KafAugToken> tokenSet : lemmas2)
			c2_size += tokenSet.size();
		
		int maxIntersectSize = Math.min(c1_size, c2_size);
		
		LinkedList<KafAugToken> flatLemmas1 = new LinkedList<KafAugToken>();
		for(LinkedList<KafAugToken> tokenSet1 : lemmas1) {
			for (KafAugToken t1 : tokenSet1)
				flatLemmas1.add(t1);
		}
		LinkedList<KafAugToken> flatLemmas2 = new LinkedList<KafAugToken>();
		for(LinkedList<KafAugToken> tokenSet2 : lemmas2) {
			for (KafAugToken t2 : tokenSet2)
				flatLemmas2.add(t2);
		}
		
		TreeSet<Integer> seen = new TreeSet<Integer>();
		for(KafAugToken t1 : flatLemmas1) {
			for(KafAugToken t2 : flatLemmas2) {
				//only allow each word to corefer with one other word
				if(!seen.contains(Objects.hashCode(t1)) && !seen.contains(Objects.hashCode(t2))) {
					seen.add(Objects.hashCode(t1));
					seen.add(Objects.hashCode(t2));
					if(t1.getLemma().toLowerCase().equals(t2.getLemma().toLowerCase())) {
						overlap++;
					}	
				}
			}
		}
		
		return overlap*1.0/maxIntersectSize;
		
	}
	
	private double docPairWord2VecSimilarity(HashMap<String,LinkedList<LinkedList<KafAugToken>>> template1, HashMap<String,LinkedList<LinkedList<KafAugToken>>> template2) {
		LinkedList<LinkedList<KafAugToken>> lemmas1 = template1.get("action");
		LinkedList<LinkedList<KafAugToken>> lemmas2 = template2.get("action");
		
		double sum = 0;
		
		for(LinkedList<KafAugToken> trigger1 : lemmas1) {
			for(LinkedList<KafAugToken> trigger2 : lemmas2) {
				String phrase1 = "";
				String phrase2 = "";
				for(KafAugToken t : trigger1)
					phrase1 += t.getLemma()+ " ";
				for(KafAugToken t : trigger2)
					phrase2 += t.getLemma()+ " ";
				
				phrase1 = phrase1.substring(0,phrase1.length()-1);
				phrase2 = phrase2.substring(0,phrase2.length()-1);
				
				phrase1 = this.cleanForW2v(phrase1).replaceAll("[^a-zA-Z0-9]", "") + "%20";
				phrase2 = this.cleanForW2v(phrase2).replaceAll("[^a-zA-Z0-9]", "") + "%20";

				phrase1 = phrase1.substring(0,phrase1.length()-3);
				phrase2 = phrase2.substring(0,phrase2.length()-3);
				
				String request = String.format("http://127.0.0.1:9000/?word1=%1$s&word2=%2$s",phrase1,phrase2);

				String response = null;
				double result = 0;
				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				}
				catch(Exception e) {
					e.printStackTrace();
				}
				result = Double.parseDouble(response);
				sum += result;
			}
		}
		
		sum = sum / (lemmas1.size() * lemmas2.size()*1.0);
		return sum;
	}

	
	private double docPairSts(LinkedList<String> sentences1, LinkedList<String> sentences2) {
		
		double sum = 0;
		
		for(String sentence1 : sentences1) {
			for(String sentence2 : sentences2) {
				sentence1 = sentence1.replaceAll("[^a-zA-Z0-9 -]", "").replaceAll("-", " ");
				sentence1 = this.cleanForW2v(sentence1);
				sentence2 = sentence2.replaceAll("[^a-zA-Z0-9 -]", "").replaceAll("-", " ");
				sentence2 = this.cleanForW2v(sentence2);
				double sts = this.getSentenceSts(sentence1, sentence2);
				sum += sts;
			}
		}
		
		sum = sum / (sentences1.size()*sentences2.size()*1.0);
		
		return sum;
	}
	
	private boolean docPairCoreferent(HashMap<String,LinkedList<LinkedList<KafAugToken>>> template1, HashMap<String,LinkedList<LinkedList<KafAugToken>>> template2) {
		LinkedList<LinkedList<KafAugToken>> actions1 = template1.get("action");
		LinkedList<LinkedList<KafAugToken>> actions2 = template2.get("action");
		
		HashSet<String> t1EvIds = new HashSet<String>();		
		for(LinkedList<KafAugToken> triggers : actions1) {
			for(KafAugToken lemma : triggers) {
				t1EvIds.add(lemma.getEvId());
			}
		}
		
		HashSet<String> t2EvIds = new HashSet<String>();
		for(LinkedList<KafAugToken> triggers : actions2) {
			for(KafAugToken lemma : triggers) {
				t2EvIds.add(lemma.getEvId());
			}
		}

		t1EvIds.retainAll(t2EvIds);
		
		return t1EvIds.size() > 0;
	}
	
	
	// #################################### Event Pairs ###########################################
	//#############################################################################################
	//#############################################################################################
	
	/*
	 * NOTE: using treebank pos... maybe something like wordnet pos might work better? -> to test
	 */
	//1
	private double eventPairLemmaOverlap(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {

		int overlap = 0;
		int maxIntersectSize = Math.min(ev1.size(), ev2.size());
		TreeSet<Integer> seen = new TreeSet<Integer>();
		
		LinkedList<GeneralTuple<String,String>> ev1Lemmas = new LinkedList<GeneralTuple<String,String>>();
		for (KafAugToken t : ev1) 
			ev1Lemmas.add(new GeneralTuple<String,String>(t.getLemma().toLowerCase().replaceAll("[^a-zA-Z0-9]", ""),t.getTreebankPos()));
		
		LinkedList<GeneralTuple<String,String>> ev2Lemmas = new LinkedList<GeneralTuple<String,String>>();
		for (KafAugToken t : ev2)
			ev2Lemmas.add(new GeneralTuple<String,String>(t.getLemma().toLowerCase().replaceAll("[^a-zA-Z0-9]", ""),t.getTreebankPos()));
		
		for(GeneralTuple<String,String> l1 : ev1Lemmas) {
			for(GeneralTuple<String,String> l2 : ev2Lemmas) {
//					if(l1.equals(l2)) {//&& l1.second.equals(l2.second)){
//						overlap++;
//					}
				if(!seen.contains(Objects.hashCode(l1)) && !seen.contains(Objects.hashCode(l2))) {
					seen.add(Objects.hashCode(l1));
					seen.add(Objects.hashCode(l2));
					if(l1.first.equals(l2.first) ) {//&& l1.second.equals(l2.second)){
						overlap++;
					}
				}
			}
		}

		return overlap*1.0/maxIntersectSize;
	}
	//2
	private double eventPairSynsetOverlap(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		
		return 0;
	}
	//3
	private double eventPairSynsetDistance(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {

		if(ev1.size() == 0 || ev2.size() == 0)
			System.out.println("PROBLEM");
		String response = auxEventPairSynsetDistance(ev1.getFirst().getWordnetID(),ev2.getFirst().getWordnetID(),"wup");

		if(!response.equals("null"))
			return Double.parseDouble(response);
		else
			return 0;
	}
	//3.1
	private String auxEventPairSynsetDistance(String sense1, String sense2, String measure) {
		if(sense1.length() == 0 || sense2.length() == 0)
			return "null";
		if(!(sense1.split("\\.")[1]).equals((sense2.split("\\.")[1])))
			return "null";
		String[] s1 = sense1.split("\\.");
		String[] s2 = sense2.split("\\.");
		String w1 = "";
		String w2 = "";
		String w1Alt = "";
		String w2Alt = "";
		for(int i  = 0; i < s1.length; i++) {
			if (s1[2].startsWith("0"))
				s1[2] = s1[2].replace("0", "");
	 		if (s2[2].startsWith("0"))
				s2[2] = s2[2].replace("0", "");
	 		if (i < s1.length - 1) {
		 		w1 += s1[i] + "%23";
		 		w2 += s2[i] + "%23";
		 		if(i < s1.length - 2) {
		 		w1Alt += s1[i] + "%23";
		 		w2Alt += s2[i] + "%23";
		 		}
		 		else {
		 			w1Alt += s1[i];
			 		w2Alt += s2[i];
		 		}
	 		}
	 		else {
	 			w1 += s1[i];
	 			w2 += s2[i];
	 		}
		}
		String request = String.format("http://127.0.0.1/cgi-bin/wnsimilarity/similarity.cgi?word1=%1$s&senses1=synset&word2=%2$s&senses2=synset&measure=%3$s&rootnode=yes",w1,w2,measure);
		String output = "null";
		String response = "";

		try {
		//"http://127.0.0.1/cgi-bin/wnsimilarity/similarity.cgi?word1=run%23v%231&senses1=synset&word2=walk%23v%231&senses2=synset&measure=wup&rootnode=yes
			HttpResponse<String> jsonResponse = Unirest.get(request).asString();
			response = jsonResponse.getBody();
			if(!response.contains(" using " + measure)) {
				request = String.format("http://127.0.0.1/cgi-bin/wnsimilarity/similarity.cgi?word1=%1$s&senses1=synset&word2=%2$s&senses2=synset&measure=%3$s&rootnode=yes",w1Alt,w2Alt,measure);
				jsonResponse = Unirest.get(request).asString();
				response = jsonResponse.getBody();
			}
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		if(response.contains("using " + measure)) {
			output = response.split("using " + measure + " is ")[1].split(".</p>")[0];
		}
		return output;
	}
	//4
	private double eventPairParticipantOverlap(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		
		String ev1Sentence = ev1.getFirst().getSentenceNum();
		String ev2Sentence = ev2.getFirst().getSentenceNum();
		
		LinkedList<String> ev1Participants = new LinkedList<String>();
		
		for (KafAugToken t : parser.sentenceNumToAugTokens.get(ev1Sentence)) {
			if(t.getEvType().contains("HUM") )
				ev1Participants.add(t.getTokenString().toLowerCase().replaceAll("[^a-zA-Z0-9]", ""));
		}
		LinkedList<String> ev2Participants = new LinkedList<String>();
		
		for (KafAugToken t : parser.sentenceNumToAugTokens.get(ev2Sentence)) {
			if(t.getEvType().contains("HUM") )
				ev2Participants.add(t.getTokenString().toLowerCase().replaceAll("[^a-zA-Z0-9]", ""));
		}
		
		int overlap = 0;
		int maxIntersectSize = Math.min(ev1Participants.size(), ev2Participants.size());
		TreeSet<Integer> seen = new TreeSet<Integer>();
		if(maxIntersectSize == 0)
			return 0;
		
		for(String p1 : ev1Participants) {
			for(String p2 : ev2Participants) {
				if(!Globals.STOP_WORDS.contains(p1) && !Globals.STOP_WORDS.contains(p2)) {
					if(!seen.contains(Objects.hashCode(p1)) && !seen.contains(Objects.hashCode(p2))) {
						seen.add(Objects.hashCode(p1));
						seen.add(Objects.hashCode(p2));
						if(p1.equals(p2) )//&& l1.second.equals(l2.second))
							overlap++;
					}
//					if(p1.equals(p2) ) {
//						overlap++;
//					}
				}
			}
		}

		return overlap*1.0/maxIntersectSize;
	}
	//5 TODO
	private double eventPairTimeOverlap(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		
		return 0.0;
	}
	//6
	private double eventPairLocationOverlap(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		
		String ev1Sentence = ev1.getFirst().getSentenceNum();
		String ev2Sentence = ev2.getFirst().getSentenceNum();
		
		LinkedList<String> ev1Locs = new LinkedList<String>();
		
		for (KafAugToken t : parser.sentenceNumToAugTokens.get(ev1Sentence)) {
			if(t.getEvType().contains("LOC") )
				ev1Locs.add(t.getLemma().toLowerCase());
		}
		LinkedList<String> ev2Locs = new LinkedList<String>();
		
		for (KafAugToken t : parser.sentenceNumToAugTokens.get(ev2Sentence)) {
			if(t.getEvType().contains("LOC") )
				ev2Locs.add(t.getLemma().toLowerCase());
		}
		
		int overlap = 0;
		int maxIntersectSize = Math.min(ev1Locs.size(), ev2Locs.size());
		TreeSet<Integer> seen = new TreeSet<Integer>();
		if(maxIntersectSize == 0)
			return 0;
		
		for(String loc1 : ev1Locs) {
			for(String loc2 : ev2Locs) {
				if(!seen.contains(Objects.hashCode(loc1)) && !seen.contains(Objects.hashCode(loc2))) {
					seen.add(Objects.hashCode(loc1));
					seen.add(Objects.hashCode(loc2));
					if(!Globals.STOP_WORDS.contains(loc1) && !Globals.STOP_WORDS.contains(loc2)) {
						if(loc1.equals(loc2) )//&& l1.second.equals(l2.second))
							overlap++;
					}
				}
			}
		}

		return overlap*1.0/maxIntersectSize;
	}
	//7 pair
	private double eventPairDiscourseDistance(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		int docLength = parser.kafAugTokensArrayList.size();
		int ev1L = ev1.size();
		int ev2L = ev2.size();
		int sum = 0;
		for(KafAugToken e : ev1)
			sum += Integer.parseInt(e.getTid().replace("t", ""));
		double ev1_mu = sum*1.0/ev1L;
		for(KafAugToken e : ev2)
			sum += Integer.parseInt(e.getTid().replace("t", ""));
		double ev2_mu = sum*1.0/ev2L;
		
		return Math.abs((ev1_mu - ev2_mu)/docLength);
	}
	//8 pair
	private double eventPairSentenceDistance(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		int numSentences = Integer.parseInt(parser.kafAugTokensArrayList.get(parser.kafAugTokensArrayList.size() - 1).getSentenceNum());
		int sent1 = Integer.parseInt(ev1.getFirst().getSentenceNum());
		int sent2 = Integer.parseInt(ev2.getFirst().getSentenceNum());
		
		return Math.abs(sent1-sent2)*1.0/numSentences;
	}
	//9
	private double eventPairWord2VecSimilarity(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		//"http://127.0.0.1/cgi-bin/wnsimilarity/similarity.cgi?word1=%1$s&senses1=synset&word2=%2$s&senses2=synset&measure=%3$s&rootnode=yes"
		String phrase1 = "";
		String phrase2 = "";
		for(KafAugToken e : ev1)
			phrase1 += e.getLemma()+ " ";
		for(KafAugToken e : ev2)
			phrase2 += e.getLemma()+ " ";
		
		phrase1 = phrase1.substring(0,phrase1.length()-1);
		phrase2 = phrase2.substring(0,phrase2.length()-1);
		
		phrase1 = this.cleanForW2v(phrase1).replaceAll("[^a-zA-Z0-9]", "") + "%20";
		phrase2 = this.cleanForW2v(phrase2).replaceAll("[^a-zA-Z0-9]", "") + "%20";

		phrase1 = phrase1.substring(0,phrase1.length()-3);
		phrase2 = phrase2.substring(0,phrase2.length()-3);

		
		String request = String.format("http://127.0.0.1:9000/?word1=%1$s&word2=%2$s",phrase1,phrase2);

		String response = null;
		double result = 0;
		try {
		//"http://127.0.0.1/cgi-bin/wnsimilarity/similarity.cgi?word1=run%23v%231&senses1=synset&word2=walk%23v%231&senses2=synset&measure=wup&rootnode=yes
			HttpResponse<String> serverResponse = Unirest.get(request).asString();
			response = serverResponse.getBody();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		try {
			result = Double.parseDouble(response);
		}
		catch(Exception e) {
			
		}
		return result;
	}
	//10 chain
	//-------> Always 0
	//11 chain
	//-------> Always 0
	//12 chain
	//-------> Always 0
	//13 chain
	//-------> Always 0
	//14 chain
	//-------> Always 0
	//15
	private double eventPairTemporalRelation(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		
		double sum = 0;
		int numRels = 0;
		for (KafAugToken t1 : ev1) {
			for (KafAugToken t2 : ev2) {
				if(!t1.getTemporalRelation().equals("") && !t2.getTemporalRelation().equals("")) {
					
					if(t1.getRelatedTo().replace("t","").equals(t2.getTid().replace("t",""))){
						numRels++;
						String relType = t1.getTemporalRelation();
						
						if(relType.equals("IDENTITY") || relType.equals("SIMULTANEOUS"))
							sum += 1;
//						else if (relType.equals("INCLUDES") || relType.equals("IS_INCLUDED"))
//							sum += 0.5;
					}
					
					if(t2.getRelatedTo().replace("t","").equals(t1.getTid().replace("t",""))) {
						numRels++;
						String relType = t2.getTemporalRelation();
						if(relType.equals("IDENTITY") || relType.equals("SIMULTANEOUS"))
							sum += 1;
//						else if (relType.equals("INCLUDES") || relType.equals("IS_INCLUDED"))
//							sum += 0.5;
					}
					
				}
			}
		}
		if (numRels == 0)
			return 0;
		
		return sum / numRels*1.0;
	}
	
	// 16
	private double eventPairSts(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		
		String sentence1 = parser.getSentenceFromSentenceId(ev1.getFirst().getSentenceNum()).replaceAll("[^a-zA-Z0-9 -]", "").replaceAll("-", " ");
		sentence1 = this.cleanForW2v(sentence1);

		String sentence2 = parser.getSentenceFromSentenceId(ev2.getFirst().getSentenceNum()).replaceAll("[^a-zA-Z0-9 -]", "").replaceAll("-", " ");
		sentence2 = this.cleanForW2v(sentence2);
		
		return this.getSentenceSts(sentence1, sentence2);
	}
	// Aux (16)
	private double getSentenceSts(String sentence1, String sentence2) {
		String request = String.format("http://127.0.0.1:9000/?s1=%1$s&s2=%2$s",sentence1.replaceAll(" ","%20")
				,sentence2.replaceAll(" ", "%20"));
		String response = null;
		
		double result = 0;
		try {
			HttpResponse<String> serverResponse = Unirest.get(request).asString();
			response = serverResponse.getBody();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		
		result = Double.parseDouble(response.replace("[", "").replace("]",""));
		result = (result-1)/4.0;

		return result;
				
	}
	
	// Aux (16)
	private String cleanForW2v(String sentence) {
		String out = "";

		for(String w : sentence.split(" ")) {
			w = w.replaceAll("[\" \\s+]","");
			if(w.length() > 0) {
				if (this.wordInW2v(w).equals("True"))
					out += w + " ";
				// try to get something
				else {
					// remove most punctuation
					w = w.replaceAll("[^a-zA-Z0-9-—]", "");
					String res = this.wordInW2v(w);
					if(res.equals("True"))
						out += w + " ";
					else {
						// lower case
						w = w.toLowerCase();
						res = this.wordInW2v(w);
						if(res.equals("True"))
							out += w + " ";
						else {
							// remove all punctuation
							w = w.replaceAll("[-—]", " ");
							for(String sub_w : w.split(" ")) {
								res = this.wordInW2v(sub_w);
								if(res.equals("True"))
									out += sub_w + " ";
							}
						}
					}
				}
			}
			
		}
		if(out.length() > 0) {
			out = out.substring(0, out.length() - 1);
			return out;
		}
		else { 
			out = "no_es_palabra";
			return out;
		}
	}
	// Aux (16)
	private String wordInW2v(String w) {
		String response = "False";
		if(w.length() <=1)
			return response;
		else {
			String request = String.format("http://127.0.0.1:9000/?w1=%1$s&w2=no_es_palabra",w);
	
			try {
				HttpResponse<String> serverResponse = Unirest.get(request).asString();
				response = serverResponse.getBody();
			}
			catch(Exception e) {
				System.out.println("error --> " + request);
//				e.printStackTrace();
			}
			return response;
		}
	}
	//class
	private boolean eventPairCoreferent(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2) {
		
		return  ev1.getFirst().getEvId().equals(ev2.getFirst().getEvId());
	}
	
	
	// #################################### Event Chains ###########################################
	//##############################################################################################
	//##############################################################################################
	
	//1
	private double eventChainLemmaOverlap(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		
		KafSaxParser f1Parser = chain1.first;
		LinkedList<LinkedList<String>> c1 = chain1.second.second;
		
		KafSaxParser f2Parser = chain2.first;
		LinkedList<LinkedList<String>> c2 = chain2.second.second;

		int overlap = 0;
		int c1_size = 0;
		int c2_size = 0;
		for(LinkedList<String> tokenSet : c1)
			c1_size += tokenSet.size();
		for(LinkedList<String> tokenSet : c2)
			c2_size += tokenSet.size();
		
		int maxIntersectSize = Math.min(c1_size, c2_size);
		
		LinkedList<KafAugToken> tokenList1 = new LinkedList<KafAugToken>();
		for(LinkedList<String> tokenSet1 : c1) {
			for (String t1 : tokenSet1)
				tokenList1.add(f1Parser.tIdToAugTokenMap.get(t1));
		}
		LinkedList<KafAugToken> tokenList2 = new LinkedList<KafAugToken>();
		for(LinkedList<String> tokenSet2 : c2) {
			for (String t2 : tokenSet2)
				tokenList2.add(f2Parser.tIdToAugTokenMap.get(t2));
		}
		
		TreeSet<Integer> seen = new TreeSet<Integer>();
		for(KafAugToken t1 : tokenList1) {
			for(KafAugToken t2 : tokenList2) {
				//only allow each word to corefer with one other word
				if(!seen.contains(Objects.hashCode(t1)) && !seen.contains(Objects.hashCode(t2))) {
					seen.add(Objects.hashCode(t1));
					seen.add(Objects.hashCode(t2));
					if(t1.getLemma().toLowerCase().equals(t2.getLemma().toLowerCase())) {
						overlap++;
					}	
				}
			}
		}
		
		return overlap*1.0/maxIntersectSize;
	}
	//2
	private double eventChainSynsetOverlap(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		
		return 0.0;
	}
	//3
	private double eventChainSynsetDistance(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		
		return 0.0;
	}
	//4 
	private double eventChainParticipantOverlap(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		
		KafSaxParser f1Parser = chain1.first;
		LinkedList<LinkedList<String>> chain1TokenLists = chain1.second.second;

		
		KafSaxParser f2Parser = chain2.first;
		LinkedList<LinkedList<String>> chain2TokenLists = chain2.second.second;
		
		HashSet<String> ev1SentenceSet = new HashSet<String>();
		for(LinkedList<String> chain : chain1TokenLists) {
			for(String t : chain) {
				ev1SentenceSet.add(f1Parser.tIdToAugTokenMap.get(t).getSentenceNum());
			}
		}
		
		LinkedList<String> ev2SentenceSet = new LinkedList<String>();
		for(LinkedList<String> chain : chain2TokenLists) {
			for(String t : chain) {
				ev2SentenceSet.add(f2Parser.tIdToAugTokenMap.get(t).getSentenceNum());
			}
		}
	
		LinkedList<String> ev1Participants = new LinkedList<String>();
		for(String sentence : ev1SentenceSet) {
			for (KafAugToken t : f1Parser.sentenceNumToAugTokens.get(sentence)) {
				if(t.getEvType().contains("HUM") )
					ev1Participants.add(t.getLemma().toLowerCase());
			}
		}
		
		LinkedList<String> ev2Participants = new LinkedList<String>();
		for(String sentence : ev2SentenceSet) {
			for (KafAugToken t : f2Parser.sentenceNumToAugTokens.get(sentence)) {
				if(t.getEvType().contains("HUM") )
					ev2Participants.add(t.getLemma().toLowerCase());
			}
		}

		int overlap = 0;
		int maxIntersectSize = Math.min(ev1Participants.size(),ev2Participants.size());
		if(maxIntersectSize == 0)
			return 0;
		
		TreeSet<Integer> seen = new TreeSet<Integer>();
		for(String p1 : ev1Participants) {
			for(String p2 : ev2Participants) {
				//only allow each word to corefer with one other word
				if(!seen.contains(Objects.hashCode(p1)) && !seen.contains(Objects.hashCode(p2))) {
					seen.add(Objects.hashCode(p1));
					seen.add(Objects.hashCode(p2));
					if(!Globals.STOP_WORDS.contains(p1) && !Globals.STOP_WORDS.contains(p2)) {
						if(p1.equals(p2)) {
							overlap++;
						}
					}
				}
			}
		}
		
		return overlap*1.0/maxIntersectSize;
	}
	//5 TODO
	private double eventChainTimeOverlap(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
	
		return 0.0;
		
	}
	//6 
	private double eventChainLocationOverlap(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
	
		KafSaxParser f1Parser = chain1.first;
		LinkedList<LinkedList<String>> chain1TokenLists = chain1.second.second;

		
		KafSaxParser f2Parser = chain2.first;
		LinkedList<LinkedList<String>> chain2TokenLists = chain2.second.second;
		
		HashSet<String> ev1SentenceSet = new HashSet<String>();
		for(LinkedList<String> chain : chain1TokenLists) {
			for(String t : chain) {
				ev1SentenceSet.add(f1Parser.tIdToAugTokenMap.get(t).getSentenceNum());
			}
		}
		
		LinkedList<String> ev2SentenceSet = new LinkedList<String>();
		for(LinkedList<String> chain : chain2TokenLists) {
			for(String t : chain) {
				ev2SentenceSet.add(f2Parser.tIdToAugTokenMap.get(t).getSentenceNum());
			}
		}
	
		LinkedList<String> ev1Locations = new LinkedList<String>();
		for(String sentence : ev1SentenceSet) {
			for (KafAugToken t : f1Parser.sentenceNumToAugTokens.get(sentence)) {
				if(t.getEvType().contains("LOC") )
					ev1Locations.add(t.getLemma().toLowerCase());
			}
		}
		
		LinkedList<String> ev2Locations = new LinkedList<String>();
		for(String sentence : ev2SentenceSet) {
			for (KafAugToken t : f2Parser.sentenceNumToAugTokens.get(sentence)) {
				if(t.getEvType().contains("LOC") )
					ev2Locations.add(t.getLemma().toLowerCase());
			}
		}

		int overlap = 0;
		int maxIntersectSize = Math.min(ev1Locations.size(), ev2Locations.size());
		if(maxIntersectSize == 0)
			return 0;
		
		TreeSet<Integer> seen = new TreeSet<Integer>();
		for(String loc1 : ev1Locations) {
			for(String loc2 : ev2Locations) {
				//only allow each word to corefer with one other word
				if(!seen.contains(Objects.hashCode(loc1)) && !seen.contains(Objects.hashCode(loc2))) {
					seen.add(Objects.hashCode(loc1));
					seen.add(Objects.hashCode(loc2));
					if(!Globals.STOP_WORDS.contains(loc1) && !Globals.STOP_WORDS.contains(loc2)) {
						if(loc1.equals(loc2)) {
							overlap++;
						}
					}
				}
			}
		}
		
		return overlap*1.0/maxIntersectSize;
	}
	//7 pair
	//-------> Always 0
	//8 pair
	//-------> Always 0
	//9
	private double eventChainWord2VecSimilarity(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		KafSaxParser f1Parser = chain1.first;
		LinkedList<LinkedList<String>> c1 = chain1.second.second;
		
		KafSaxParser f2Parser = chain2.first;
		LinkedList<LinkedList<String>> c2 = chain2.second.second;
		
		String phrase1 = "";
		for(LinkedList<String> tokenSet1 : c1) {
			for (String t : tokenSet1)
				phrase1 += f1Parser.tIdToAugTokenMap.get(t).getLemma() + " ";
		}
		String phrase2 = "";
		for(LinkedList<String> tokenSet2 : c2) {
			for (String t : tokenSet2)
				phrase2 += f2Parser.tIdToAugTokenMap.get(t).getLemma() + " ";
		}
		phrase1 = phrase1.substring(0,phrase1.length()-1);
		phrase2 = phrase2.substring(0,phrase2.length()-1);
		
		phrase1 = this.cleanForW2v(phrase1).replaceAll("[^a-zA-Z0-9]", "") + "%20";
		phrase2 = this.cleanForW2v(phrase2).replaceAll("[^a-zA-Z0-9]", "") + "%20";
		
		phrase1 = phrase1.substring(0,phrase1.length()-3);
		phrase2 = phrase2.substring(0,phrase2.length()-3);

		
		String request = String.format("http://127.0.0.1:9000/?word1=%1$s&word2=%2$s",phrase1,phrase2);

		String response = null;
		double result = 0;
		try {
		//"http://127.0.0.1/cgi-bin/wnsimilarity/similarity.cgi?word1=run%23v%231&senses1=synset&word2=walk%23v%231&senses2=synset&measure=wup&rootnode=yes
			HttpResponse<String> serverResponse = Unirest.get(request).asString();
			response = serverResponse.getBody();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		try {
			result = Double.parseDouble(response);
		}
		catch(Exception e) {
			
		}
		return result;
	}
	//10 chain
	private double relativeChainStartPosition(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		KafSaxParser f1Parser = chain1.first;
		LinkedList<LinkedList<String>> chain1TokenLists = chain1.second.second;
		int doc1Length = f1Parser.kafAugTokensArrayList.size();
		int chain1Start = Integer.parseInt(f1Parser.tIdToAugTokenMap.get(chain1TokenLists.getFirst().getFirst()).getTid().replace("t", ""));
		
		KafSaxParser f2Parser = chain2.first;
		LinkedList<LinkedList<String>> chain2TokenLists = chain2.second.second;
		int doc2Length = f2Parser.kafAugTokensArrayList.size();
		int chain2Start = Integer.parseInt(f2Parser.tIdToAugTokenMap.get(chain2TokenLists.getFirst().getFirst()).getTid().replace("t", ""));
		
		double c1RelativeStart = chain1Start*1.0/doc1Length;
		double c2RelativeStart = chain2Start*1.0/doc2Length;

		return Math.abs(c1RelativeStart - c2RelativeStart);
	}
	//11 chain
	private double relativeChainEndPosition(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		KafSaxParser f1Parser = chain1.first;
		LinkedList<LinkedList<String>> chain1TokenLists = chain1.second.second;
		int doc1Length = f1Parser.kafAugTokensArrayList.size();
		int chain1End = Integer.parseInt(f1Parser.tIdToAugTokenMap.get(chain1TokenLists.getLast().getLast()).getTid().replace("t", ""));
		
		KafSaxParser f2Parser = chain2.first;
		LinkedList<LinkedList<String>> chain2TokenLists = chain2.second.second;
		int doc2Length = f2Parser.kafAugTokensArrayList.size();
		int chain2End = Integer.parseInt(f2Parser.tIdToAugTokenMap.get(chain2TokenLists.getLast().getLast()).getTid().replace("t", ""));
		
		double c1RelativeEnd = chain1End*1.0/doc1Length;
		double c2RelativeEnd = chain2End*1.0/doc2Length;
		
		return Math.abs(c1RelativeEnd - c2RelativeEnd);
	}
	//12 chain
	private double relativeChainSentenceStartPosition(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		KafSaxParser f1Parser = chain1.first;
		LinkedList<LinkedList<String>> chain1TokenLists = chain1.second.second;
		int doc1NumSentences = Integer.parseInt(f1Parser.kafAugTokensArrayList.get(f1Parser.kafAugTokensArrayList.size()-1).getSentenceNum());
		int chain1SentenceStart = Integer.parseInt(f1Parser.tIdToAugTokenMap.get(chain1TokenLists.getFirst().getFirst()).getSentenceNum());
		
		KafSaxParser f2Parser = chain2.first;
		LinkedList<LinkedList<String>> chain2TokenLists = chain2.second.second;
		int doc2NumSentences = Integer.parseInt(f2Parser.kafAugTokensArrayList.get(f2Parser.kafAugTokensArrayList.size()-1).getSentenceNum());
		int chain2SentenceStart = Integer.parseInt(f2Parser.tIdToAugTokenMap.get(chain2TokenLists.getFirst().getFirst()).getSentenceNum());
		
		double c1RelativeStart = chain1SentenceStart*1.0/doc1NumSentences;
		double c2RelativeStart = chain2SentenceStart*1.0/doc2NumSentences;
		
		return Math.abs(c1RelativeStart - c2RelativeStart);
	}
	//13 chain
	private double relativeChainSentenceEndPosition(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		KafSaxParser f1Parser = chain1.first;
		LinkedList<LinkedList<String>> chain1TokenLists = chain1.second.second;
		int doc1NumSentences = Integer.parseInt(f1Parser.kafAugTokensArrayList.get(f1Parser.kafAugTokensArrayList.size()-1).getSentenceNum());
		int chain1SentenceEnd = Integer.parseInt(f1Parser.tIdToAugTokenMap.get(chain1TokenLists.getLast().getLast()).getSentenceNum());
		
		KafSaxParser f2Parser = chain2.first;
		LinkedList<LinkedList<String>> chain2TokenLists = chain2.second.second;
		int doc2NumSentences = Integer.parseInt(f2Parser.kafAugTokensArrayList.get(f2Parser.kafAugTokensArrayList.size()-1).getSentenceNum());
		int chain2SentenceEnd = Integer.parseInt(f2Parser.tIdToAugTokenMap.get(chain2TokenLists.getLast().getLast()).getSentenceNum());
		
		double c1RelativeEnd = chain1SentenceEnd*1.0/doc1NumSentences;
		double c2RelativeEnd = chain2SentenceEnd*1.0/doc2NumSentences;
		
		return Math.abs(c1RelativeEnd - c2RelativeEnd);
	}
	//14 chain TODO
	private double relativeChainSize(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {

		return 0;
	}
	//15
	private double eventChainTemporalRelation(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		KafSaxParser f1Parser = chain1.first;
		LinkedList<LinkedList<String>> c1 = chain1.second.second;
		
		KafSaxParser f2Parser = chain2.first;
		LinkedList<LinkedList<String>> c2 = chain2.second.second;

		LinkedList<KafAugToken> tokenList1 = new LinkedList<KafAugToken>();
		for(LinkedList<String> tokenSet1 : c1) {
			for (String t1 : tokenSet1)
				tokenList1.add(f1Parser.tIdToAugTokenMap.get(t1));
		}
		LinkedList<KafAugToken> tokenList2 = new LinkedList<KafAugToken>();
		for(LinkedList<String> tokenSet2 : c2) {
			for (String t2 : tokenSet2)
				tokenList2.add(f2Parser.tIdToAugTokenMap.get(t2));
		}
		
		double sum = 0;
		int numRels = 0;
		
		for(KafAugToken t1 : tokenList1) {
			for(KafAugToken t2 : tokenList2) {
				if(!t1.getTemporalRelation().equals("") && !t2.getTemporalRelation().equals("")) {
					if(t1.getRelatedTo().replace("t","").equals(t2.getTid().replace("t",""))
							&& t1.getDocId().equals(f2Parser.file.getName().replace("_aug.en.naf", ""))){
						numRels++;
						String relType = t1.getTemporalRelation();
						if(relType.equals("IDENTITY") || relType.equals("SIMULTANEOUS"))
							sum += 1;
//						else if (relType.equals("INCLUDES") || relType.equals("IS_INCLUDED"))
//							sum += 0.5;
					}
					if(t2.getRelatedTo().replace("t","").equals(t1.getTid().replace("t",""))
							&& t2.getDocId().equals(f1Parser.file.getName().replace("_aug.en.naf", ""))) {
						numRels++;
						String relType = t2.getTemporalRelation();
						if(relType.equals("IDENTITY") || relType.equals("SIMULTANEOUS"))
							sum += 1;
//						else if (relType.equals("INCLUDES") || relType.equals("IS_INCLUDED"))
//							sum += 0.5;
					}
				}
			}
		}
		

		if (numRels == 0)
			return 0;
		
		return sum / numRels*1.0;
	}
	
	//16 
	public double eventChainSts(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		KafSaxParser f1Parser = chain1.first;
		LinkedList<LinkedList<String>> c1 = chain1.second.second;
		
		KafSaxParser f2Parser = chain2.first;
		LinkedList<LinkedList<String>> c2 = chain2.second.second;
		
		
		LinkedList<KafAugToken> tokenList1 = new LinkedList<KafAugToken>();
		for(LinkedList<String> tokenSet1 : c1) {
			for (String t1 : tokenSet1)
				tokenList1.add(f1Parser.tIdToAugTokenMap.get(t1));
		}
		LinkedList<KafAugToken> tokenList2 = new LinkedList<KafAugToken>();
		for(LinkedList<String> tokenSet2 : c2) {
			for (String t2 : tokenSet2)
				tokenList2.add(f2Parser.tIdToAugTokenMap.get(t2));
		}
		
		HashSet<String> c1SentenceNums = new HashSet<String>();
		HashSet<String> c2SentenceNums = new HashSet<String>();
		
		for(KafAugToken t1 : tokenList1)
			c1SentenceNums.add(t1.getSentenceNum());
		for (KafAugToken t2 : tokenList2)
			c2SentenceNums.add(t2.getSentenceNum());
		
		Double sts = 0.0;
		for(String s1 : c1SentenceNums) {
			for(String s2 : c2SentenceNums) {
				String sentence1 = f1Parser.getSentenceFromSentenceId(s1).replaceAll("[^a-zA-Z0-9 -]", "").replaceAll("-", " ");
				sentence1 = this.cleanForW2v(sentence1);

				String sentence2 = f2Parser.getSentenceFromSentenceId(s2).replaceAll("[^a-zA-Z0-9 -]", "").replaceAll("-", " ");
				sentence2 = this.cleanForW2v(sentence2);
				
				Double sim = this.getSentenceSts(sentence1, sentence2);
				sts += sim;
			}
		}
		
		sts = sts / (c1SentenceNums.size()*c2SentenceNums.size()*1.0);
		
		return sts;
	}
	
	//class
	private boolean eventChainCoreferent(GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain1, GeneralTuple<KafSaxParser,GeneralTuple<String,LinkedList<LinkedList<String>>>> chain2) {
		
		return chain1.second.first.equals(chain2.second.first);
	}
}
