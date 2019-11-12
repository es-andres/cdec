package comparer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.Unirest;

import common.GeneralTuple;
import common.Globals;
import edu.emory.clir.clearnlp.dependency.DEPNode;
import eu.kyotoproject.kaf.KafAugToken;
import eu.kyotoproject.kaf.KafSaxParser;
import vu.wntools.wnsimilarity.WordnetSimilarityApi;
import vu.wntools.wnsimilarity.measures.SimilarityPair;
import vu.wntools.wordnet.WordnetLmfSaxParser;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.not;
import static org.junit.Assert.assertThat;

public class EvPairSimVector {

	private final HashSet<String> DISCRETE_ROLES = new HashSet<String>(
			Arrays.asList("trigger", "A0", "A1", "TMP", "LOC"));

	// 1
	private double trigger_sim_vec = 0;
	// 2
	private double A0_sim_vec = 0;
	// 3
	private double A1_sim_vec = 0;
	// 4
	private double TMP_sim_vec = 0;
	// 5
	private double LOC_sim_vec = 0;
	// 6
	private double aux_arg_sim_vec = 0;
	// 7
	private double discourse_distance = 0;
	// 8
	private double sentence_distance = 0;
	// 9
	private double context_sts = 0;
	// 10
	private double event_sim_strict = 0;
	// 11
	private double event_sts = 0;
	// 12
	private double trigger_sts = 0;
	// 13
	private double trigger_sim_strict = 0;
	// 14
	private double A0_sim_strict = 0;
	// 15
	private double A1_sim_strict = 0;
	// 16
	private double TMP_sim_strict = 0;
	// 17
	private double LOC_sim_strict = 0;
	// 18
	private double aux_arg_sim_strict = 0;
	// 19
	private double lstm_trigger = 0;
	// 20
	private double lstm_context = 0;
	// 21
	private boolean same_sentence;
	// 22
	private boolean same_doc;
	// 23
	private double trigger_wn_sim = 0;

	// class
	private boolean coreferent;
	
	private WordnetLmfSaxParser wn_parser;
	

	public EvPairSimVector() {
		wn_parser = new WordnetLmfSaxParser();
		wn_parser.parseFile(Globals.WN_LMF_PATH);
	}

	public String calculateWDEventPairSimVector(String m_id1, String m_id2, KafSaxParser parser1, KafSaxParser parser2,
			HashMap<String, List<DEPNode>> parsedEv1, HashMap<String, List<DEPNode>> parsedEv2) {

		LinkedList<KafAugToken> ev1 = new LinkedList<KafAugToken>();
		LinkedList<KafAugToken> ev2 = new LinkedList<KafAugToken>();
		for (String t_id : parser1.goldMIdToTokenSetMap.get(m_id1))
			ev1.add(parser1.tIdToAugTokenMap.get(t_id));
		for (String t_id : parser2.goldMIdToTokenSetMap.get(m_id2))
			ev2.add(parser2.tIdToAugTokenMap.get(t_id));

		StringBuilder vector = new StringBuilder();

		// 1
		trigger_sim_vec = phraseOverlapVec(parsedEv1.get("trigger"), parsedEv2.get("trigger"));
		vector.append(trigger_sim_vec + ",");
		// 2
		A0_sim_vec = ((parsedEv1.containsKey("A0") && parsedEv2.containsKey("A0"))
				? phraseOverlapVec(parsedEv1.get("A0"), parsedEv2.get("A0"))
				: 0);
		vector.append(A0_sim_vec + ",");
		// 3
		A1_sim_vec = ((parsedEv1.containsKey("A1") && parsedEv2.containsKey("A1"))
				? phraseOverlapVec(parsedEv1.get("A1"), parsedEv2.get("A1"))
				: 0);
		vector.append(A1_sim_vec + ",");
		// 4
		TMP_sim_vec = ((parsedEv1.containsKey("TMP") && parsedEv2.containsKey("TMP"))
				? phraseOverlapVec(parsedEv1.get("TMP"), parsedEv2.get("TMP"))
				: 0);
		vector.append(TMP_sim_vec + ",");
		// 5
		LOC_sim_vec = ((parsedEv1.containsKey("LOC") && parsedEv2.containsKey("LOC"))
				? phraseOverlapVec(parsedEv1.get("LOC"), parsedEv2.get("LOC"))
				: 0);
		vector.append(LOC_sim_vec + ",");
		// 6
		LinkedList<DEPNode> aux1 = getAuxNodes(parsedEv1);
		LinkedList<DEPNode> aux2 = getAuxNodes(parsedEv2);
		aux_arg_sim_vec = ((aux1.size() > 1 && aux2.size() > 1) ? phraseOverlapVec(aux1, aux2) : 0);
		vector.append(aux_arg_sim_vec + ",");
		// 7
		discourse_distance = parser1.file.equals(parser2.file) ? discourseDistance(ev1, ev2, parser1) : relativeDiscourseDistance(ev1,ev2,parser1,parser2);
		vector.append(discourse_distance + ",");
		// 8
		sentence_distance = parser1.file.equals(parser2.file) ? sentenceDistance(ev1, ev2, parser1) : relativeSentenceDistance(ev1,ev2,parser1,parser2);
		vector.append(sentence_distance + ",");
		// 9
		boolean lemma = false;
		context_sts = phraseSts(getEvContext(parser1, m_id1, lemma), getEvContext(parser2, m_id2, lemma));
		vector.append(context_sts + ",");
		// 10
		LinkedList<DEPNode> nodeList1 = new LinkedList<DEPNode>();
		for (String role : parsedEv1.keySet())
			nodeList1.addAll(parsedEv1.get(role));
		LinkedList<DEPNode> nodeList2 = new LinkedList<DEPNode>();
		for (String role : parsedEv2.keySet())
			nodeList2.addAll(parsedEv2.get(role));
		event_sim_strict = phraseOverlapStrict(nodeList1, nodeList2);
		vector.append(event_sim_strict + ",");
		// 11
		event_sts = phraseSts(getEventString(parsedEv1), getEventString(parsedEv2));
		vector.append(event_sts + ",");
		// 12
		trigger_sts = phraseSts(getRoleString(parsedEv1.get("trigger")), getRoleString(parsedEv2.get("trigger")));
		vector.append(trigger_sts + ",");
		// 13
		trigger_sim_strict = phraseOverlapStrict(parsedEv1.get("trigger"), parsedEv2.get("trigger"));
		vector.append(trigger_sim_strict + ",");
		// 14
		A0_sim_strict = ((parsedEv1.containsKey("A0") && parsedEv2.containsKey("A0"))
				? phraseOverlapStrict(parsedEv1.get("A0"), parsedEv2.get("A0"))
				: 0);
		vector.append(A0_sim_strict + ",");
		// 15
		A1_sim_strict = ((parsedEv1.containsKey("A1") && parsedEv2.containsKey("A1"))
				? phraseOverlapStrict(parsedEv1.get("A1"), parsedEv2.get("A1"))
				: 0);
		vector.append(A1_sim_strict + ",");
		// 16
		TMP_sim_strict = ((parsedEv1.containsKey("TMP") && parsedEv2.containsKey("TMP"))
				? phraseOverlapStrict(parsedEv1.get("TMP"), parsedEv2.get("TMP"))
				: 0);
		vector.append(TMP_sim_strict + ",");
		// 17
		LOC_sim_strict = ((parsedEv1.containsKey("LOC") && parsedEv2.containsKey("LOC"))
				? phraseOverlapStrict(parsedEv1.get("LOC"), parsedEv2.get("LOC"))
				: 0);
		vector.append(LOC_sim_strict + ",");
		// 18
		aux_arg_sim_strict = ((aux1.size() > 1 && aux2.size() > 1) ? phraseOverlapStrict(aux1, aux2) : 0);
		vector.append(aux_arg_sim_strict + ",");
		// 19
		lstm_trigger = lstm(getRoleString(parsedEv1.get("trigger")), getRoleString(parsedEv1.get("trigger")),
				"trigger");
		vector.append(lstm_trigger + ",");
		// 20
		lstm_context = lstm(getEvContext(parser1, m_id1, lemma), getEvContext(parser2, m_id2, lemma), "context");
		vector.append(lstm_context + ",");
		// 21
		same_sentence = !parser1.file.equals(parser2.file) ? false : sameSentence(ev1, ev2, parser1);
		vector.append(same_sentence + ",");
		// 22
		same_doc = parser1.file.getName().equals(parser2.file.getName());
		vector.append(same_doc + ",");
		// 23
		trigger_wn_sim = wnSim(parsedEv1.get("trigger"), parsedEv2.get("trigger"));
//		trigger_wn_sim = 0;
		vector.append(trigger_wn_sim + ",");

		// class
		coreferent = coreferent(ev1, ev2);
		vector.append(coreferent);

		return vector.toString();

	}

	/*
	 * AUX methods
	 */
	private double wnSim(List<DEPNode> p1, List<DEPNode> p2) {
		String idiom1 = "";
		String idiom2 = "";
		List<String> phrase1;
		List<String> phrase2;
		Pattern p = Pattern.compile("[^a-zA-Z0-9]");
		
		for(DEPNode n : p1)
			idiom1 += ((!p.matcher(n.getLemma()).find()) ? n.getLemma().toLowerCase()
					: n.getWordForm().toLowerCase().replaceAll("[^a-zA-Z0-9]", "")) + "_";
		
		idiom1 = idiom1.substring(0, idiom1.length() - 1);
		
		if(wn_parser.wordnetData.entryToSynsets.containsKey(idiom1)) {
			phrase1 = Stream.of(idiom1).collect(Collectors.toCollection(LinkedList<String>::new));
		}
		else
			phrase1 = Arrays.asList(idiom1.split("_"));
		
		for(DEPNode n : p2)
			idiom2 += ((!p.matcher(n.getLemma()).find()) ? n.getLemma().toLowerCase()
					: n.getWordForm().toLowerCase().replaceAll("[^a-zA-Z0-9]", "")) + "_";
		idiom2 = idiom2.substring(0,idiom2.length() - 1);
		
		if(wn_parser.wordnetData.entryToSynsets.containsKey(idiom2)) {
			phrase2 = Stream.of(idiom2).collect(Collectors.toCollection(LinkedList<String>::new));
		}
		else
			phrase2 = Arrays.asList(idiom2.split("_"));
		double sum = 0;
		double count = 0;
		HashSet<HashSet<Integer>> seen = new HashSet<HashSet<Integer>>();
		for(String w1 : phrase1) {
			for(String w2 : phrase2) {
				HashSet<Integer> thisPair = Stream.of(w1.hashCode(),w2.hashCode()).collect(Collectors.toCollection(HashSet<Integer>::new));
				if(seen.contains(thisPair))
					continue;
				seen.add(thisPair);
				count++;
				ArrayList<SimilarityPair> pairs = WordnetSimilarityApi.wordLeacockChodorowSimilarity(wn_parser.wordnetData, w1, w2);
				if(pairs.size() > 0) {
					SimilarityPair pair = WordnetSimilarityApi.getTopScoringSimilarityPair(pairs);
					sum += pair.getScore();
				}
			}
		}
		double avg_sim = sum / count*1.0;
//		assertThat(avg_sim >= 0 && avg_sim <= 1,is(true));
		
		return avg_sim;
		
	}
	private double phraseOverlapStrict(List<DEPNode> p1, List<DEPNode> p2) {


		HashSet<HashSet<Integer>> seen = new HashSet<HashSet<Integer>>();
		Pattern p = Pattern.compile("[^a-zA-Z0-9]");
		LinkedList<GeneralTuple<String, String>> p1Lemmas = new LinkedList<GeneralTuple<String, String>>();
		for (DEPNode t : p1)
			p1Lemmas.add(
					new GeneralTuple<String, String>(((!p.matcher(t.getLemma()).find()) ? t.getLemma().toLowerCase()
							: t.getWordForm().toLowerCase().replaceAll("[^a-zA-Z0-9]", "")), t.getPOSTag()));

		LinkedList<GeneralTuple<String, String>> p2Lemmas = new LinkedList<GeneralTuple<String, String>>();
		for (DEPNode t : p2)
			p2Lemmas.add(
					new GeneralTuple<String, String>(((!p.matcher(t.getLemma()).find()) ? t.getLemma().toLowerCase()
							: t.getWordForm().toLowerCase().replaceAll("[^a-zA-Z0-9]", "")), t.getPOSTag()));
		
		int overlap = 0;
		for (GeneralTuple<String, String> l1 : p1Lemmas) {
			for (GeneralTuple<String, String> l2 : p2Lemmas) {
				HashSet<Integer> thisPair = Stream.of(l1.first.hashCode(),l2.first.hashCode()).collect(Collectors.toCollection(HashSet<Integer>::new));
				if(seen.contains(thisPair))
					continue;
				seen.add(thisPair);
				if (l1.first.equals(l2.first))
					overlap++;
			}
		}
		
		double score = overlap * 1.0 / seen.size();
		
		assertThat(score >=0 && score <= 1, is(true));
		
		return score;
	}

	public static String getHTML(String urlToRead) {
		StringBuilder result = new StringBuilder();
		URL url = null;

		try {
			url = new URL(urlToRead);
		} catch (MalformedURLException e) {
			e.printStackTrace();
		}

		HttpURLConnection conn = null;
		try {
			conn = (HttpURLConnection) url.openConnection();
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
			conn.setRequestMethod("GET");
		} catch (ProtocolException e) {
			e.printStackTrace();
		}
		BufferedReader rd = null;
		try {
			rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
		} catch (IOException e) {
			e.printStackTrace();
		}
		String line;
		try {
			while ((line = rd.readLine()) != null) {
				result.append(line);
			}
			rd.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return result.toString();
	}

	private boolean entityCoref(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		String s_id1 = ev1.getFirst().getSentenceNum();
		String s_id2 = ev2.getFirst().getSentenceNum();

		if (s_id1.equals(s_id2))
			return true;

		List<KafAugToken> s1 = parser.sentenceNumToAugTokens.get(s_id1);
		List<KafAugToken> s2 = parser.sentenceNumToAugTokens.get(s_id2);

		HashSet<String> s1Entities = new HashSet<String>();
		HashSet<String> s2Entities = new HashSet<String>();

		for (KafAugToken tok : s1) {
			if (tok.getEvType().contains("PART"))
				s1Entities.add(tok.getEvId());
		}
		for (KafAugToken tok : s2) {
			if (tok.getEvType().contains("PART"))
				s2Entities.add(tok.getEvId());
		}

		s1Entities.retainAll(s2Entities);

		int count = s1Entities.size();

		return count > 0;

	}

	private double phraseOverlapVec(List<DEPNode> p1, List<DEPNode> p2) {

		String phrase1 = "";
		String phrase2 = "";
		Pattern p = Pattern.compile("[^a-zA-Z0-9]");
		for (DEPNode e : p1)
			// phrase1 += ((!p.matcher(e.getLemma()).find()) ? e.getLemma() :
			// e.getWordForm()) + " ";
			phrase1 += e.getWordForm() + " ";
		for (DEPNode e : p2)
			// phrase2 += ((!p.matcher(e.getLemma()).find()) ? e.getLemma() :
			// e.getWordForm()) + " ";
			phrase2 += e.getWordForm() + " ";

		// phrase1 = phrase1.substring(0,phrase1.length()-1).replaceAll(" ",
		// "%20").replaceAll("\\s+", "");
		// phrase2 = phrase2.substring(0,phrase2.length()-1).replaceAll(" ",
		// "%20").replaceAll("\\s+", "");
		try {
			phrase1 = URLEncoder.encode(phrase1.substring(0, phrase1.length() - 1), "utf-8");
			phrase2 = URLEncoder.encode(phrase2.substring(0, phrase2.length() - 1), "utf-8");
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}

		String request = String.format("http://127.0.0.1:9000/?word1=%1$s&word2=%2$s", phrase1, phrase2);
		// request = request.replaceAll("\\s+","");

		String response = getHTML(request);
		double result = 0;
		try {
			result = Double.parseDouble(response);
		} catch (Exception e) {

		}

		return result;
	}

	private double phraseSts(String sentence1, String sentence2) {
		// sentence1 = sentence1.replaceAll(" ", "%20").replaceAll(" ",
		// "%20").replaceAll("\\s+", "");
		// sentence2 = sentence2.replaceAll(" ", "%20").replaceAll(" ",
		// "%20").replaceAll("\\s+", "");
		try {
			sentence1 = URLEncoder.encode(sentence1, "utf-8");
			sentence2 = URLEncoder.encode(sentence2, "utf-8");
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
		String request = String.format("http://127.0.0.1:9000/?s1=%1$s&s2=%2$s", sentence1, sentence2);

		String response = getHTML(request);

		double result = 0;
		result = Double.parseDouble(response);

		return result;
	}

	private double lstm(String phrase1, String phrase2, String type) {
		String request = null;
		// phrase1 = phrase1.replaceAll(" ","%20").replaceAll("\\s+", "");
		// phrase2 = phrase2.replaceAll(" ","%20").replaceAll("\\s+", "");
		try {
			phrase1 = URLEncoder.encode(phrase1, "utf-8");
			phrase2 = URLEncoder.encode(phrase2, "utf-8");
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
		if (type.equals("trigger"))
			request = String.format("http://127.0.0.1:9000/?word1=%1$s&word2=%2$s", phrase1, phrase2);
		else if (type.equals("context"))
			request = String.format("http://127.0.0.1:9000/?s1=%1$s&s2=%2$s", phrase1, phrase2);

		String response = getHTML(request);

		double result = Double.parseDouble(response);

		return result;

	}

	public String getEvContext(KafSaxParser parser, String m_id, boolean lemma) {

		String t_id = parser.goldMIdToTokenSetMap.get(m_id).getFirst();
		String s_id = parser.tIdToAugTokenMap.get(t_id).getSentenceNum();
		List<KafAugToken> sentenceTokenList = parser.sentenceNumToAugTokens.get(s_id);

		String context = "";
		for (KafAugToken t : sentenceTokenList) {
			if (!t.getEvType().contains("ACT") || t.getMId().equals(m_id)) {
				if (lemma)
					context += t.getLemma() + " ";
				else
					context += t.getTokenString() + " ";
			}
		}
		context = context.substring(0, context.length() - 1);

		return context;
	}

	private String getEventString(HashMap<String, List<DEPNode>> parsedEv) {
		String expansion = "";

		TreeMap<Integer, String> eventPhrase = new TreeMap<Integer, String>();
		for (String role : parsedEv.keySet()) {
			for (DEPNode n : parsedEv.get(role))
				eventPhrase.put(n.getID(), n.getWordForm());
		}
		for (String w : eventPhrase.values())
			expansion += w + " ";

		return expansion.substring(0, expansion.length() - 1);
	}

	private String getRoleString(List<DEPNode> ev) {
		String expansion = "";

		TreeMap<Integer, String> eventPhrase = new TreeMap<Integer, String>();
		for (DEPNode n : ev)
			eventPhrase.put(n.getID(), n.getWordForm());

		for (String w : eventPhrase.values())
			expansion += w + " ";

		return expansion.substring(0, expansion.length() - 1);

	}

	private LinkedList<DEPNode> getAuxNodes(HashMap<String, List<DEPNode>> parsedEv) {
		LinkedList<DEPNode> aux = new LinkedList<DEPNode>();
		for (String role : parsedEv.keySet()) {
			if (!DISCRETE_ROLES.contains(role)) {
				for (DEPNode n : parsedEv.get(role))
					aux.add(n);
			}
		}
		return aux;
	}

	private String getSentence(LinkedList<KafAugToken> toks, KafSaxParser parser) {
		String sentence = parser.getSentenceFromSentenceId(toks.getFirst().getSentenceNum())
				.replaceAll("[^a-zA-Z0-9 -]", "").replaceAll("-", " ");

		return sentence;
	}
	private double relativeDiscourseDistance(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser1, KafSaxParser parser2) {
		
		int docLength1 = parser1.kafAugTokensArrayList.size();
		int docLength2 = parser2.kafAugTokensArrayList.size();
		
		int ev1L = ev1.size();
		int ev2L = ev2.size();
		int sum = 0;
		for (KafAugToken e : ev1)
			sum += Integer.parseInt(e.getTid().replace("t", ""));
		// get central token for event 1
		double ev1_mu = sum * 1.0 / ev1L;
		sum = 0;
		for (KafAugToken e : ev2)
			sum += Integer.parseInt(e.getTid().replace("t", ""));
		// get central token for event 2
		double ev2_mu = sum * 1.0 / ev2L;
		
		double dist = Math.abs((ev1_mu/docLength1*1.0 - ev2_mu/docLength2*1.0));
		
		String error_msg = "ev1_mu: " + ev1_mu + ", ev2_mu: " + ev2_mu + ", docLength1: " + docLength1 + ", docLength2: " + docLength2 + ", dist: " + dist;
		
		assertThat(error_msg, dist >= 0 && dist <= 1, is(true));
		
		return dist;
		
	}
	private double discourseDistance(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		int docLength = parser.kafAugTokensArrayList.size();
		int ev1L = ev1.size();
		int ev2L = ev2.size();
		int sum = 0;
		for (KafAugToken e : ev1)
			sum += Integer.parseInt(e.getTid().replace("t", ""));
		// get central token for event 1
		double ev1_mu = sum * 1.0 / ev1L;
		sum = 0;
		for (KafAugToken e : ev2)
			sum += Integer.parseInt(e.getTid().replace("t", ""));
		// get central token for event 2
		double ev2_mu = sum * 1.0 / ev2L;
		
		double dist = Math.abs((ev1_mu - ev2_mu) / docLength*1.0);
		
		String error_msg = "ev1_mu: " + ev1_mu + ", ev2_mu: " + ev2_mu + ", docLength: " + docLength + ", dist: " + dist;
		
		assertThat(error_msg, dist >= 0 && dist <= 1, is(true));
		
		return dist;
	}
	
	private double relativeSentenceDistance(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser1, KafSaxParser parser2) {
		int numSentences1 = Integer
				.parseInt(parser1.kafAugTokensArrayList.get(parser1.kafAugTokensArrayList.size() - 1).getSentenceNum());
		if(Integer.parseInt(parser1.kafAugTokensArrayList.get(0).getSentenceNum()) == 0)
			numSentences1++;
		
		int numSentences2 = Integer
				.parseInt(parser2.kafAugTokensArrayList.get(parser2.kafAugTokensArrayList.size() - 1).getSentenceNum());
		if(Integer.parseInt(parser2.kafAugTokensArrayList.get(0).getSentenceNum()) == 0)
			numSentences2++;
		
		int sent1 = Integer.parseInt(ev1.getFirst().getSentenceNum());
		int sent2 = Integer.parseInt(ev2.getFirst().getSentenceNum());
		
		double dist = Math.abs(sent1/numSentences1*1.0 - sent2/numSentences2*1.0);
		String error_msg = "sent1: " + sent1 + ", sent2: " + sent2 + ", numSent1: " + numSentences1 + ", numSent2: " + numSentences2 +  ", dist: " + dist;
		
		assertThat(error_msg, dist >= 0 && dist <= 1, is(true));
		
		return dist;
		
	}

	private double sentenceDistance(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		int numSentences = Integer
				.parseInt(parser.kafAugTokensArrayList.get(parser.kafAugTokensArrayList.size() - 1).getSentenceNum());
		
		if(Integer.parseInt(parser.kafAugTokensArrayList.get(0).getSentenceNum()) == 0)
			numSentences++;
		
		int sent1 = Integer.parseInt(ev1.getFirst().getSentenceNum());
		int sent2 = Integer.parseInt(ev2.getFirst().getSentenceNum());
		
		double dist = Math.abs(sent1 - sent2) * 1.0 / numSentences;
		String error_msg = "sent1: " + sent1 + ", sent2: " + sent2 + ", dist: " + dist;
		
		assertThat(error_msg, dist >= 0 && dist <= 1, is(true));
		
		return dist;
	}

	private boolean sameSentence(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2, KafSaxParser parser) {
		return ev1.getFirst().getSentenceNum().equals(ev2.getFirst().getSentenceNum());
	}

	// class
	private boolean coreferent(LinkedList<KafAugToken> ev1, LinkedList<KafAugToken> ev2) {
		return ev1.getFirst().getEvId().equals(ev2.getFirst().getEvId());
	}

	private void printParsedEvent(HashMap<String, List<DEPNode>> parsedEv) {
		String expansion = "";
		TreeMap<Integer, String> eventPhrase = new TreeMap<Integer, String>();
		for (String role : parsedEv.keySet()) {
			for (DEPNode n : parsedEv.get(role))
				eventPhrase.put(n.getID(), n.getWordForm());
		}
		for (String w : eventPhrase.values())
			expansion += w + " ";
		System.out.println(expansion);
		String depString = "";
		for (String role : parsedEv.keySet()) {
			depString = role + ": ";
			for (DEPNode n : parsedEv.get(role)) {
				depString += n.getWordForm() + " ";
			}
			System.out.println("\t" + depString);
		}
		System.out.println();
	}

	////////////////// not using ////////////////////////////

	public String getWeightedSum(String m_id1, String m_id2, KafSaxParser parser,
			HashMap<String, List<DEPNode>> parsedEv1, HashMap<String, List<DEPNode>> parsedEv2) {

		LinkedList<KafAugToken> ev1 = new LinkedList<KafAugToken>();
		LinkedList<KafAugToken> ev2 = new LinkedList<KafAugToken>();
		for (String t_id : parser.goldMIdToTokenSetMap.get(m_id1))
			ev1.add(parser.tIdToAugTokenMap.get(t_id));
		for (String t_id : parser.goldMIdToTokenSetMap.get(m_id2))
			ev2.add(parser.tIdToAugTokenMap.get(t_id));

		Double[] vec1 = new Double[300];
		Double[] vec2 = new Double[300];
		for (int i = 0; i < vec1.length; i++) {
			vec1[i] = 0.0;
			vec2[i] = 0.0;
		}

		// 0.6
		// trigger
		double c = 0.60;
		// ev 1
		for (DEPNode word : parsedEv1.get("trigger")) {
			// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
			String vector = "";
			String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
					? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
					: "no_es_palabra";
			String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);

			String response = null;

			try {
				HttpResponse<String> serverResponse = Unirest.get(request).asString();
				response = serverResponse.getBody();
			} catch (Exception e) {
				e.printStackTrace();
			}
			try {
				vector = response;
			} catch (Exception e) {

			}
			String[] vecList = vector.split(",");
			for (int i = 0; i < vecList.length; i++) {
				vec1[i] += c * Double.parseDouble(vecList[i]);
			}
		}
		// ev 2
		for (DEPNode word : parsedEv2.get("trigger")) {
			// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
			String vector = "";
			String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
					? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
					: "no_es_palabra";
			String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);

			String response = null;

			try {
				HttpResponse<String> serverResponse = Unirest.get(request).asString();
				response = serverResponse.getBody();
			} catch (Exception e) {
				e.printStackTrace();
			}
			try {
				vector = response;
			} catch (Exception e) {

			}
			String[] vecList = vector.split(",");
			for (int i = 0; i < vecList.length; i++) {
				vec2[i] += c * Double.parseDouble(vecList[i]);
			}
		}

		// 0.19
		// A0
		c = 0.19;
		if (parsedEv1.containsKey("A0")) {
			for (DEPNode word : parsedEv1.get("A0")) {
				// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
				String vector = "";
				String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
						? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
						: "no_es_palabra";
				String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);

				String response = null;

				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try {
					vector = response;
				} catch (Exception e) {

				}
				String[] vecList = vector.split(",");
				for (int i = 0; i < vecList.length; i++) {
					vec1[i] += c * Double.parseDouble(vecList[i]);
				}
			}
		}
		if (parsedEv2.containsKey("A0")) {
			for (DEPNode word : parsedEv2.get("A0")) {
				// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
				String vector = "";
				String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
						? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
						: "no_es_palabra";
				String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);
				String response = null;

				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try {
					vector = response;
				} catch (Exception e) {

				}
				String[] vecList = vector.split(",");
				for (int i = 0; i < vecList.length; i++) {
					vec2[i] += c * Double.parseDouble(vecList[i]);
				}
			}
		}

		// 0.26
		// A1
		c = 0.26;
		if (parsedEv1.containsKey("A1")) {
			for (DEPNode word : parsedEv1.get("A1")) {
				// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
				String vector = "";
				String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
						? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
						: "no_es_palabra";
				String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);
				String response = null;

				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try {
					vector = response;
				} catch (Exception e) {

				}
				String[] vecList = vector.split(",");
				for (int i = 0; i < vecList.length; i++) {
					vec1[i] += c * Double.parseDouble(vecList[i]);
				}
			}
		}
		if (parsedEv2.containsKey("A1")) {
			for (DEPNode word : parsedEv2.get("A1")) {
				// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
				String vector = "";
				String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
						? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
						: "no_es_palabra";
				String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);
				String response = null;

				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try {
					vector = response;
				} catch (Exception e) {

				}
				String[] vecList = vector.split(",");
				for (int i = 0; i < vecList.length; i++) {
					vec2[i] += c * Double.parseDouble(vecList[i]);
				}
			}
		}

		// 0.09
		// TMP
		c = 0.09;
		if (parsedEv1.containsKey("TMP")) {
			for (DEPNode word : parsedEv1.get("TMP")) {
				// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
				String vector = "";
				String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
						? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
						: "no_es_palabra";
				String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);
				String response = null;

				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try {
					vector = response;
				} catch (Exception e) {

				}
				String[] vecList = vector.split(",");
				for (int i = 0; i < vecList.length; i++) {
					vec1[i] += c * Double.parseDouble(vecList[i]);
				}
			}
		}
		if (parsedEv2.containsKey("TMP")) {
			for (DEPNode word : parsedEv2.get("TMP")) {
				// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
				String vector = "";
				String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
						? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
						: "no_es_palabra";
				String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);

				String response = null;

				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try {
					vector = response;
				} catch (Exception e) {

				}
				String[] vecList = vector.split(",");
				for (int i = 0; i < vecList.length; i++) {
					vec2[i] += c * Double.parseDouble(vecList[i]);
				}
			}
		}

		// 0.16
		// LOC
		c = 0.16;
		if (parsedEv1.containsKey("LOC")) {
			for (DEPNode word : parsedEv1.get("LOC")) {
				// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
				String vector = "";
				String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
						? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
						: "no_es_palabra";
				String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);
				String response = null;

				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try {
					vector = response;
				} catch (Exception e) {

				}
				String[] vecList = vector.split(",");
				for (int i = 0; i < vecList.length; i++) {
					vec1[i] += c * Double.parseDouble(vecList[i]);
				}
			}
		}
		if (parsedEv2.containsKey("LOC")) {
			for (DEPNode word : parsedEv2.get("LOC")) {
				// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
				String vector = "";
				String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
						? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
						: "no_es_palabra";
				String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);
				String response = null;

				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try {
					vector = response;
				} catch (Exception e) {

				}
				String[] vecList = vector.split(",");
				for (int i = 0; i < vecList.length; i++) {
					vec2[i] += c * Double.parseDouble(vecList[i]);
				}
			}
		}

		// 0.14
		// Aux

		LinkedList<DEPNode> aux1 = getAuxNodes(parsedEv1);
		LinkedList<DEPNode> aux2 = getAuxNodes(parsedEv2);

		if (aux1.size() > 1) {
			for (DEPNode word : aux1) {
				// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
				String vector = "";
				String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
						? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
						: "no_es_palabra";
				String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);
				String response = null;

				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try {
					vector = response;
				} catch (Exception e) {

				}
				String[] vecList = vector.split(",");
				for (int i = 0; i < vecList.length; i++) {
					vec1[i] += c * Double.parseDouble(vecList[i]);
				}
			}
		}

		if (aux2.size() > 1) {
			for (DEPNode word : aux2) {
				// String cleaned = ComparerUtil.cleanForW2v(word.getWordForm());
				String vector = "";
				String req = word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "").length() > 1
						? word.getWordForm().replaceAll("[^a-zA-Z0-9 -]", "")
						: "no_es_palabra";
				String request = String.format("http://127.0.0.1:9000/?vec=%1$s", req);
				String response = null;

				try {
					HttpResponse<String> serverResponse = Unirest.get(request).asString();
					response = serverResponse.getBody();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try {
					vector = response;
				} catch (Exception e) {

				}
				String[] vecList = vector.split(",");
				for (int i = 0; i < vecList.length; i++) {
					vec2[i] += c * Double.parseDouble(vecList[i]);
				}
			}
		}

		double[] result = new double[300];
		String resultString = "";
		for (int i = 0; i < vec1.length; i++) {

			resultString += String.valueOf(vec1[i] - vec2[i]) + ",";
		}
		resultString += coreferent(ev1, ev2);
		;

		return resultString;

	}

}
