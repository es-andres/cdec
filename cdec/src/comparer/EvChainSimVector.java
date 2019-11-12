package comparer;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.not;
import static org.junit.Assert.assertThat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

import common.GeneralTuple;
import edu.emory.clir.clearnlp.dependency.DEPNode;
import eu.kyotoproject.kaf.KafAugToken;
import eu.kyotoproject.kaf.KafSaxParser;

public class EvChainSimVector {

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
	// 7.1 (discourse_distance)
	private double relative_sentence_start_position = 0;
	// 7.2 (discourse_distance)
	private double relative_sentence_end_position = 0;
	// 8 (sentence distance N/A for chains)
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

	// class
	private boolean coreferent;

	public EvChainSimVector() {

	}

	public String calculateCDEventChainSimVector(HashMap<String, LinkedList<String>> ev1mIdToTokenIDList,
												 HashMap<String, LinkedList<String>> ev2mIdToTokenIDList, 
												 KafSaxParser parser1, 
												 KafSaxParser parser2,
												 String ev_id1,
												 String ev_id2) {

		LinkedList<KafAugToken> ev1 = new LinkedList<KafAugToken>();

		for (String m_id : ev1mIdToTokenIDList.keySet()) {
			for (String t_id : parser1.goldMIdToTokenSetMap.get(m_id))
				ev1.add(parser1.tIdToAugTokenMap.get(t_id));
		}

		LinkedList<KafAugToken> ev2 = new LinkedList<KafAugToken>();

		for (String m_id : ev2mIdToTokenIDList.keySet()) {
			for (String t_id : parser2.goldMIdToTokenSetMap.get(m_id))
				ev2.add(parser2.tIdToAugTokenMap.get(t_id));
		}

		// m_id, <role, nodes>
		HashMap<String, List<DEPNode>> flatParsedEvChain1 = new HashMap<String, List<DEPNode>>();
		HashSet<String> seenSentences = new HashSet<String>();
		HashMap<String,String> chain1MidToSentence = new HashMap<String,String>();
		HashMap<String, HashMap<String, List<DEPNode>>> chain1MidToParsedEv = new HashMap<String, HashMap<String, List<DEPNode>>> ();
		
		
		// add all tokens across all roles occurring across all sentences in this event
		// chain
		// do this for all events (each event given by an m_id)
		for (String m_id : ev1mIdToTokenIDList.keySet()) {
			
			String anchor_t_id = ev1mIdToTokenIDList.get(m_id).getFirst();
			KafAugToken anchorTok = parser1.tIdToAugTokenMap.get(anchor_t_id);
			String sentence = parser1.getSentenceFromSentenceId(anchorTok.getSentenceNum());
			
			
			// only process each sentence once
			if (seenSentences.add(anchorTok.getSentenceNum())) {
				
				chain1MidToSentence.put(m_id, sentence);
				
				HashMap<String, Integer> dummyPosCounts = new HashMap<String, Integer>();
				AtomicInteger dummySrlCount = new AtomicInteger();
				HashMap<String, HashMap<String, List<DEPNode>>> mIdToEvents = ComparerUtil.partitionEventsWithinSentence(sentence,
																				parser1.sentenceNumToMidSet.get(anchorTok.getSentenceNum()), 
																			  parser1, 
																			  dummyPosCounts,
																			  dummySrlCount);
				
				for(String parsed_m_id : mIdToEvents.keySet()) {
					if(ev1mIdToTokenIDList.containsKey(parsed_m_id))
						chain1MidToParsedEv.put(parsed_m_id, mIdToEvents.get(parsed_m_id));
				}
				
				for (String parsed_m_id : mIdToEvents.keySet()) {
					if(ev1mIdToTokenIDList.containsKey(parsed_m_id)) {
						for(String role : mIdToEvents.get(parsed_m_id).keySet()) {
							if(!flatParsedEvChain1.containsKey(role))
								flatParsedEvChain1.put(role, new LinkedList<DEPNode>());
							flatParsedEvChain1.get(role).addAll(mIdToEvents.get(parsed_m_id).get(role));
						}
					}
				}
			}
		}

		// <role, nodes>
		HashMap<String, List<DEPNode>> flatParsedEvChain2 = new HashMap<String, List<DEPNode>>();
		seenSentences = new HashSet<String>();
		HashMap<String,String> chain2MidToSentence = new HashMap<String,String>();
		HashMap<String, HashMap<String, List<DEPNode>>>  chain2MidToParsedEv = new HashMap<String, HashMap<String, List<DEPNode>>>();

		// add all tokens across all roles occurring across all sentences in this event
		// chain
		// do this for all events (each event given by an m_id)
		for (String m_id : ev2mIdToTokenIDList.keySet()) {
			
			String anchor_t_id = ev2mIdToTokenIDList.get(m_id).getFirst();
			KafAugToken anchorTok = parser2.tIdToAugTokenMap.get(anchor_t_id);
			String sentence = parser2.getSentenceFromSentenceId(anchorTok.getSentenceNum());
			
			// only process each sentence once
			if (seenSentences.add(anchorTok.getSentenceNum())) {
				
				chain2MidToSentence.put(m_id, sentence);
				
				HashMap<String, Integer> dummyPosCounts = new HashMap<String, Integer>();
				AtomicInteger dummySrlCount = new AtomicInteger();
				HashMap<String, HashMap<String, List<DEPNode>>> mIdToEvents = ComparerUtil.partitionEventsWithinSentence(sentence,
																				parser2.sentenceNumToMidSet.get(anchorTok.getSentenceNum()), 
																			  parser2, 
																			  dummyPosCounts,
																			  dummySrlCount);
				
				for(String parsed_m_id : mIdToEvents.keySet()) {
					if(ev2mIdToTokenIDList.containsKey(parsed_m_id))
						chain2MidToParsedEv.put(parsed_m_id, mIdToEvents.get(parsed_m_id));
				}
				
				for (String parsed_m_id : mIdToEvents.keySet()) {
					if(ev2mIdToTokenIDList.containsKey(parsed_m_id)) {
						for (String role : mIdToEvents.get(parsed_m_id).keySet()) {
							if (!flatParsedEvChain2.containsKey(role))
								flatParsedEvChain2.put(role, new LinkedList<DEPNode>());
							flatParsedEvChain2.get(role).addAll(mIdToEvents.get(parsed_m_id).get(role));
						}
					}
				}
			}
		}
		
		assertThat(flatParsedEvChain1.get("trigger").size() > 0,is(true));
		assertThat(flatParsedEvChain2.get("trigger").size() > 0,is(true));
		
		StringBuilder vector = new StringBuilder();

		// 1
		trigger_sim_vec = phraseOverlapVec(flatParsedEvChain1.get("trigger"), flatParsedEvChain2.get("trigger"));
		vector.append(trigger_sim_vec + ",");
		// 2
		A0_sim_vec = ((flatParsedEvChain1.containsKey("A0") && flatParsedEvChain2.containsKey("A0"))
				? phraseOverlapVec(flatParsedEvChain1.get("A0"), flatParsedEvChain2.get("A0"))
				: 0);
		vector.append(A0_sim_vec + ",");
		// 3
		A1_sim_vec = ((flatParsedEvChain1.containsKey("A1") && flatParsedEvChain2.containsKey("A1"))
				? phraseOverlapVec(flatParsedEvChain1.get("A1"), flatParsedEvChain2.get("A1"))
				: 0);
		vector.append(A1_sim_vec + ",");
		// 4
		TMP_sim_vec = ((flatParsedEvChain1.containsKey("TMP") && flatParsedEvChain2.containsKey("TMP"))
				? phraseOverlapVec(flatParsedEvChain1.get("TMP"), flatParsedEvChain2.get("TMP"))
				: 0);
		vector.append(TMP_sim_vec + ",");
		// 5
		LOC_sim_vec = ((flatParsedEvChain1.containsKey("LOC") && flatParsedEvChain2.containsKey("LOC"))
				? phraseOverlapVec(flatParsedEvChain1.get("LOC"), flatParsedEvChain2.get("LOC"))
				: 0);
		vector.append(LOC_sim_vec + ",");
		// 6
		LinkedList<DEPNode> aux1 = getAuxNodes(flatParsedEvChain1);
		LinkedList<DEPNode> aux2 = getAuxNodes(flatParsedEvChain2);
		aux_arg_sim_vec = ((aux1.size() > 1 && aux2.size() > 1) ? phraseOverlapVec(aux1, aux2) : 0);
		vector.append(aux_arg_sim_vec + ",");
		// 7.1
		relative_sentence_start_position = relativeChainSentenceStartPosition(parser1,
																			  parser2,
																			  ev1,
																			  ev2);
		vector.append(relative_sentence_start_position + ",");
		// 7.2
		relative_sentence_end_position = relativeChainSentenceEndPosition(parser1,
				  														  parser2,
				  														  ev1,
				  														  ev2);
		vector.append(relative_sentence_end_position + ",");
		// 9
		context_sts = contextSts(chain1MidToParsedEv, 
								 chain2MidToParsedEv, 
								 parser1,
								 parser2);
		vector.append(context_sts + ",");
		// 10
		LinkedList<DEPNode> nodeList1 = new LinkedList<DEPNode>();
		for (String role : flatParsedEvChain1.keySet())
			nodeList1.addAll(flatParsedEvChain1.get(role));
		LinkedList<DEPNode> nodeList2 = new LinkedList<DEPNode>();
		for (String role : flatParsedEvChain2.keySet())
			nodeList2.addAll(flatParsedEvChain2.get(role));
		event_sim_strict = phraseOverlapStrict(nodeList1, nodeList2);
		vector.append(event_sim_strict + ",");
		// 11
		event_sts = evStringSts(chain1MidToParsedEv,chain1MidToParsedEv);
		vector.append(event_sts + ",");
		// 12
		trigger_sts = phraseSts(getRoleString(flatParsedEvChain1.get("trigger")), getRoleString(flatParsedEvChain2.get("trigger")));
		vector.append(trigger_sts + ",");
		// 13
		trigger_sim_strict = phraseOverlapStrict(flatParsedEvChain1.get("trigger"), flatParsedEvChain2.get("trigger"));
		vector.append(trigger_sim_strict + ",");
		A0_sim_strict = ((flatParsedEvChain1.containsKey("A0") && flatParsedEvChain2.containsKey("A0"))
				? phraseOverlapStrict(flatParsedEvChain1.get("A0"), flatParsedEvChain2.get("A0"))
				: 0);
		vector.append(A0_sim_strict + ",");
		// 15
		A1_sim_strict = ((flatParsedEvChain1.containsKey("A1") && flatParsedEvChain2.containsKey("A1"))
				? phraseOverlapStrict(flatParsedEvChain1.get("A1"), flatParsedEvChain2.get("A1"))
				: 0);
		vector.append(A1_sim_strict + ",");
		// 16
		TMP_sim_strict = ((flatParsedEvChain1.containsKey("TMP") && flatParsedEvChain2.containsKey("TMP"))
				? phraseOverlapStrict(flatParsedEvChain1.get("TMP"), flatParsedEvChain2.get("TMP"))
				: 0);
		vector.append(TMP_sim_strict + ",");
		// 17
		LOC_sim_strict = ((flatParsedEvChain1.containsKey("LOC") && flatParsedEvChain2.containsKey("LOC"))
				? phraseOverlapStrict(flatParsedEvChain1.get("LOC"), flatParsedEvChain2.get("LOC"))
				: 0);
		vector.append(LOC_sim_strict + ",");
		// 18
		aux_arg_sim_strict = ((aux1.size() > 1 && aux2.size() > 1) ? phraseOverlapStrict(aux1, aux2) : 0);
		vector.append(aux_arg_sim_strict + ",");
		// 19
		lstm_trigger = lstm(getRoleString(flatParsedEvChain1.get("trigger")), getRoleString(flatParsedEvChain1.get("trigger")),
				"trigger");
		vector.append(lstm_trigger + ",");
		// 20
		lstm_context = lstm(getEvContext(parser1, chain1MidToParsedEv.keySet()), getEvContext(parser2, chain2MidToParsedEv.keySet()), "context");
		vector.append(lstm_context + ",");
		
		// class
		coreferent = ev_id1.equals(ev_id2);
		vector.append(coreferent);


		return vector.toString();
	}

	private double phraseOverlapVec(List<DEPNode> p1, List<DEPNode> p2) {

		String phrase1 = "";
		String phrase2 = "";

		for (DEPNode e : p1)
			phrase1 += e.getWordForm() + " ";
		for (DEPNode e : p2)
			phrase2 += e.getWordForm() + " ";

		try {
			phrase1 = URLEncoder.encode(phrase1.substring(0, phrase1.length() - 1), "utf-8");
			phrase2 = URLEncoder.encode(phrase2.substring(0, phrase2.length() - 1), "utf-8");
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}

		String request = String.format("http://127.0.0.1:9000/?word1=%1$s&word2=%2$s", phrase1, phrase2);

		String response = getHTML(request);
		double result = 0;
		try {
			result = Double.parseDouble(response);
		} catch (Exception e) {

		}

		return result;
	}
	
	private double phraseOverlapStrict(List<DEPNode> p1, List<DEPNode> p2) {

		int overlap = 0;
		int maxIntersectSize = Math.min(p1.size(), p2.size());
		TreeSet<Integer> seen = new TreeSet<Integer>();
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

		for (GeneralTuple<String, String> l1 : p1Lemmas) {
			for (GeneralTuple<String, String> l2 : p2Lemmas) {

				if (!seen.contains(Objects.hashCode(l1)) && !seen.contains(Objects.hashCode(l2))) {
					seen.add(Objects.hashCode(l1));
					seen.add(Objects.hashCode(l2));
					if (l1.first.equals(l2.first)) {// && l1.second.equals(l2.second)){
						overlap++;
					}
				}
			}
		}

		return overlap * 1.0 / maxIntersectSize;
	}
	
	private double contextSts(HashMap<String, HashMap<String, List<DEPNode>>> chain1MidToParsedEv, 
							  HashMap<String, HashMap<String, List<DEPNode>>> chain2MidToParsedEv,
							  KafSaxParser f1Parser,
							  KafSaxParser f2Parser) {
		
		double sum = 0;
		int n = 0;
		
		HashSet<String> f1SeenSentences = new HashSet<String>();
		HashSet<String> f2SeenSentences = new HashSet<String>();

		for(String m_id1 : chain1MidToParsedEv.keySet()) {
			String t_id1 = f1Parser.goldMIdToTokenSetMap.get(m_id1).getFirst();
			String chain1_s_id = f1Parser.tIdToAugTokenMap.get(t_id1).getSentenceNum();
			
			if(f1SeenSentences.contains(chain1_s_id))
				continue;
			f1SeenSentences.add(chain1_s_id);
			
			String context1 = getEvChainContextWithinSentence(f1Parser, chain1_s_id, chain1MidToParsedEv.keySet());
		
			for(String m_id2 : chain2MidToParsedEv.keySet()) {
				String t_id2 = f2Parser.goldMIdToTokenSetMap.get(m_id2).getFirst();
				String chain2_s_id = f2Parser.tIdToAugTokenMap.get(t_id2).getSentenceNum();
				
				if(f2SeenSentences.contains(chain2_s_id))
					continue;
				f2SeenSentences.add(chain2_s_id);
				n++;
				
				String context2 = getEvChainContextWithinSentence(f2Parser, chain2_s_id, chain1MidToParsedEv.keySet());
				sum += phraseSts(context1,context2);
			}
		}

		return sum / n*1.0;
	}
	
	private String  getEvChainContextWithinSentence(KafSaxParser parser, String query_s_id, Set<String> chain_m_id_list) {
		
		String context = "";
		
		List<KafAugToken> querySentenceTokenList = parser.sentenceNumToAugTokens.get(query_s_id);
		
		for(KafAugToken t : querySentenceTokenList) {
			// get mentions that corefer with given chain
			if(!t.getEvType().contains("ACT") || chain_m_id_list.contains(t.getMId()))
				context += t.getTokenString() + " ";
		}
		
		context = context.substring(0, context.length() - 1);

		return context;
	}
	
	
	
	private double evStringSts(HashMap<String, HashMap<String, List<DEPNode>>> chain1MidToParsedEv, 
			  			       HashMap<String, HashMap<String, List<DEPNode>>> chain2MidToParsedEv) {
		
		double sum = 0;
		
		for(String m_id1 : chain1MidToParsedEv.keySet()) {
			String evString1 = getEventString(chain1MidToParsedEv.get(m_id1));
			for(String m_id2 : chain2MidToParsedEv.keySet()) {
				String evString2 = getEventString(chain2MidToParsedEv.get(m_id2));
				sum += phraseSts(evString1, evString2);
			}
		}
		
		int n = chain1MidToParsedEv.size() * chain2MidToParsedEv.size();

		return sum / n*1.0;
	}
	
	
	private double phraseSts(String phrase1, String phrase2) {

		try {
			phrase1 = URLEncoder.encode(phrase1, "utf-8");
			phrase2 = URLEncoder.encode(phrase2, "utf-8");
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
		String request = String.format("http://127.0.0.1:9000/?s1=%1$s&s2=%2$s", phrase1, phrase2);

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
			request = String.format("http://127.0.0.1:9001/?cdt1=%1$s&cdt2=%2$s", phrase1, phrase2);
		else if (type.equals("context"))
			request = String.format("http://127.0.0.1:9001/?cdc1=%1$s&cdc2=%2$s", phrase1, phrase2);

		String response = getHTML(request);

		double result = Double.parseDouble(response);

		return result;

	}
	
	private String getRoleString(List<DEPNode> roleNodes) {
		String expansion = "";

		TreeMap<Integer, String> eventPhrase = new TreeMap<Integer, String>();
		for (DEPNode n : roleNodes)
			eventPhrase.put(n.getID(), n.getWordForm());

		for (String w : eventPhrase.values())
			expansion += w + " ";

		return expansion.substring(0, expansion.length() - 1);
	}
		
	public String getEvContext(KafSaxParser parser, String m_id) {

		String t_id = parser.goldMIdToTokenSetMap.get(m_id).getFirst();
		String s_id = parser.tIdToAugTokenMap.get(t_id).getSentenceNum();
		List<KafAugToken> sentenceTokenList = parser.sentenceNumToAugTokens.get(s_id);

		String context = "";
		for (KafAugToken t : sentenceTokenList) {
			if (!t.getEvType().contains("ACT") || t.getMId().equals(m_id)) {
				context += t.getTokenString() + " ";
			}
		}
		context = context.substring(0, context.length() - 1);

		return context;
	}
	
	public String getEvContext(KafSaxParser parser, Set<String> chain_m_id_list) {
		
		String context = "";
		// organize corefering m_ids by sentence
		HashMap<String, Set<String>> sIdTomIdSet = new HashMap<String, Set<String>>();
		for(String m_id : chain_m_id_list) {

			String t_id = parser.goldMIdToTokenSetMap.get(m_id).getFirst();
			String s_id = parser.tIdToAugTokenMap.get(t_id).getSentenceNum();
			List<KafAugToken> sentenceTokenList = parser.sentenceNumToAugTokens.get(s_id);

			for(KafAugToken t : sentenceTokenList) {
				if(chain_m_id_list.contains(t.getMId())) {
					if(!sIdTomIdSet.containsKey(t.getSentenceNum()))
						sIdTomIdSet.put(t.getSentenceNum(), new HashSet<String>());

					assertThat(m_id,is(not((Object)null)));
					assertThat(m_id,is(not("")));
					sIdTomIdSet.get(t.getSentenceNum()).add(m_id);
				}
			}
		}

		HashSet<String> processedT_ids = new HashSet<String>();
		for(String m_id : chain_m_id_list) {
			String t_id = parser.goldMIdToTokenSetMap.get(m_id).getFirst();
			String s_id = parser.tIdToAugTokenMap.get(t_id).getSentenceNum();
			// get sentence of m_id
			List<KafAugToken> sentenceTokenList = parser.sentenceNumToAugTokens.get(s_id);

			for (KafAugToken t : sentenceTokenList) {
				if (!t.getEvType().contains("ACT") || sIdTomIdSet.get(t.getSentenceNum()).contains(t.getMId())) {
					if(!processedT_ids.contains(t.getTid())){
						context += t.getTokenString() + " ";
						processedT_ids.add(t.getTid());
					}
				}
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
	
	//12 chain
	private double relativeChainSentenceStartPosition(KafSaxParser f1Parser,
			  										  KafSaxParser f2Parser, 
			  										  LinkedList<KafAugToken> ev1,
			  										  LinkedList<KafAugToken> ev2) {
		
		int chain1Start = 1000000;
		for(KafAugToken tok : ev1) {
			if (Integer.parseInt(tok.getSentenceNum()) < chain1Start)
				chain1Start = (Integer.parseInt(tok.getSentenceNum()));
		}
		int doc1NumSentences = Integer.parseInt(f1Parser.kafAugTokensArrayList.get(f1Parser.kafAugTokensArrayList.size()-1).getSentenceNum());
		
		if(Integer.parseInt(f1Parser.kafAugTokensArrayList.get(0).getSentenceNum()) == 0)
			 doc1NumSentences++;
		
			int chain2Start = 1000000;
		for(KafAugToken tok : ev2) {
			if (Integer.parseInt(tok.getSentenceNum()) < chain2Start)
				chain2Start = (Integer.parseInt(tok.getSentenceNum()));
		}
		int doc2NumSentences = Integer.parseInt(f2Parser.kafAugTokensArrayList.get(f2Parser.kafAugTokensArrayList.size()-1).getSentenceNum());
		
		if(Integer.parseInt(f2Parser.kafAugTokensArrayList.get(0).getSentenceNum()) == 0)
			 doc2NumSentences++;
		
		
		double c1RelativeStart = chain1Start*1.0/doc1NumSentences;
		double c2RelativeStart = chain2Start*1.0/doc2NumSentences;
		
		double res = Math.abs(c1RelativeStart - c2RelativeStart);
		String error_msg = "c1Start: " + c1RelativeStart + ", c2Start: " + c2RelativeStart + ", d1Num: " + doc1NumSentences + ", d2Num: " + doc2NumSentences + ", dist: " + res;
		assertThat(error_msg, res >= 0 && res <= 1,is(true));
		
		return res;
	}
	//13 chain
	private double relativeChainSentenceEndPosition(KafSaxParser f1Parser,
			  										KafSaxParser f2Parser, 
			  										LinkedList<KafAugToken> ev1,
			  										LinkedList<KafAugToken> ev2) {
		
		int chain1End = -1;
		for(KafAugToken tok : ev1) {
			if (Integer.parseInt(tok.getSentenceNum()) > chain1End)
				chain1End = (Integer.parseInt(tok.getSentenceNum()));
		}
		int doc1NumSentences = Integer.parseInt(f1Parser.kafAugTokensArrayList.get(f1Parser.kafAugTokensArrayList.size()-1).getSentenceNum());
		
		if(Integer.parseInt(f1Parser.kafAugTokensArrayList.get(0).getSentenceNum()) == 0)
			 doc1NumSentences++;
		
		int chain2End = -1;
		for(KafAugToken tok : ev2) {
			if (Integer.parseInt(tok.getSentenceNum()) > chain2End)
				chain2End = (Integer.parseInt(tok.getSentenceNum()));
		}
		int doc2NumSentences = Integer.parseInt(f2Parser.kafAugTokensArrayList.get(f2Parser.kafAugTokensArrayList.size()-1).getSentenceNum());
		
		if(Integer.parseInt(f2Parser.kafAugTokensArrayList.get(0).getSentenceNum()) == 0)
			 doc2NumSentences++;
		
		
		
		double c1RelativeEnd = chain1End*1.0/doc1NumSentences;
		double c2RelativeEnd = chain2End*1.0/doc2NumSentences;
		
		double res = Math.abs(c1RelativeEnd - c2RelativeEnd);
		String error_msg = "c1End: " + c1RelativeEnd + ", c2End: " + c2RelativeEnd + ", d1Num: " + doc1NumSentences + ", d2Num: " + doc2NumSentences + ", dist: " + res;
		assertThat(error_msg, res >= 0 && res <= 1,is(true));

		return res;
	}


}
