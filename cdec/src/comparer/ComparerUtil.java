package comparer;

import java.io.InputStream;
import java.io.ObjectInputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Map.Entry;
import java.util.regex.Pattern;

import com.google.common.net.UrlEscapers;
import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.Unirest;

import common.GeneralTuple;
import edu.emory.clir.clearnlp.dependency.DEPNode;
import edu.emory.clir.clearnlp.dependency.DEPTree;
import edu.emory.clir.clearnlp.feature.type.FieldType;
import edu.emory.clir.clearnlp.srl.SRLTree;
import edu.emory.clir.clearnlp.util.arc.SRLArc;
import eu.kyotoproject.kaf.KafAugToken;
import eu.kyotoproject.kaf.KafSaxParser;

public class ComparerUtil {
	
	public static HashMap<String, HashMap<String, List<DEPNode>>> partitionEventsWithinSentence(String sentence, Set<String> m_ids, KafSaxParser parser,HashMap<String, Integer> posCounts, AtomicInteger srlCount) {

		HashMap<String, HashMap<String, List<DEPNode>>> mIdToEventPartition = new HashMap<String, HashMap<String, List<DEPNode>>>();
		boolean doSrl = true;
		boolean doNonSrl = true;
		boolean expandNodes = false;
		
		// get tree
		
		String q = UrlEscapers.urlFormParameterEscaper().escape(sentence);
		String request = String.format("http://localhost:8080/ClearNLPServer/query?nt=%1$s",q);
		
		DEPNode[] nodes = null;
		DEPTree tree = null;
		try {
			InputStream input = new URL(request).openStream();
			ObjectInputStream serverResponse = new ObjectInputStream(input);
			nodes = (DEPNode[])serverResponse.readObject();
			tree = new DEPTree(Arrays.asList(nodes));
		}
		catch(Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
		
		// get event triggers
		
		HashMap<String,LinkedList<KafAugToken>> mIdToAllEventTriggers = new HashMap<String,LinkedList<KafAugToken>>();
		LinkedList<DEPNode> srlEventTriggers = new LinkedList<DEPNode>();
		LinkedList<String> nonSrlEventTriggers = new LinkedList<String>();
		LinkedList<GeneralTuple<DEPNode, LinkedList<DEPNode>>> eventTriggerPhrases = new LinkedList<GeneralTuple<DEPNode, LinkedList<DEPNode>>>();
		
		// only look at ACT (ie. event triggers)
		
		for(String m_id : m_ids) {
			
			if(!parser.goldMIdToEvIdMap.get(m_id).contains("ACT") && !parser.goldMIdToEvIdMap.get(m_id).contains("NEG"))
				continue;

			mIdToAllEventTriggers.put(m_id, new LinkedList<KafAugToken>());
			for(String t_id : parser.goldActionMidToTokenSetMap.get(m_id)) {
				mIdToAllEventTriggers.get(m_id).add(parser.tIdToAugTokenMap.get(t_id));
			}
		}
		
		// Look at all events

		for(String m_id : mIdToAllEventTriggers.keySet()) {
			DEPNode depEventHead = null;
			SRLTree srl = null;
			
			// check if one of the words in the trigger is head in an srl predicate
			
			for(KafAugToken kafTok : mIdToAllEventTriggers.get(m_id)) {
				int kafToDep = Integer.parseInt(kafTok.getIdxInSentence()) + 2;
				DEPNode node = tree.get(kafToDep);
				srl = tree.getSRLTree(node);
				if(srl != null) {
					depEventHead = node;
					break;
				}	
			}
			
			// add all event triggers, in order
			
			TreeMap<Integer, DEPNode> eventTriggerNodes = new TreeMap<Integer, DEPNode>();
			for(KafAugToken kafTok : mIdToAllEventTriggers.get(m_id)) {
				int kafToDep = Integer.parseInt(kafTok.getIdxInSentence()) + 2;
				DEPNode node = tree.get(kafToDep);
				eventTriggerNodes.put(node.getID(), node);
			}
			
			// srl event triggers
			if(doSrl) {
				if(depEventHead != null) {
					// log coungs
					srlCount.incrementAndGet();
					String pos = (depEventHead.getPOSTag().toLowerCase().contains("v") ? "srl_V" : "srl_O" );
					if(!posCounts.containsKey(pos))
						posCounts.put(pos, 0);
					int count = posCounts.get(pos);
					posCounts.put(pos, count + 1);
					
					// add srl event trigger
					if(!mIdToEventPartition.containsKey(m_id))
						mIdToEventPartition.put(m_id, new HashMap<String, List<DEPNode>>());
					LinkedList<DEPNode> danglingPhrase = new LinkedList<DEPNode>();
					for(DEPNode headNode : eventTriggerNodes.values()) {
						if(headNode.getID() != depEventHead.getID()) {
							danglingPhrase.add(headNode);
							eventTriggerNodes.put(headNode.getID(), headNode);
						}
					}
					eventTriggerPhrases.add(new GeneralTuple<DEPNode, LinkedList<DEPNode>>(depEventHead, danglingPhrase));
					srlEventTriggers.add(depEventHead);
					
					// get dependent nodes, in order
					TreeMap<Integer, DEPNode> orderedDeps = new TreeMap<Integer, DEPNode>();
					HashMap<String, TreeMap<Integer, DEPNode>> expandedRoles = new HashMap<String, TreeMap<Integer, DEPNode>>();
					
					orderedDeps.put(depEventHead.getID(), depEventHead);
					// add triggers
					expandedRoles.put("trigger", eventTriggerNodes);
					
					// add roles
					for(SRLArc arc : srl.getArgumentArcList()) {
						DEPNode dep = arc.getNode();
						if(!expandedRoles.containsKey(arc.getLabel()))
							expandedRoles.put(arc.getLabel(), new TreeMap<Integer, DEPNode>());
						Pattern p = Pattern.compile("[a-zA-Z0-9]");
						if(p.matcher(dep.getWordForm()).find()) {
							expandedRoles.get(arc.getLabel()).put(dep.getID(), dep);
							orderedDeps.put(dep.getID(), dep);
						}
						else {
							System.out.println("empty word form");
						}
						
						if (expandNodes) {
							List<DEPNode> subDeps = dep.getDependentList();
							List<DEPNode> compounds = dep.getDependentListByLabel(Pattern.compile("compound"));
							for(DEPNode c : compounds)
								expandedRoles.get(arc.getLabel()).put(c.getID(), c);
							
							if (subDeps != null) {
								for(DEPNode subDep : subDeps) {
									orderedDeps.put(subDep.getID(), subDep);
									if(!eventTriggerNodes.containsKey(subDep.getID()))
										expandedRoles.get(arc.getLabel()).put(subDep.getID(), subDep);
		
										compounds = subDep.getDependentListByLabel(Pattern.compile("compound"));
										for(DEPNode c : compounds)
											expandedRoles.get(arc.getLabel()).put(c.getID(), c);
								}
							}
						}
						
					}
	
					mIdToEventPartition.put(m_id, new HashMap<String, List<DEPNode>>());
					for(String role :  expandedRoles.keySet()) {
						mIdToEventPartition.get(m_id).put(role, new LinkedList<DEPNode>());
						for(DEPNode n : expandedRoles.get(role).values()) {
							mIdToEventPartition.get(m_id).get(role).add(n);
						}
					}
					
				}
				
				// no token in m_id is head of srl predicate
				
				else {
					nonSrlEventTriggers.add(m_id);
					
				}
			} // end srl triggers
			if(!doSrl) {
				if(depEventHead == null)
					nonSrlEventTriggers.add(m_id);
			}
		}

	
		// non srl head event triggers
		
		if(doNonSrl) {
			for(String m_id : nonSrlEventTriggers) {
				LinkedList<KafAugToken> toks = mIdToAllEventTriggers.get(m_id);
				mIdToEventPartition.put(m_id, new HashMap<String, List<DEPNode>>());
				mIdToEventPartition.get(m_id).put("trigger", new LinkedList<DEPNode>());
				DEPNode depNode = null;
				for(KafAugToken tok : toks) {
					int kafToDep = Integer.parseInt(tok.getIdxInSentence()) + 2;
					depNode = tree.get(kafToDep);
					mIdToEventPartition.get(m_id).get("trigger").add(depNode);
				}
				String pos = (depNode.getPOSTag().toLowerCase().contains("v") ? "nonSrl_V" : "nonSrl_O");
				if(!posCounts.containsKey(pos))
					posCounts.put(pos, 0);
				int count = posCounts.get(pos);
				posCounts.put(pos, count + 1);
	
			}
		}
		
		// clean triggers from other roles
		
		for(GeneralTuple<DEPNode, LinkedList<DEPNode>> triggerPhrase : eventTriggerPhrases) {
			for(String m_id : mIdToEventPartition.keySet()) {
				for(String role : mIdToEventPartition.get(m_id).keySet()) {
					// clean all non-trigger roles
					if(!role.equals("trigger")) {
						Iterator<DEPNode> argumentRoleMembers = mIdToEventPartition.get(m_id).get(role).iterator();
						while(argumentRoleMembers.hasNext()) {
							DEPNode n = argumentRoleMembers.next();
							// if this trigger is in any other argument, remove from argument
							// head of trigger
							if(n.getID() == triggerPhrase.first.getID()) {
								argumentRoleMembers.remove();
							}
							// non-head of trigger
							for(DEPNode phraseNode : triggerPhrase.second) {
								if(n.getID() == phraseNode.getID())
									argumentRoleMembers.remove();
							}
								
						}
					}
				}
			}
		}
		
		// remove emptied roles
		Pattern p = Pattern.compile("[a-zA-Z0-9]");
		for(String m_id : mIdToEventPartition.keySet()) {
			Iterator<String> roles = mIdToEventPartition.get(m_id).keySet().iterator();
			while(roles.hasNext()) {
				String role = roles.next();
				// first remove nodes with no word forms
				Iterator<DEPNode> roleNodes = mIdToEventPartition.get(m_id).get(role).iterator();
				while(roleNodes.hasNext()) {
					DEPNode n = roleNodes.next();
					if(!p.matcher(n.getWordForm()).find()) {
						roleNodes.remove();
					}
				}
				// then remove entire roles if emtpy
				if(mIdToEventPartition.get(m_id).get(role).size() == 0)
					roles.remove();
			}
		}
		for(String m_id : mIdToEventPartition.keySet()) {
			for(String role : mIdToEventPartition.get(m_id).keySet()) {
				for(DEPNode n : mIdToEventPartition.get(m_id).get(role)) {
					if(!p.matcher(n.getWordForm()).find()) {
						System.out.println("bad word form:");
						System.out.println("\t" + n.getWordForm());
					}
				}
			}
		}
		
		return mIdToEventPartition;
	}
	
	
	public static String cleanForW2v(String sentence) {
		String out = "";

		for(String w : sentence.split(" ")) {
			w = w.replaceAll("[\" \\s+]","");
			if(w.length() > 0) {
				if (wordInW2v(w).equals("True"))
					out += w + " ";
				// try to get something
				else {
					// remove most punctuation
					w = w.replaceAll("[^a-zA-Z0-9-—]", "");
					String res = wordInW2v(w);
					if(res.equals("True"))
						out += w + " ";
					else {
						// lower case
						w = w.toLowerCase();
						res = wordInW2v(w);
						if(res.equals("True"))
							out += w + " ";
						else {
							// remove all punctuation
							w = w.replaceAll("[-—]", " ");
							for(String sub_w : w.split(" ")) {
								res = wordInW2v(sub_w);
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

	private static String wordInW2v(String w) {
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
	
	 public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
	        List<Entry<K, V>> list = new ArrayList<>(map.entrySet());
	        list.sort(Entry.comparingByValue());
	        Collections.reverse(list);

	        Map<K, V> result = new LinkedHashMap<>();
	        for (Entry<K, V> entry : list) {
	            result.put(entry.getKey(), entry.getValue());
	        }

	        return result;
	    }

}
