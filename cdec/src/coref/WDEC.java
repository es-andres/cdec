package coref;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.not;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import com.google.common.collect.*;
import com.google.common.graph.EndpointPair;
import com.google.common.graph.Graph;
import com.google.common.graph.GraphBuilder;
import com.google.common.graph.Graphs;
import com.google.common.graph.MutableGraph;
import static org.junit.Assert.*;

import common.GeneralTuple;
import common.Globals;
import comparer.EvSimilarityVector;
import eu.kyotoproject.kaf.KafSaxParser;
import weka.Weka;

public class WDEC {
	
	public HashMap<File, HashSet<String>> fileToWD_idSet;
	//WD_id is unique
	public HashMap<String, HashSet<String>> WD_idToM_idSet;
	//M_id (from ECB+) is not unique, so needs to be
	//mapped to file it belongs to
	public HashMap<File,HashMap<String,String>> fileToM_idToWD_id;
	public LinkedList<File> evaluatedFiles;
	private List<File> allFiles;
	public HashMap<HashSet<String>, String> cachedVectors;
	public HashMap<String,HashSet<String>> cleanSentences;
	public Weka classifier;
	public int maxWD_id;
	int wd_id;
	List<Double> confidenceStats = new LinkedList<Double>();
	DoubleSummaryStatistics confStats;
	
	public WDEC(List<File> allFiles, String cachedVectorsPath, HashMap<String, HashSet<String>> cleanSentences, Weka classifier) {
		
		this.evaluatedFiles = new LinkedList<File>();
		this.allFiles = allFiles;
		this.cachedVectors = loadCachedVectors(cachedVectorsPath);
		this.classifier = classifier;
		this.cleanSentences = cleanSentences;
		fileToWD_idSet = new HashMap<File, HashSet<String>>();
		WD_idToM_idSet = new HashMap<String, HashSet<String>>();
		fileToM_idToWD_id = new HashMap<File,HashMap<String,String>>();
	}
	
	private HashMap<HashSet<String>, String> loadCachedVectors(String cachedVectorsPath) {
		HashMap<HashSet<String>, String> cache = new HashMap<HashSet<String>,String>();
		
		try (BufferedReader br = new BufferedReader(new FileReader(cachedVectorsPath))) {
		    String line;
		    while ((line = br.readLine()) != null) {
		        String row = line;
		        int firstComma = row.indexOf(",");
		        HashSet<String> key = Stream.of(row.substring(0,firstComma).split("\\|")).collect(Collectors.toCollection(HashSet::new));
		        String vector = row.substring(firstComma + 1,  row.length());
		        cache.put(key, vector);
		    }
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		return cache;
	}
	public HashMap<String, HashMap<String, LinkedList<File>>> getFileClusters() {
		
		HashMap<String, HashMap<String, LinkedList<File>>> clusters = new HashMap<String, HashMap<String,LinkedList<File>>>();
		
		for(File f : allFiles) {
			String topic = f.getName().split("_")[0];
			String subTopic = f.getName().split("_")[1].replaceAll("[0-9]", "");
			if(!Globals.TEST_TOPICS.contains(topic))
				continue;
			if(Globals.DOING_DEV) {
				if(!Globals.DEV_TOPICS.contains(topic))
					continue;
			}
			
			if(!clusters.containsKey(topic))
				clusters.put(topic, new HashMap<String, LinkedList<File>>());
			if(!clusters.get(topic).containsKey(subTopic))
				clusters.get(topic).put(subTopic, new LinkedList<File>());
			clusters.get(topic).get(subTopic).add(f);
		}
		
		return clusters;
	}
	private void wdec(LinkedList<GeneralTuple<String,KafSaxParser>> allMids, boolean transitive, boolean crossDoc){
		
		HashMap<String,HashSet<String>> mIdToWD_id = new HashMap<String,HashSet<String>>();
		HashMap<HashSet<String>,Double> confidenceMap = new HashMap<HashSet<String>,Double>();
		
		HashMap<String, HashSet<String>> mIdToCorefs = new HashMap<String, HashSet<String>>();
		HashSet<HashSet<String>> seen = new HashSet<HashSet<String>>();
		MutableGraph<String> graph = GraphBuilder.undirected().build();
		
		// do all m_ids in this cluster
		for(GeneralTuple<String, KafSaxParser> m_id_and_parser1 : allMids) {
			for(GeneralTuple<String, KafSaxParser> m_id_and_parser2 : allMids) {
			
				KafSaxParser parser1 = m_id_and_parser1.second;
				KafSaxParser parser2 = m_id_and_parser2.second;
				
				File f1 = parser1.file;
				File f2 =  parser2.file;

				String fileKey1 = f1.getName().replace("_aug.en.naf", "");
				String fileKey2 = f2.getName().replace("_aug.en.naf", "");
				
				// one of the files wasn't cleaned by ecb+ people
				if(!cleanSentences.containsKey(fileKey1) || !cleanSentences.containsKey(fileKey2))
					continue;
				
				evaluatedFiles.add(f1);
				evaluatedFiles.add(f2);
				
				String m_id1 = m_id_and_parser1.first;
				String m_id2 = m_id_and_parser2.first;
				
				String m_id1_id = fileKey1 + "_" + m_id1;
				String m_id2_id = fileKey2 + "_" + m_id2;
				
				if(fileKey1.equals(fileKey2) && m_id1.equals(m_id2))
					continue;
				
				HashSet<String> thisPair = Stream.of(m_id1_id,m_id2_id).collect(Collectors.toCollection(HashSet::new));
				if(seen.contains(thisPair))
					continue;
				seen.add(thisPair);
				
					
				String t_id1 = parser1.goldMIdToTokenSetMap.get(m_id1).getFirst();
				String t_id2 = parser2.goldMIdToTokenSetMap.get(m_id2).getFirst();
				
				String s_id1 = parser1.tIdToAugTokenMap.get(t_id1).getSentenceNum();
				String s_id2 = parser2.tIdToAugTokenMap.get(t_id2).getSentenceNum();
				
				// one of the sentences wasn't cleaned, continue
				if(!cleanSentences.get(fileKey1).contains(s_id1) || !cleanSentences.get(fileKey2).contains(s_id2))
					continue;

				String ev_id1 = parser1.tIdToAugTokenMap.get(t_id1).getEvId();
				String ev_id2 = parser2.tIdToAugTokenMap.get(t_id2).getEvId();
				
				if(!ev_id1.contains("ACT") || !ev_id2.contains("ACT"))
					continue;

				assertThat(ev_id1.contains("ACT"),is(true));
				assertThat(ev_id2.contains("ACT"),is(true));
				HashSet<String> vectorKey = null;
				if(crossDoc)
					vectorKey = Stream.of(fileKey1, fileKey2, m_id1, m_id2).collect(Collectors.toCollection(HashSet::new));
				else
					vectorKey = Stream.of(fileKey1, fileKey2, ev_id1,ev_id2).collect(Collectors.toCollection(HashSet::new));

				String vector = null;
				
				vector = cachedVectors.get(vectorKey);
				
				assertNotNull((Object)vector);

				GeneralTuple<String,Double> predResult = this.classifier.classifyInstance(vector);

				//record pred confidence for this pair
				HashSet<String> thisMidPair = Stream.of(m_id1_id,m_id2_id).collect(Collectors.toCollection(HashSet::new));
				assertThat(confidenceMap.containsKey(thisMidPair), is(false));
				confidenceMap.put(thisMidPair, predResult.second);
				confidenceStats.add(predResult.second);

				if(predResult.first.equals("true")) {

					if(transitive) {
						graph.putEdge(m_id1_id,m_id2_id);
					}
					else {
						if(!mIdToCorefs.containsKey(m_id1_id))
							mIdToCorefs.put(m_id1_id, new HashSet<String>());
						mIdToCorefs.get(m_id1_id).add(m_id2_id);
						if(!mIdToCorefs.containsKey(m_id2_id))
							mIdToCorefs.put(m_id2_id, new HashSet<String>());
						mIdToCorefs.get(m_id2_id).add(m_id1_id);
					}
				}
			}
		}//done with pairwise comparisons in file, now to disambiguate
			
		//this returns an unique chain assignment for every event
		if(transitive) {
			Graph<String> chains = Graphs.transitiveClosure(graph);

			ArrayList<HashSet<String>> corefChains = new ArrayList<HashSet<String>>();

			boolean first = true;
			for(EndpointPair<String> edge : (Set<EndpointPair<String>>)chains.edges()) {
				if(first) {
					HashSet<String> corefChain = new HashSet<String>();
					for(String m_id : edge) {

						corefChain.add(m_id);
					}
					
					corefChains.add(corefChain);
					first = false;
				}
				else {
					ListIterator<HashSet<String>> it = corefChains.listIterator();
					int i = 0;
					boolean contains = false;
					while(it.hasNext()) {
						HashSet<String> corefChain = it.next();
						for(String m_id : edge) {
							if(corefChain.contains(m_id)) {
								contains = true;
							}
						}
						if(contains) {
							for(String m_id : edge) {
								corefChains.get(i).add(m_id);
							}
						}
						i++;	
					}
					if(!contains) {
						HashSet<String> newChain = new HashSet<String>();
						for(String m_id : edge)
							newChain.add(m_id);
						corefChains.add(newChain);
					}
				}
			}
			for(HashSet<String> corefChain : corefChains){
				for(String m_id : corefChain) {
					assertThat(mIdToWD_id.get(m_id), is((Object)null));
					mIdToWD_id.put(m_id, new HashSet<String>());
					mIdToWD_id.get(m_id).add(String.valueOf(wd_id));
				}
				wd_id++;
			}
		}
		// broken right now (need to add fileKeys to m_id1, m_id2 in confidenceMap and then from where
		// it's searched from
		//otherwise do clustering
		else {
			for(String m_id_id : mIdToCorefs.keySet()) {
				for(String corefMid : mIdToCorefs.get(m_id_id)) {
					HashSet<String> checkPair = Stream.of(m_id_id,corefMid).collect(Collectors.toCollection(HashSet::new));
					assertThat(confidenceMap.containsKey(checkPair), is(true));
				}
			}
			mIdToWD_id = constructAndFindBestCandidateChains(mIdToCorefs,confidenceMap);
		}

		/*
		 * record results for this file (which could be a single file or many
		 */
		// non-singletons
		for(GeneralTuple<String, KafSaxParser> m_id_and_parser : allMids) {
			File f = m_id_and_parser.second.file;
			String fileKey = f.getName().replace("_aug.en.naf", "");
			String m_id_key = fileKey + "_" + m_id_and_parser.first;
			String m_id = m_id_and_parser.first;
			
			if(!mIdToWD_id.containsKey(m_id_key))
				continue;
			
			if(!fileToWD_idSet.containsKey(f))
				fileToWD_idSet.put(f, new HashSet<String>());
			if(!fileToM_idToWD_id.containsKey(f))
				fileToM_idToWD_id.put(f, new HashMap<String,String>());
			
			assertThat(mIdToWD_id.get(m_id_key).size(),is(1));
			String wdId = mIdToWD_id.get(m_id_key).iterator().next();
			
			fileToM_idToWD_id.get(f).put(m_id, wdId);
			fileToWD_idSet.get(f).add(wdId);
			if(!WD_idToM_idSet.containsKey(wdId))
				WD_idToM_idSet.put(wdId, new HashSet<String>());
			WD_idToM_idSet.get(wdId).add(m_id);
		
		}
		// singletons
		for(GeneralTuple<String, KafSaxParser> m_id_and_parser : allMids) {
			File f = m_id_and_parser.second.file;
			String fileKey = f.getName().replace("_aug.en.naf", "");
			String m_id_key = fileKey + "_" + m_id_and_parser.first;
			String m_id = m_id_and_parser.first;
			
			if(mIdToWD_id.containsKey(m_id_key))
				continue;
			
			String wdId = String.valueOf(wd_id++);
			if(!fileToWD_idSet.containsKey(f))
				fileToWD_idSet.put(f, new HashSet<String>());
			if(!fileToM_idToWD_id.containsKey(f))
				fileToM_idToWD_id.put(f, new HashMap<String,String>());
			
			fileToM_idToWD_id.get(f).put(m_id, wdId);
			fileToWD_idSet.get(f).add(wdId);
			
			if(!WD_idToM_idSet.containsKey(wdId))
				WD_idToM_idSet.put(wdId, new HashSet<String>());
			WD_idToM_idSet.get(wdId).add(m_id);
		}
	}
	public HashMap<File,HashMap<String,String>> doGoldWDEC(boolean transitive, boolean crossDoc) {
		
		HashMap<String, HashMap<String, LinkedList<File>>> clusters = getFileClusters();
		
		//unique id for each chain
		wd_id = 1;
		
		for(String topic : clusters.keySet()) {
			System.out.println(topic);
			for(String subTopic : clusters.get(topic).keySet()) {
				
				if(crossDoc) {
					LinkedList<GeneralTuple<String,KafSaxParser>> allMids = new LinkedList<GeneralTuple<String,KafSaxParser>>();
					
					for(File f : clusters.get(topic).get(subTopic)) {

						KafSaxParser parser = new KafSaxParser();
						boolean augmented = true;
						parser.parseFile(f,augmented);
						
						// combine mentions in all files 
						for(String m_id : parser.getGoldActions().keySet())
							allMids.add(new GeneralTuple<String,KafSaxParser>(m_id, parser));
					}

					wdec(allMids,transitive,crossDoc);
					if(!transitive)
						System.out.println("done with " + topic + ", " + subTopic);
				}
				else {
					
					for(File f : clusters.get(topic).get(subTopic)) {
						LinkedList<GeneralTuple<String,KafSaxParser>> allMids = new LinkedList<GeneralTuple<String,KafSaxParser>>();
						

						KafSaxParser parser = new KafSaxParser();
						boolean augmented = true;
						parser.parseFile(f,augmented);
						
						// combine mentions in all files 
						for(String m_id : parser.getGoldActions().keySet()) {

							allMids.add(new GeneralTuple<String,KafSaxParser>(m_id, parser));
						}

						wdec(allMids,transitive,crossDoc);

					}
				}
			}
		}//done with all files	
//		list.stream()
//	     .collect(Collectors.summarizingDouble(Rectangle::getWidth)); 
		confStats = confidenceStats.stream()
								   .collect(DoubleSummaryStatistics::new, 
										   	DoubleSummaryStatistics::accept,
										   	DoubleSummaryStatistics::combine);
		if(!transitive) {
			System.out.println("Summary stats for binary prediction confidence:");
			System.out.println(confStats);
		}
		
		maxWD_id = wd_id; 
		return fileToM_idToWD_id;	
	}

	private HashMap<String,HashSet<String>> constructAndFindBestCandidateChains(HashMap<String, HashSet<String>> mIdToCorefs, HashMap<HashSet<String>,Double> confidenceMap) {

		//this will contain every possible coreference chain
		HashMap<String,HashSet<HashSet<String>>> anchorMidToItsPowerChains = new HashMap<String,HashSet<HashSet<String>>>();
		/*
		 * 2
		 */
		//get all coref-chains where are least one pair corefers
		for(String anchorMid : mIdToCorefs.keySet()) {

			anchorMidToItsPowerChains.put(anchorMid, new HashSet<HashSet<String>>());
			// add singleton
			anchorMidToItsPowerChains.get(anchorMid).add(Stream.of(anchorMid).collect(Collectors.toCollection(HashSet::new)));
			
			Set<Set<String>> powerChains = Sets.powerSet(mIdToCorefs.get(anchorMid));
			for(Set<String> chain : powerChains) {
				
				if(chain.size() <= 1 )
					continue;
				
				double sum = 0;
				double count = 0;
				Set<Set<String>> pairs = Sets.powerSet(chain);
				for(Set<String> pair : pairs) {
					if(pair.size() == 2) {
						sum += confidenceMap.get(pair);
						count++;
					}
				}
				double avg_conf = sum/count*1.0;
				assertThat(avg_conf >= 0 && avg_conf <= 1, is(true));
				
				if(avg_conf < 0.9)
					continue;
				
				HashSet<String> auxSet = new HashSet<String>(chain);
				auxSet.add(anchorMid);
				anchorMidToItsPowerChains.get(anchorMid).add(auxSet);
				
			}
		}
		HashMap<String,HashSet<String>> auxM_idToWD_id = new HashMap<String,HashSet<String>>();
		HashMap<String,HashSet<String>> anchorMidToChainIds = new HashMap<String,HashSet<String>>();
		HashSet<HashSet<String>> candidateChains = new HashSet<HashSet<String>>(); 
		//record chain membership for each mention
		//for each chain of mention ids
		for(String anchor_mId : anchorMidToItsPowerChains.keySet()) {
			//all include anchor_mid
			for(HashSet<String> coreferingSubsetOfPowerset : anchorMidToItsPowerChains.get(anchor_mId)) {
				candidateChains.add(coreferingSubsetOfPowerset);
				for(String m_id : coreferingSubsetOfPowerset) {
					if(!auxM_idToWD_id.containsKey(m_id))
						auxM_idToWD_id.put(m_id, new HashSet<String>());
					auxM_idToWD_id.get(m_id).add(String.valueOf(wd_id));
					if(!anchorMidToChainIds.containsKey(anchor_mId))
						anchorMidToChainIds.put(anchor_mId,new HashSet<String>());
					anchorMidToChainIds.get(anchor_mId).add(String.valueOf(wd_id));
				}
				//declared at top, it's global
				wd_id++;
			}
		}
		/* 
		 * 2.1 
		 */

		while(disambiguationNecessary(auxM_idToWD_id)){ 

			TreeMap<Double,TreeMap<Integer,HashSet<HashSet<String>>>> rankedChains = rankSubsets(candidateChains,confidenceMap);
			HashSet<String> bestChain = getAHighestRankedChain(rankedChains,auxM_idToWD_id,anchorMidToChainIds);

			for(HashSet<String> chain : candidateChains) {
				for(String m_id : bestChain) {
					if(chain.contains(m_id) && !chain.equals(bestChain)) {
						chain.remove(m_id);
					}
				}
			}
			candidateChains.remove(bestChain);
			HashSet<HashSet<String>> auxSet = new HashSet<HashSet<String>>(candidateChains);
			candidateChains.clear();
			for(HashSet<String> s : auxSet) {
				if(s.size() > 0)
					candidateChains.add(s);
			}
		}

		
		
		return auxM_idToWD_id;
		
	}
	
	private boolean disambiguationNecessary(HashMap<String,HashSet<String>> auxM_idToWD_id) {
		for(String m_id : auxM_idToWD_id.keySet()) {
			if (auxM_idToWD_id.get(m_id).size() > 1)
				return true;
		}
		return false;
	}
	private TreeMap<Double,TreeMap<Integer,HashSet<HashSet<String>>>> rankSubsets(HashSet<HashSet<String>> chains,HashMap<HashSet<String>,Double> confidenceMap) {
		//goodness to size to set of md_id
		TreeMap<Double,TreeMap<Integer,HashSet<HashSet<String>>>> rankedChains = new TreeMap<Double,TreeMap<Integer,HashSet<HashSet<String>>>>();
		
		for(HashSet<String> chain : chains) {
			double goodness = goodness(chain,confidenceMap);
			if(!rankedChains.containsKey(goodness))
				rankedChains.put(goodness, new TreeMap<Integer,HashSet<HashSet<String>>>());
			if(!rankedChains.get(goodness).containsKey(chain.size()))
				rankedChains.get(goodness).put(chain.size(), new HashSet<HashSet<String>>());
			rankedChains.get(goodness).get(chain.size()).add(chain);
		}
		

		
		return rankedChains;
	}
	private HashSet<String> getAHighestRankedChain(TreeMap<Double,TreeMap<Integer,HashSet<HashSet<String>>>> rankedChains,HashMap<String,HashSet<String>> auxM_idToWD_id,HashMap<String,HashSet<String>> anchorMidToChainIds) {
		
		double best_goodness = rankedChains.lastKey();
		int largest_size = rankedChains.get(best_goodness).lastKey();
		HashSet<String> aBestChain = rankedChains.get(best_goodness).get(largest_size).iterator().next();
		HashSet<String> anchoredChains = new HashSet<String>();

		for(String anchor : aBestChain)
			anchoredChains.addAll(anchorMidToChainIds.get(anchor));
		Iterator<HashSet<String>> mapIterator = auxM_idToWD_id.values().iterator();

		while(mapIterator.hasNext()) {
			Iterator<String> candidateChain = mapIterator.next().iterator();
			while(candidateChain.hasNext()) {
				if(anchoredChains.contains(candidateChain.next()))
					candidateChain.remove();
			}
		}

		String newId = String.valueOf(wd_id++);
		for (String anchorId : aBestChain) {
			auxM_idToWD_id.put(anchorId, new HashSet<String>());
			auxM_idToWD_id.get(anchorId).add(newId);
		}
		
		return aBestChain;
	}
	
	private double goodness(HashSet<String> chain,HashMap<HashSet<String>,Double> confidenceMap) {
		
		if(chain.size() == 1)
			return 1.0;

		HashSet<HashSet<String>> seen = new HashSet<HashSet<String>>();
		double sum = 0;
		int count = 0;
		HashSet<String> completelyCoreferingEvents = new HashSet<String>();
		for(String m_id1 : chain) {
			for(String m_id2 : chain) {
				if(!m_id1.equals(m_id2) && seen.add(Stream.of(m_id1,m_id2).collect(Collectors.toCollection(HashSet::new)))){
					double confidence = confidenceMap.get(Stream.of(m_id1,m_id2).collect(Collectors.toCollection(HashSet::new)));
					sum += confidence;
					count++;
					// {count=41915, sum=37171.420000, min=0.230000, average=0.886829, max=1.000000}
					if(confidence > 0.95) {
						completelyCoreferingEvents.add(m_id1);
						completelyCoreferingEvents.add(m_id2);
					}
				}
			}
		}
		double purity = completelyCoreferingEvents.size()*1.0 / chain.size();
		assertThat(purity >=0 && purity <= 1, is(true));
		
		double avg_conf = sum/count*1.0;
		double goodness = (purity+0.001) * avg_conf;
		assertThat(avg_conf >=0 && avg_conf <= 1, is(true));

		return purity;
	}

}
