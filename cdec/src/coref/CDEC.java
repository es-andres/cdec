package coref;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.not;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.google.common.collect.Sets;
import com.google.common.graph.EndpointPair;
import com.google.common.graph.Graph;
import com.google.common.graph.GraphBuilder;
import com.google.common.graph.Graphs;
import com.google.common.graph.MutableGraph;

import common.GeneralTuple;
import eu.kyotoproject.kaf.KafSaxParser;
import weka.Weka;

public class CDEC {
	
	int cd_id;
	WDEC wdec;
	Weka classifier;
	public HashMap<String, String> cachedVectors;
	public HashMap<String,String> wd_idTocd_id;
	
	public CDEC(WDEC WDEC, Weka classifier, String cachedVectorsPath) {
		wdec = WDEC;
		cd_id = WDEC.maxWD_id + 1;
		this.classifier = classifier;
		wd_idTocd_id = new HashMap<String,String>();
		this.cachedVectors = loadCachedVectors(cachedVectorsPath);
	}
	
	private HashMap<String, String> loadCachedVectors(String cachedVectorsPath) {
		HashMap<String, String> cache = new HashMap<String,String>();
		
		try (BufferedReader br = new BufferedReader(new FileReader(cachedVectorsPath))) {
		    String line;
		    while ((line = br.readLine()) != null) {
		        String row = line;
		        int firstComma = row.indexOf(",");
		        String key = row.substring(0,firstComma);
		        String vector = row.substring(firstComma + 1,  row.length());
		        cache.put(key, vector);
		    }
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		return cache;
	}
	
	public HashMap<String,String> doGoldCDEC(boolean transitive) {
		
		//different for all files
		HashMap<HashSet<String>,Double> confidenceMap = new HashMap<HashSet<String>,Double>();
		HashMap<String, HashSet<String>> wdIdToCorefs = new HashMap<String, HashSet<String>>();
		HashSet<HashSet<String>> filesCompared = new HashSet<HashSet<String>>();
		HashSet<HashSet<String>> wdChainsCompared = new HashSet<HashSet<String>>();
		HashMap<String,HashSet<String>> auxWd_idToCd_id = new HashMap<String,HashSet<String>>();
		MutableGraph<String> graph = GraphBuilder.undirected().build();
		String delim = "|";
		
		for(File f1 : wdec.evaluatedFiles) {
			for(File f2 : wdec.evaluatedFiles) {
				
				// doing cdec
				if(f1.getName().equals(f2.getName()))
					continue;
				
				String file1Key = f1.getName().replace("_aug.en.naf", "");
				String file2Key =  f2.getName().replace("_aug.en.naf", "");
				
				// one of the files wasn't cleaned by ecb+ people
				if(!wdec.cleanSentences.containsKey(file1Key) || !wdec.cleanSentences.containsKey(file2Key))
					continue;
				
				// topics are numbers, subtopics are ecbplus or ecb
				String topic1 = f1.getName().split("_")[0];
				String subTopic1 = f1.getName().split("_")[1].replaceAll("[0-9]", "");
				String topic2 = f2.getName().split("_")[0];
				String subTopic2 = f2.getName().split("_")[1].replaceAll("[0-9]", "");
				
				// out of topic files, skip
				if(!topic1.equals(topic2))
					continue;
				// out of sub-topic files, skip
				if(!subTopic1.equals(subTopic2))
					continue;
				
				// only look at each pair once
				HashSet<String> thisPair = Stream.of(file1Key, file2Key).collect(Collectors.toCollection(HashSet::new));
				if(filesCompared.contains(thisPair))
					continue;
				filesCompared.add(thisPair);
				
				// recap conditions
				assertThat(f1.getName(), is(not(f2.getName())));
				assertThat(wdec.cleanSentences.get(file1Key),is(not((Object)null)));
				assertThat(wdec.cleanSentences.get(file2Key),is(not((Object)null)));
				assertThat(topic1,is(topic2));
				assertThat(subTopic1,is(subTopic2));

				
				boolean augmented = true;
				
				KafSaxParser f1Parser = new KafSaxParser();
				KafSaxParser f2Parser = new KafSaxParser();
				f1Parser.parseFile(f1,augmented);
				f2Parser.parseFile(f2,augmented);
				
				/*
				 * 1: compare every chain in f1 with every chain in f2
				 */
				for(String wdChain1_id : wdec.fileToWD_idSet.get(f1)) {
					
					// get <m_id, tokens> for every event in chain 1
					HashMap<String, LinkedList<String>> chain1mIdToTokenIDList = new HashMap<String,LinkedList<String>>();
					for(String m_id : wdec.WD_idToM_idSet.get(wdChain1_id)) {
						assertThat(chain1mIdToTokenIDList.get(m_id),is((Object)null));
						
						chain1mIdToTokenIDList.put(m_id, new LinkedList<String>());
						chain1mIdToTokenIDList.get(m_id).addAll(f1Parser.goldMIdToTokenSetMap.get(m_id));
					}
					
					for(String wdChain2_id : wdec.fileToWD_idSet.get(f2)) {
						
						HashSet<String> thisCorefPair = Stream.of(wdChain1_id,wdChain2_id).collect(Collectors.toCollection(HashSet::new));
						if(wdChainsCompared.contains(thisCorefPair))
							continue;
						wdChainsCompared.add(thisCorefPair);
						
						// get <m_id, tokens> for every event in chain 2
						HashMap<String, LinkedList<String>> chain2mIdToTokenIDList = new HashMap<String,LinkedList<String>>();
						for(String m_id : wdec.WD_idToM_idSet.get(wdChain2_id)) {
							assertThat(chain2mIdToTokenIDList.get(m_id),is((Object)null));
							
							chain2mIdToTokenIDList.put(m_id, new LinkedList<String>());
							chain2mIdToTokenIDList.get(m_id).addAll(f2Parser.goldMIdToTokenSetMap.get(m_id));
						}
						String vectorKey1 = wdChain1_id + delim + wdChain2_id;
						String vectorKey2 = wdChain2_id + delim + wdChain1_id;
						
						String vector = null;
						
						if(cachedVectors.containsKey(vectorKey1))
							vector = cachedVectors.get(vectorKey1);
						else if(cachedVectors.containsKey(vectorKey2))
							vector = cachedVectors.get(vectorKey2);
						
						assertNotNull((Object)vector);

						GeneralTuple<String,Double> predResult = this.classifier.classifyInstance(vector);
						confidenceMap.put(Stream.of(wdChain1_id,wdChain2_id).collect(Collectors.toCollection(HashSet::new)), predResult.second);

						if(predResult.first.equals("true")) {
							if(transitive) {
								graph.putEdge(wdChain1_id,wdChain2_id);
							}
							else {
								if(!wdIdToCorefs.containsKey(wdChain1_id))
									wdIdToCorefs.put(wdChain1_id, new HashSet<String>());
								wdIdToCorefs.get(wdChain1_id).add(wdChain2_id);
								if(!wdIdToCorefs.containsKey(wdChain2_id))
									wdIdToCorefs.put(wdChain2_id, new HashSet<String>());
								wdIdToCorefs.get(wdChain2_id).add(wdChain1_id);
							}
						}
					}
				}
			}//f2
		}//f1'
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
					auxWd_idToCd_id.put(m_id, new HashSet<String>());
					auxWd_idToCd_id.get(m_id).add(String.valueOf(cd_id));
				}
				cd_id++;
			}
		}
		//otherwise do clustering
		else {
			auxWd_idToCd_id = constructAndFindBestCandidateChains(wdIdToCorefs,confidenceMap);
		}
		
		
//		System.out.println("-> done comparing files");
				
		for(String wd_id :  auxWd_idToCd_id.keySet())
			wd_idTocd_id.put(wd_id, auxWd_idToCd_id.get(wd_id).iterator().next());
		//add singletons
		for (File f : wdec.evaluatedFiles) {
			KafSaxParser parser = new KafSaxParser();
			parser.parseFile(f,true);
			wdec.fileToM_idToWD_id.get(f);
			for(String m_id : parser.goldActionMidToTokenSetMap.keySet()) {
				if(!wd_idTocd_id.containsKey(wdec.fileToM_idToWD_id.get(f).get(m_id))) {
					String cdId = String.valueOf(cd_id++);
					wd_idTocd_id.put(wdec.fileToM_idToWD_id.get(f).get(m_id), cdId);
				}
			}
		}
		
		return wd_idTocd_id;	
	}
	public HashMap<String,HashSet<String>> constructAndFindBestCandidateChains(HashMap<String, HashSet<String>> wdIdToCorefs, HashMap<HashSet<String>,Double> confidenceMap) {
		HashMap<String,HashSet<HashSet<String>>> anchorWdIdToItsPowerChains = new HashMap<String,HashSet<HashSet<String>>>();
		/*
		 * 2
		 */
		//get all coref-chains where are least one pair corefers
		System.out.println("generating power sets...");
		int anc = 1;
		for(String anchorWdId : wdIdToCorefs.keySet()) {
			anchorWdIdToItsPowerChains.put(anchorWdId, new HashSet<HashSet<String>>());
			
			Set<Set<String>> powerChains = Sets.powerSet(wdIdToCorefs.get(anchorWdId));
			System.out.println("size of this anchor's power set: " + powerChains.size());
			for(Set<String> chain : powerChains) {
				if(chain.size() > 0) {
					HashSet<String> auxSet = new HashSet<String>(chain);
					if(goodness(auxSet,confidenceMap) > .5) {
						auxSet.add(anchorWdId);
						anchorWdIdToItsPowerChains.get(anchorWdId).add(auxSet);
					}
				}
			}
			System.out.println("pruned size " + anchorWdIdToItsPowerChains.get(anchorWdId).size());
			System.out.println("anchor " + anc++ +  " / " + wdIdToCorefs.size() +" done");
		}
		System.out.println("-> done generating power sets");
		
		HashMap<String,HashSet<String>> auxWd_idToCd_id = new HashMap<String,HashSet<String>>();
		HashMap<String,HashSet<String>> anchorWdIdToChainIds = new HashMap<String,HashSet<String>>();
		HashSet<HashSet<String>> candidateChains = new HashSet<HashSet<String>>(); 
		//record chain membership for each mention
		//for each chain of mention ids
		System.out.println("assiging a c_id to each possible chain...");
		for(String anchor_WdId : anchorWdIdToItsPowerChains.keySet()) {
			//all include anchor_mid
			for(HashSet<String> coreferingSubsetOfPowerset : anchorWdIdToItsPowerChains.get(anchor_WdId)) {
				candidateChains.add(coreferingSubsetOfPowerset);
				for(String m_id : coreferingSubsetOfPowerset) {
					if(!auxWd_idToCd_id.containsKey(m_id))
						auxWd_idToCd_id.put(m_id, new HashSet<String>());
					auxWd_idToCd_id.get(m_id).add(String.valueOf(cd_id));
					if(!anchorWdIdToChainIds.containsKey(anchor_WdId))
						anchorWdIdToChainIds.put(anchor_WdId,new HashSet<String>());
					anchorWdIdToChainIds.get(anchor_WdId).add(String.valueOf(cd_id));
				}
				//declared at top, it's global
				cd_id++;
			}
		}
		System.out.println(cd_id);
		System.out.println("-> done");
		System.out.println("size of candidate chains " + candidateChains.size());
		/* 
		 * 2.1 
		 */

		while(disambiguationNecessary(auxWd_idToCd_id)){ 
			System.out.println("disambiguating");
			System.out.println(candidateChains.size());

			TreeMap<Double,TreeMap<Integer,HashSet<HashSet<String>>>> rankedChains = rankSubsets(candidateChains,confidenceMap);
			HashSet<String> bestChain = getAHighestRankedChain(rankedChains,auxWd_idToCd_id,anchorWdIdToChainIds);

			for(HashSet<String> chain : candidateChains) {
				for(String wd_id : bestChain) {
					if(chain.contains(wd_id) && !chain.equals(bestChain)) {
						chain.remove(wd_id);
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
		return auxWd_idToCd_id;
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

		String newId = String.valueOf(cd_id++);
		for (String anchorId : aBestChain) {
			auxM_idToWD_id.put(anchorId, new HashSet<String>());
			auxM_idToWD_id.get(anchorId).add(newId);
		}
		
//		if(aBestWD_id == null)
//			System.out.println(rankedChains.get(best_goodness).get(largest_size).keySet());
		return aBestChain;
	}
	
	private double goodness(HashSet<String> chain,HashMap<HashSet<String>,Double> confidenceMap) {
		HashSet<HashSet<String>> seen = new HashSet<HashSet<String>>();
		double sum = 0;
		HashSet<String> completelyCoreferingEvents = new HashSet<String>();
		for(String m_id1 : chain) {
			for(String m_id2 : chain) {
				if(!m_id1.equals(m_id2) && seen.add(Stream.of(m_id1,m_id2).collect(Collectors.toCollection(HashSet::new)))){
					if(confidenceMap.containsKey(Stream.of(m_id1,m_id2).collect(Collectors.toCollection(HashSet::new)))) {
						double confidence = confidenceMap.get(Stream.of(m_id1,m_id2).collect(Collectors.toCollection(HashSet::new)));
						sum += confidence;
						if(confidence > 0.5) {
							completelyCoreferingEvents.add(m_id1);
							completelyCoreferingEvents.add(m_id2);
						}
					}
				}
			}
		}
		double purity = completelyCoreferingEvents.size()*1.0 / chain.size();
		double goodness = purity * sum;
		return goodness;
	}
}
