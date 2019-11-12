package common;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.not;
import static org.junit.Assert.assertThat;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IntSummaryStatistics;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import com.google.common.graph.Graph;
import com.google.common.graph.Graphs;

import comparer.ComparerUtil;
import comparer.EvChainSimVector;
import comparer.EvPairSimVector;
import comparer.EvSimilarityVector;
import coref.WDEC;
import edu.emory.clir.clearnlp.dependency.DEPNode;
import eu.kyotoproject.kaf.KafAugToken;
import eu.kyotoproject.kaf.KafSaxParser;
import naf.EventNode;
import naf.NafDoc;

public class ArffWriter {

	public ArffWriter() {

	}
	
	

	public File writePairwiseCorefFile(List<File> files, String mode, String arffOutDir, HashMap<String, HashSet<String>> cleanSentences, String fileName) {
		StringBuilder header = new StringBuilder();

		//header
		header.append("@relation ECB+Gold_WDEC\n\n");
		//1
		header.append("@attribute trigger_sim_vec numeric\n");
		//2
		header.append("@attribute A0_sim_vec numeric\n");
		//3
		header.append("@attribute A1_sim_vec numeric\n");
		//4
		header.append("@attribute TMP_sim_vec numeric\n");
		//5
		header.append("@attribute LOC_sim_vec numeric\n");
		//6
		header.append("@attribute aux_arg_sim_vec numeric\n");
		//7
		header.append("@attribute discourse-distance numeric\n");
		//8
		header.append("@attribute sentence-distance numeric\n");
		//9
		header.append("@attribute context_sts numeric\n");
		//10
		header.append("@attribute event_sim_strict numeric\n");
		//11
		header.append("@attribute event_sts numeric\n");
		//12
		header.append("@attribute trigger_sts numeric\n");
		//13
		header.append("@attribute trigger_sim_strict numeric\n");
		//14
		header.append("@attribute A0_sim_strict numeric\n");
		//15
		header.append("@attribute A1_sim_strict numeric\n");
		//16
		header.append("@attribute TMP_sim_strict numeric\n");
		//17
		header.append("@attribute LOC_sim_strict numeric\n");
		//18
		header.append("@attribute aux_arg_sim_strict numeric\n");
		//19
		header.append("@attribute lstm_trigger numeric\n");
		//20
		header.append("@attribute lstm_context numeric\n");
		//21
		header.append("@attribute same_sentence {true,false}\n");
		//22
		header.append("@attribute same_doc {true,false}\n");
		//23 
		header.append("@attribute trigger_wn_sim numeric\n");

		//class
		header.append("@attribute coreferent {true,false}\n\n");
		header.append("@data\n");

		//train/test
		StringBuilder train = new StringBuilder(header);
		StringBuilder test = new StringBuilder(header);
		StringBuilder indexFile = new StringBuilder();


		int i = 0;
		int numEvents = 0;
		HashMap<String, Integer> posCounts = new HashMap<String, Integer>();
		AtomicInteger srl = new AtomicInteger(0);
		HashSet<HashSet<String>> seenFiles = new HashSet<HashSet<String>>();
		EvPairSimVector sv = new EvPairSimVector();
		for(File f1 : files) {
			for(File f2 : files) {

				String fileKey1 = f1.getName().replace("_aug.en.naf", "");
				String fileKey2 = f2.getName().replace("_aug.en.naf", "");
				
				// one of the files wasn't cleaned by ecb+ people
				if(!cleanSentences.containsKey(fileKey1) || !cleanSentences.containsKey(fileKey2))
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
				HashSet<String> thisPair = Stream.of(fileKey1, fileKey2).collect(Collectors.toCollection(HashSet::new));
				if(seenFiles.contains(thisPair))
					continue;
				seenFiles.add(thisPair);
				
				String topic = topic1;
				HashSet<HashSet<String>> seen = new HashSet<HashSet<String>>();
				KafSaxParser parser1 = new KafSaxParser();
				KafSaxParser parser2 = new KafSaxParser();
				boolean augmented = true;
				parser1.parseFile(f1,augmented);
				if(!fileKey1.equals(fileKey2))
					parser2.parseFile(f2,augmented);
				
				// combine mentions in both files 
				LinkedList<GeneralTuple<String,String>> combinedSentNums = new LinkedList<GeneralTuple<String,String>>();
				for(String s_id : parser1.sentenceNumToMidSet.keySet())
					combinedSentNums.add(new GeneralTuple<String,String>(s_id, "1"));
				if(!fileKey1.equals(fileKey2)) {
					for(String s_id : parser2.sentenceNumToMidSet.keySet())
						combinedSentNums.add(new GeneralTuple<String,String>(s_id, "2"));
				}
				
				boolean across = true;
				boolean within = true;
				boolean onlyCounts = false;
				String delim = "|";
	
				// within sentence
	
				if(within) {
					for(GeneralTuple<String,String> s_id_and_file : combinedSentNums) {
						
						String s_id = s_id_and_file.first;
						KafSaxParser parser = s_id_and_file.second.equals("1") ? parser1 : parser2;
						String fileKey = s_id_and_file.second.equals("1") ? fileKey1 : fileKey2;
	
						// this sentence wasn't cleaned
						if(!cleanSentences.get(fileKey).contains(s_id))
							continue;
	
						String sentence = parser.getSentenceFromSentenceId(s_id);
						HashMap<String, HashMap<String, List<DEPNode>>> mIdToEvents = ComparerUtil.partitionEventsWithinSentence(sentence,parser.sentenceNumToMidSet.get(s_id), parser, posCounts, srl);
	
						// log the number of events in this sentence
	
						numEvents+=mIdToEvents.size();
	
	
						// compare all pairs of events
						if(!onlyCounts) {
							for(String m_id1 : mIdToEvents.keySet()) {
								for(String m_id2 : mIdToEvents.keySet()) {
	
									if(m_id1.equals(m_id2))
										continue;
	
									// don't duplicate
									if(seen.contains(Stream.of(fileKey,m_id1,m_id2).collect(Collectors.toCollection(HashSet::new))))
										continue;
									else
										seen.add(Stream.of(fileKey,m_id1,m_id2).collect(Collectors.toCollection(HashSet::new)));
	
									// if making indexing file, don't need to calculate train samples
									String sim = sv.calculateWDEventPairSimVector(m_id1,m_id2,parser,parser,mIdToEvents.get(m_id1),mIdToEvents.get(m_id2)) + "\n";

									if(Globals.TEST_TOPICS.contains(topic)) {
										
										assert sim.split(",").length - 1 == header.toString().split("@").length;
										// test
										test.append(sim);
										// index
										String ev_id1 = parser.goldMIdToEvIdMap.get(m_id1);
										String ev_id2 = parser.goldMIdToEvIdMap.get(m_id2);
										String indexKey = fileKey + delim + fileKey + delim + m_id1 + delim + m_id2;
										sim = indexKey + "," + sim;
										indexFile.append(sim);
										
									}
									else{
										assert sim.split(",").length == header.toString().split("@").length;
										train.append(sim);
									}
								}
	
								i += 1;
								if (i % 100 == 0) {
									System.out.println(i + " done for WDEC");
								}
	
							}
						}
					}
				}
	
				// across sentence
	
				if(across) {
					
					for(GeneralTuple<String,String> s_id_and_file1 : combinedSentNums) {
						for(GeneralTuple<String,String> s_id_and_file2 : combinedSentNums) {
							
							String s_id1 = s_id_and_file1.first;
							String s_id2 = s_id_and_file2.first;
							// could be within or across document
							KafSaxParser thisParser1 = s_id_and_file1.second.equals("1") ? parser1 : parser2;
							KafSaxParser thisParser2 = s_id_and_file2.second.equals("2") ? parser2 : parser1;
							String thisFileKey1 = s_id_and_file1.second.equals("1") ? fileKey1 : fileKey2;
							String thisFileKey2 = s_id_and_file2.second.equals("2") ? fileKey2 : fileKey1;
							
							// same file, same sentence (covered in within above)
							if(thisFileKey1.equals(thisFileKey2) && s_id1.equals(s_id2))
								continue;
	
							// one of the sentences wasn't cleaned
							if(!cleanSentences.get(thisFileKey1).contains(s_id1) || !cleanSentences.get(thisFileKey2).contains(s_id2))
								continue;
	
							String sentence1 = thisParser1.getSentenceFromSentenceId(s_id1);
							HashMap<String, HashMap<String, List<DEPNode>>> mIdToEvents1 = ComparerUtil.partitionEventsWithinSentence(sentence1,thisParser1.sentenceNumToMidSet.get(s_id1), thisParser1,posCounts,srl);
							String sentence2 = thisParser2.getSentenceFromSentenceId(s_id2);
							HashMap<String, HashMap<String, List<DEPNode>>> mIdToEvents2 = ComparerUtil.partitionEventsWithinSentence(sentence2,thisParser2.sentenceNumToMidSet.get(s_id2), thisParser2,posCounts, srl);
	
							if(!onlyCounts) {
								for(String m_id1 : mIdToEvents1.keySet()) {
									for(String m_id2 : mIdToEvents2.keySet()) {
	
										if(thisFileKey1.equals(thisFileKey2) && m_id1.equals(m_id2))
											continue;
	
										// don't duplicate
										if(seen.contains(Stream.of(thisFileKey1, thisFileKey2, m_id1,m_id2).collect(Collectors.toCollection(HashSet::new))))
											continue;
										else
											seen.add(Stream.of(thisFileKey1, thisFileKey2, m_id1,m_id2).collect(Collectors.toCollection(HashSet::new)));
	
										// if making indexing file, don't need to calculate train samples
										String sim = sv.calculateWDEventPairSimVector(m_id1,m_id2,thisParser1,thisParser2,mIdToEvents1.get(m_id1),mIdToEvents2.get(m_id2)) + "\n";
	
	
										if(Globals.TEST_TOPICS.contains(topic)) {
											assert sim.split(",").length == header.toString().split("@").length;
											// test
											test.append(sim);
											// index
											String ev_id1 = thisParser1.goldMIdToEvIdMap.get(m_id1);
											String ev_id2 = thisParser2.goldMIdToEvIdMap.get(m_id2);
											String indexKey = thisFileKey1 + delim + thisFileKey2 + delim + m_id1 + delim + m_id2;
											sim = indexKey + "," + sim;
											indexFile.append(sim);
											
											
											
										}
										else{
											assert sim.split(",").length == header.toString().split("@").length;
											train.append(sim);
										}
	
	
										i += 1;
										if (i % 100 == 0) {
											System.out.println(i + " done for WDEC");
										}
	
									}
	
								}
							}
						}
					}
				}
		}
		}

		posCounts = (HashMap<String, Integer>)ComparerUtil.sortByValue(posCounts);
		int total = 0;
		for(String pos : posCounts.keySet())
			total += posCounts.get(pos);
		for(String pos : posCounts.keySet()) {
			System.out.println(pos + " : " + posCounts.get(pos) + "(" + posCounts.get(pos)*1.0/total + ")");
		}
		System.out.println("% srl triggers: " + srl.intValue()*1.0/numEvents);

		System.out.println("WDEC done,  writing files");
		File f = null;
		try {
			FileWriter writer = null;

			//train
			f = new File(arffOutDir  + "/" + fileName + "_train.arff");
			System.out.println("writing to " + f);
			f.createNewFile();
			writer = new FileWriter(f);
			writer.write(train.toString());
			writer.flush();
			writer.close();
			
			// test
			f = new File(arffOutDir  + "/" + fileName + "_test.arff");
			System.out.println("writing to " + f);
			f.createNewFile();
			writer = new FileWriter(f);
			writer.write(test.toString());
			writer.flush();
			writer.close();
			
			// index
			f = new File(arffOutDir  + "/" + fileName + "_test_index.arff");
			System.out.println("writing to " + f);
			f.createNewFile();
			writer = new FileWriter(f);
			writer.write(indexFile.toString());
			writer.flush();
			writer.close();
			

		}

		catch(IOException e) {
			e.printStackTrace();
		}
		System.out.println("----> WDEC done");
		return f;
	}

	//	============== For CDEC Event Chain Pair Classifier ==================

	public File writeCDECArffFromGold(List<File> files, String mode, String arffOutDir, HashMap<String, HashSet<String>> cleanSentences, String fileName) {

		StringBuilder header = new StringBuilder();

		//header
		header.append("@relation ECB+Gold_CDEC\n\n");
		//1
		header.append("@attribute trigger_sim_vec numeric\n");
		//2
		header.append("@attribute A0_sim_vec numeric\n");
		//3
		header.append("@attribute A1_sim_vec numeric\n");
		//4
		header.append("@attribute TMP_sim_vec numeric\n");
		//5
		header.append("@attribute LOC_sim_vec numeric\n");
		//6
		header.append("@attribute aux_arg_sim_vec numeric\n");
		//7.1
		header.append("@attribute relative_sentence_start_position numeric\n");
		//7.1
		header.append("@attribute relative_sentence_end_position numeric\n");
		// 8 N/A for CDEC
		//9
		header.append("@attribute context_sts numeric\n");
		//10
		header.append("@attribute event_sim_strict numeric\n");
		//11
		header.append("@attribute event_sts numeric\n");
		//12
		header.append("@attribute trigger_sts numeric\n");
		//13
		header.append("@attribute trigger_sim_strict numeric\n");
		//14
		header.append("@attribute A0_sim_strict numeric\n");
		//15
		header.append("@attribute A1_sim_strict numeric\n");
		//16
		header.append("@attribute TMP_sim_strict numeric\n");
		//17
		header.append("@attribute LOC_sim_strict numeric\n");
		//18
		header.append("@attribute aux_arg_sim_strict numeric\n");
		//19
		header.append("@attribute lstm_trigger numeric\n");
		//20
		header.append("@attribute lstm_context numeric\n");

		//class
		header.append("@attribute coreferent {true,false}\n\n");
		header.append("@data\n");

		//train/test
		StringBuilder train = new StringBuilder(header);
		StringBuilder test = new StringBuilder(header);

		String delim = "|";

		int counter = 0;
		HashSet<HashSet<String>> seen = new HashSet<HashSet<String>>();
		for(File f1 : files) {
			for(File f2 : files) {
				
				// doing cdec
				if(f1.getName().equals(f2.getName()))
					continue;

				String file1Key = f1.getName().replace("_aug.en.naf", "");
				String file2Key =  f2.getName().replace("_aug.en.naf", "");

				// one of the files wasn't cleaned by ecb+ people
				if(!cleanSentences.containsKey(file1Key) || !cleanSentences.containsKey(file2Key))
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
				if(seen.contains(thisPair))
					continue;
				seen.add(thisPair);
				
				// recap conditions
				assertThat(f1.getName(), is(not(f2.getName())));
				assertThat(cleanSentences.get(file1Key),is(not((Object)null)));
				assertThat(cleanSentences.get(file2Key),is(not((Object)null)));
				assertThat(topic1,is(topic2));
				assertThat(subTopic1,is(subTopic2));

				KafSaxParser parser1 = new KafSaxParser();
				KafSaxParser parser2 = new KafSaxParser();

				boolean augmented = true;
				parser1.parseFile(f1,augmented);
				parser2.parseFile(f2,augmented);

				String topic = topic1;

				KafSaxParser f1Parser = new KafSaxParser();
				f1Parser.parseFile(f1,augmented);
				KafSaxParser f2Parser = new KafSaxParser();
				f2Parser.parseFile(f2,augmented);

				for(String ev_id1 : f1Parser.goldEvIdToMidToCorefChainTokenIDSet.keySet()) {
					for(String ev_id2 : f2Parser.goldEvIdToMidToCorefChainTokenIDSet.keySet()) {

						if(!ev_id1.contains("ACT") || !ev_id2.contains("ACT"))
							continue;
						
						// chain 1
						HashMap<String,LinkedList<String>> cleanEv1mIdToTokenIDList = new HashMap<String,LinkedList<String>>();
						for(String m_id : f1Parser.goldEvIdToMidToCorefChainTokenIDSet.get(ev_id1).keySet()) {
							LinkedList<String> tIdList = f1Parser.goldEvIdToMidToCorefChainTokenIDSet.get(ev_id1).get(m_id);

							KafAugToken anchorTok = f1Parser.tIdToAugTokenMap.get(tIdList.getFirst());
							if(cleanSentences.get(file1Key).contains(anchorTok.getSentenceNum())) {
								// make sure the anchor token for every chain has an mid (sanity check)
								assertThat(anchorTok.getMId(), is(not("")));
								// make sure all anchor token m_ids are unique across events in the chain
								assertThat((Object)cleanEv1mIdToTokenIDList.get(anchorTok.getMId()), is((Object)null));
								// store pair
								cleanEv1mIdToTokenIDList.put(anchorTok.getMId(), tIdList);
							}
						}

						// chain 2
						HashMap<String,LinkedList<String>> cleanEv2mIdToTokenIDList = new HashMap<String,LinkedList<String>>();
						for(String m_id : f2Parser.goldEvIdToMidToCorefChainTokenIDSet.get(ev_id2).keySet()) {
							LinkedList<String> tIdList = f2Parser.goldEvIdToMidToCorefChainTokenIDSet.get(ev_id2).get(m_id);

							KafAugToken anchorTok = f2Parser.tIdToAugTokenMap.get(tIdList.getFirst());
							if(cleanSentences.get(file1Key).contains(anchorTok.getSentenceNum())) {
								// make sure the anchor token for every chain has an mid (sanity check)
								assertThat(anchorTok.getMId(), is(not("")));
								// make sure all anchor token m_ids are unique across events in the chain
								assertThat((Object)cleanEv2mIdToTokenIDList.get(anchorTok.getMId()), is((Object)null));
								// store pair
								cleanEv2mIdToTokenIDList.put(anchorTok.getMId(), tIdList);
							}
						}

						// one of ev_id1 or ev_id2 has no mentions in a cleaned sentence
						if(cleanEv1mIdToTokenIDList.keySet().isEmpty() || cleanEv2mIdToTokenIDList.keySet().isEmpty())
							continue;
						
						// recap conditions
						assertThat(ev_id1.contains("ACT"),is(true));
						assertThat(ev_id2.contains("ACT"),is(true));
						assertThat(cleanEv1mIdToTokenIDList.keySet(),is(not((Object)null)));
						assertThat(cleanEv2mIdToTokenIDList.keySet(),is(not((Object)null)));
						
						EvChainSimVector sv = new EvChainSimVector();
						String sim = sv.calculateCDEventChainSimVector(cleanEv1mIdToTokenIDList,
																	   cleanEv2mIdToTokenIDList,
																	   f1Parser,
																	   f2Parser,
																	   ev_id1,
																	   ev_id2) + "\n";
						assert sim.split(",").length - 1 == header.toString().split("@").length;
					
						
						if(Globals.TEST_TOPICS.contains(topic))
							test.append(sim);
						else
							train.append(sim);

						counter += 1;
						if (counter % 100 == 0) {
							System.out.println(counter + " done for CDEC");
						}
					}
				}

			}
		}
		File f = null;
		try {
			FileWriter writer = null;
			/*
			 * train
			 */

			f = new File(arffOutDir  + "/" + fileName + "_train.arff");
			System.out.println("writing " + f.getPath());
			f.createNewFile();
			writer = new FileWriter(f);
			writer.write(train.toString());
			writer.flush();
			writer.close();



			/*
			 * test
			 */

			f = new File(arffOutDir  + "/" + fileName + "_test.arff");
			System.out.println("writing " + f.getPath());
			f.createNewFile();
			writer = new FileWriter(f);
			writer.write(test.toString());
			writer.flush();
			writer.close();


		}
		catch(IOException e) {
			e.printStackTrace();
		}
		System.out.println("----------> CDEC done");
		return f;
	}

	public File writeCDECIndexFile(WDEC wdec,String arffOutDir,String fileName, boolean clusterBySubTopic) {
		
		StringBuilder test = new StringBuilder();
		String delim = "|";
		HashSet<String> keys = new HashSet<String>();
		int counter = 0;
		HashSet<HashSet<String>> comparedFiles = new HashSet<HashSet<String>>();
		HashSet<HashSet<String>> wdChainsCompared = new HashSet<HashSet<String>>();
		for(File f1 : wdec.evaluatedFiles) {
			for(File f2 : wdec.evaluatedFiles) {
				
				if(f1.getName().equals(f2.getName()))
					continue;
				
				String file1Key = f1.getName().replace("_aug.en.naf", "");
				String file2Key =  f2.getName().replace("_aug.en.naf", "");
				
				// only look at each pair once
				HashSet<String> thisPair = Stream.of(file1Key, file2Key).collect(Collectors.toCollection(HashSet::new));
				if(comparedFiles.contains(thisPair))
					continue;
				comparedFiles.add(thisPair);
				
				// topics are numbers, subtopics are ecbplus or ecb
				String topic1 = f1.getName().split("_")[0];
				String subTopic1 = f1.getName().split("_")[1].replaceAll("[0-9]", "");
				String topic2 = f2.getName().split("_")[0];
				String subTopic2 = f2.getName().split("_")[1].replaceAll("[0-9]", "");
				
				if(!topic1.equals(topic2))
					continue;
				
				if(clusterBySubTopic) {
					if(!subTopic1.equals(subTopic2))
						continue;
				}
				
				boolean augmented = true;
				KafSaxParser f1Parser = new KafSaxParser();
				KafSaxParser f2Parser = new KafSaxParser();
				f1Parser.parseFile(f1,augmented);
				f2Parser.parseFile(f2,augmented);
				
				/*
				 * 1: compare every chain in f1 with every chain in f2
				 */
				// the chains here are only generated from cleaned sentences in wdec, so don't need to check that
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
						
						// wd chain ids are unique
						assertThat(wdChain1_id,is(not(wdChain2_id)));
						
						// get <m_id, tokens> for every event in chain 2
						HashMap<String, LinkedList<String>> chain2mIdToTokenIDList = new HashMap<String,LinkedList<String>>();
						for(String m_id : wdec.WD_idToM_idSet.get(wdChain2_id)) {
							assertThat(chain2mIdToTokenIDList.get(m_id),is((Object)null));
							
							chain2mIdToTokenIDList.put(m_id, new LinkedList<String>());
							chain2mIdToTokenIDList.get(m_id).addAll(f2Parser.goldMIdToTokenSetMap.get(m_id));
						}
						
						// generate vector
						String indexKey = wdChain1_id + delim + wdChain2_id;
						
						assertThat(keys.contains(indexKey),is(false));
						keys.add(indexKey);
						EvChainSimVector sv = new EvChainSimVector();
						String sim = sv.calculateCDEventChainSimVector(chain1mIdToTokenIDList,
																	   chain2mIdToTokenIDList,
																	   f1Parser,
																	   f2Parser,
																	   "n_a",
																	   "n_a") + "\n";
						sim = indexKey + "," + sim;
						test.append(sim);

						
						counter += 1;
						if (counter % 100 == 0) {
							System.out.println(counter + " done for CDEC Index");
						}
					}
				}
			}
		}
		File f = null;
		try {
			FileWriter writer = null;
			f = new File(arffOutDir  + "/" + fileName + "_test.arff");
			System.out.println("writing " + f.getPath());
			f.createNewFile();
			writer = new FileWriter(f);
			writer.write(test.toString());
			writer.flush();
			writer.close();


		}
		catch(IOException e) {
			e.printStackTrace();
		}
		System.out.println("----------> CDEC Index done");
		return f;
	}
}
