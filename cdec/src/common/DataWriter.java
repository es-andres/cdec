package common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import comparer.ComparerUtil;
import edu.emory.clir.clearnlp.dependency.DEPNode;
import eu.kyotoproject.kaf.KafAugToken;
import eu.kyotoproject.kaf.KafSaxParser;
import static org.junit.Assert.*;
import static org.hamcrest.CoreMatchers.*;

public class DataWriter {

	public static HashMap<String, HashSet<String>> cleanSentenceDict(File cleanTable) {
		HashMap<String, HashSet<String>> records = new HashMap<String,HashSet<String>>();

		try (BufferedReader br = new BufferedReader(new FileReader(cleanTable))) {
			String line;
			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				String fileName = values[0] + "_" + values[1];
				String sentenceNum = values[2].replaceAll(" ", "");
				if(!records.containsKey(fileName))
					records.put(fileName, new HashSet<String>());
				records.get(fileName).add(sentenceNum);
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}

		return records;
	}

	public static File writeCDEventDataset(List<File> files, String arffOutDir, String fileName, HashMap<String, HashSet<String>> cleanSentences, boolean lemma) {
		// train/test
		String delim = "|";
		String header = String.format("s1%ss2%slabel%s",delim,delim,"\n");
		StringBuilder train_trigger = new StringBuilder(header);
		StringBuilder test_trigger = new StringBuilder(header);
		StringBuilder train_context = new StringBuilder(header);
		StringBuilder test_context = new StringBuilder(header);


		int counter = 0;
		
		HashSet<HashSet<String>> seen = new HashSet<HashSet<String>>();
		for(File f1 : files) {
			for(File f2 : files) {
				
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
						
						HashSet<String> thisPair = Stream.of(file1Key, file2Key, ev_id1,ev_id2).collect(Collectors.toCollection(HashSet::new));
						if(seen.contains(thisPair))
							continue;
						seen.add(thisPair);
						
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
						
						//  chain 2
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

						HashMap<String, HashMap<String, List<DEPNode>>> chain1 = getParsedChain(cleanEv1mIdToTokenIDList,
																							    f1Parser);
						HashMap<String, HashMap<String, List<DEPNode>>> chain2 = getParsedChain(cleanEv2mIdToTokenIDList,
																							    f2Parser);

						// event triggers
						String chain1Trigger = DataWriter.getEvTrigger(chain1);
						String chain2Trigger = DataWriter.getEvTrigger(chain2);
						// event contexts
						String chain1Context = DataWriter.getEvChainContext(parser1, chain1.keySet());
						String chain2Context = DataWriter.getEvChainContext(parser2, chain2.keySet());

						//label
						String coreferent = (ev_id1.equals(ev_id2)) ? "1" : "0"; 

						// make rows
						String trigger_row = chain1Trigger  + delim + chain2Trigger + delim + coreferent + "\n";
						String context_row = chain1Context + delim + chain2Context + delim + coreferent + "\n";

						if(Globals.TEST_TOPICS.contains(topic)) {
							test_trigger.append(trigger_row);
							test_context.append(context_row);
						}

						else {
							train_trigger.append(trigger_row);
							train_context.append(context_row);
						}

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
			// trigger
			f = new File(arffOutDir  + "/" + fileName + "_trigger-train.arff");
			System.out.println("Writing to " + f.getAbsolutePath());
			f.createNewFile();
			writer = new FileWriter(f);
			writer.write(train_trigger.toString());
			writer.flush();
			writer.close();
			// surface
			f = new File(arffOutDir  + "/" + fileName + "_context-train.arff");
			System.out.println("Writing to " + f.getAbsolutePath());
			f.createNewFile();
			writer = new FileWriter(f);
			writer.write(train_context.toString());
			writer.flush();
			writer.close();

			/*
			 * test
			 */
			// trigger
			f = new File(arffOutDir  + "/" + fileName + "_trigger-test.arff");
			System.out.println("Writing to " + f.getAbsolutePath());
			f.createNewFile();
			writer = new FileWriter(f);
			writer.write(test_trigger.toString());
			writer.flush();
			writer.close();
			//surface
			f = new File(arffOutDir  + "/" + fileName + "_context-test.arff");
			System.out.println("Writing to " + f.getAbsolutePath());
			f.createNewFile();
			writer = new FileWriter(f);
			writer.write(test_context.toString());
			writer.flush();
			writer.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		System.out.println("----------> CDEC done");
		return f;
	}
	
	/*
	 * Write a dataset to train the lstm 
	 */
	public static File writeWDEventDataset(List<File> files, String arffOutDir, String fileName, HashMap<String, HashSet<String>> cleanSentences, boolean lemma) {

		// train/test
		String delim = "|";
		String header = String.format("s1%ss2%slabel%s",delim,delim,"\n");
		StringBuilder train_trigger = new StringBuilder(header);
		StringBuilder test_trigger = new StringBuilder(header);
		StringBuilder train_context = new StringBuilder(header);
		StringBuilder test_context = new StringBuilder(header);


		int i = 0;
		HashMap<String, Integer> posCounts = new HashMap<String, Integer>();
		AtomicInteger srl = new AtomicInteger(0);
		for(File f : files) {
			String fileKey = f.getName().replace("_aug.en.naf", "");

			// file wasn't cleaned by ecb+ people
			if(!cleanSentences.containsKey(fileKey))
				continue;

			String topic = f.getName().split("_")[0];
			HashSet<HashSet<String>> seen = new HashSet<HashSet<String>>();
			KafSaxParser parser = new KafSaxParser();
			boolean augmented = true;
			parser.parseFile(f,augmented);

			// switches
			boolean across = true;
			boolean within = true;
			boolean onlyCounts = false;

			// within sentence
			if(within) {

				for(String s_id : parser.sentenceNumToMidSet.keySet()) {

					// this sentence wasn't cleaned
					if(!cleanSentences.get(fileKey).contains(s_id))
						continue;

					String sentence = parser.getSentenceFromSentenceId(s_id);
					HashMap<String, HashMap<String, List<DEPNode>>> mIdToEvents = ComparerUtil.partitionEventsWithinSentence(sentence,parser.sentenceNumToMidSet.get(s_id), parser, posCounts, srl);


					// compare all pairs of events
					if(!onlyCounts) {
						for(String m_id1 : mIdToEvents.keySet()) {
							for(String m_id2 : mIdToEvents.keySet()) {
								if(!m_id1.equals(m_id2) && seen.add(Stream.of(m_id1,m_id2).collect(Collectors.toCollection(HashSet::new)))) {

									HashMap<String, List<DEPNode>> e1 = mIdToEvents.get(m_id1);
									HashMap<String, List<DEPNode>> e2 = mIdToEvents.get(m_id2);

									// event triggers
									String ev1 = DataWriter.getEvTrigger(e1, lemma);
									String ev2 = DataWriter.getEvTrigger(e2,lemma);
									// event contexts
									String c1 = DataWriter.getEvContext(parser, m_id1, lemma);
									String c2 = DataWriter.getEvContext(parser, m_id2, lemma);

									//label
									String t_id1 = parser.goldMIdToTokenSetMap.get(m_id1).getFirst();
									String ev1_id = parser.tIdToAugTokenMap.get(t_id1).getEvId();
									String t_id2 = parser.goldMIdToTokenSetMap.get(m_id2).getFirst();
									String ev2_id = parser.tIdToAugTokenMap.get(t_id2).getEvId();
									String coreferent = (ev1_id.equals(ev2_id)) ? "1" : "0"; 

									// make rows
									String trigger_row = ev1  + delim + ev2 + delim + coreferent + "\n";
									String context_row = c1 + delim + c2 + delim + coreferent + "\n";


									// add to appropriate datasets 
									if(Globals.SPLIT_ARFF) {

										if(Globals.TEST_TOPICS.contains(topic)) {
											test_trigger.append(trigger_row);
											test_context.append(context_row);
										}
										else {
											train_trigger.append(trigger_row);
											train_context.append(context_row);
										}
									};

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

			// across sentence

			if(across) {
				for(String s_id1 : parser.sentenceNumToMidSet.keySet()) {
					for(String s_id2 : parser.sentenceNumToMidSet.keySet()) {

						if(s_id1.equals(s_id2))
							continue;

						// sentence wasn't cleaned
						if(!cleanSentences.get(fileKey).contains(s_id1) || !cleanSentences.get(fileKey).contains(s_id2))
							continue;
						//						
						String sentence1 = parser.getSentenceFromSentenceId(s_id1);
						HashMap<String, HashMap<String, List<DEPNode>>> mIdToEvents1 = ComparerUtil.partitionEventsWithinSentence(sentence1,parser.sentenceNumToMidSet.get(s_id1), parser,posCounts,srl);
						String sentence2 = parser.getSentenceFromSentenceId(s_id2);
						HashMap<String, HashMap<String, List<DEPNode>>> mIdToEvents2 = ComparerUtil.partitionEventsWithinSentence(sentence2,parser.sentenceNumToMidSet.get(s_id2), parser,posCounts, srl);
						//						
						if(!onlyCounts) {
							for(String m_id1 : mIdToEvents1.keySet()) {
								for(String m_id2 : mIdToEvents2.keySet()) {
									if(!m_id1.equals(m_id2) && seen.add(Stream.of(m_id1,m_id2).collect(Collectors.toCollection(HashSet::new)))) {


										HashMap<String, List<DEPNode>> e1 = mIdToEvents1.get(m_id1);
										HashMap<String, List<DEPNode>> e2 = mIdToEvents2.get(m_id2);

										// event triggers
										String ev1 = DataWriter.getEvTrigger(e1, lemma);
										String ev2 = DataWriter.getEvTrigger(e2,lemma);
										// event contexts
										String c1 = DataWriter.getEvContext(parser, m_id1, lemma);
										String c2 = DataWriter.getEvContext(parser, m_id2, lemma);

										//label
										String t_id1 = parser.goldMIdToTokenSetMap.get(m_id1).getFirst();
										String ev1_id = parser.tIdToAugTokenMap.get(t_id1).getEvId();
										String t_id2 = parser.goldMIdToTokenSetMap.get(m_id2).getFirst();
										String ev2_id = parser.tIdToAugTokenMap.get(t_id2).getEvId();
										String coreferent = (ev1_id.equals(ev2_id)) ? "1" : "0"; 

										// make rows;
										String trigger_row = ev1  + delim + ev2 + delim + coreferent + "\n";
										String context_row = c1 + delim + c2 + delim + coreferent + "\n";

										// add to appropriate datasets 
										if(Globals.SPLIT_ARFF) {

											if(Globals.TEST_TOPICS.contains(topic)) {
												test_trigger.append(trigger_row);
												test_context.append(context_row);
											}
											else {
												train_trigger.append(trigger_row);
												train_context.append(context_row);
											}
										};

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

		// remove last \n and add closing bracket
		// trigger
		test_trigger.setLength(test_trigger.length() - 1);
		train_trigger.setLength(train_trigger.length() - 1);
		// context
		test_context.setLength(test_context.length() - 1);
		train_context.setLength(train_context.length() - 1);

		System.out.println("WDEC done,  writing files");
		File f = null;
		String slstmDataDir = "/home/acrem003/Documents/Projects/class_slstm/data";
		try {
			FileWriter writer = null;
			if (Globals.SPLIT_ARFF) {

				/*
				 * train
				 */
				// trigger
				f = new File(slstmDataDir  + "/" + fileName + "_trigger_train.txt");
				System.out.println("writing " + f);
				f.createNewFile();
				writer = new FileWriter(f);
				writer.write(train_trigger.toString());
				writer.flush();
				writer.close();
				// context
				f = new File(slstmDataDir  + "/" + fileName + "_context_train.txt");
				System.out.println("writing " + f);
				f.createNewFile();
				writer = new FileWriter(f);
				writer.write(train_context.toString());
				writer.flush();
				writer.close();

				/*
				 * test
				 */

				// trigger
				f = new File(slstmDataDir  + "/" + fileName + "_trigger_test.txt");
				System.out.println("writing " + f);
				f.createNewFile();
				writer = new FileWriter(f);
				writer.write(test_trigger.toString());
				writer.flush();
				writer.close();

				// context
				f = new File(slstmDataDir  + "/" + fileName + "_context_test.txt");
				System.out.println("writing " + f);
				f.createNewFile();
				writer = new FileWriter(f);
				writer.write(test_context.toString());
				writer.flush();
				writer.close();

			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		System.out.println("----> WDEC done");
		return f;
	}

	public static HashMap<String, HashMap<String, List<DEPNode>>> getParsedChain(HashMap<String,LinkedList<String>> chainMIdToTokenIDList,
																				 KafSaxParser parser) {


		HashSet<String> seenSentences = new HashSet<String>();
		// m_id, <role, nodes>
		HashMap<String, HashMap<String, List<DEPNode>>> mIdToParsedEv = new HashMap<String, HashMap<String, List<DEPNode>>> ();

		// add all tokens across all roles occurring across all sentences in this event chain
		// chain
		// do this for all events (each event given by a m_id)
		for (String m_id : chainMIdToTokenIDList.keySet()) {

			String anchor_t_id = chainMIdToTokenIDList.get(m_id).getFirst();
			KafAugToken anchorTok = parser.tIdToAugTokenMap.get(anchor_t_id);
			String sentence = parser.getSentenceFromSentenceId(anchorTok.getSentenceNum());
			// only process each sentence once
			if (seenSentences.add(anchorTok.getSentenceNum())) {

				HashMap<String, Integer> dummyPosCounts = new HashMap<String, Integer>();
				AtomicInteger dummySrlCount = new AtomicInteger();
				HashMap<String, HashMap<String, List<DEPNode>>> mIdToEvents = ComparerUtil.partitionEventsWithinSentence(sentence,
																				parser.sentenceNumToMidSet.get(anchorTok.getSentenceNum()), 
																			  parser, 
																			  dummyPosCounts,
																			  dummySrlCount);

				for(String parsed_m_id : mIdToEvents.keySet()) {
					if(chainMIdToTokenIDList.containsKey(parsed_m_id))
						mIdToParsedEv.put(parsed_m_id, mIdToEvents.get(parsed_m_id));
				}

			}
		}

		return mIdToParsedEv;
	}

	public static String getEvChainContext(KafSaxParser parser, Set<String> chain_m_id_list) {
		
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

	public static String getEvContext(KafSaxParser parser, String m_id, boolean lemma) {

		String t_id = parser.goldMIdToTokenSetMap.get(m_id).getFirst();
		String s_id = parser.tIdToAugTokenMap.get(t_id).getSentenceNum();
		List<KafAugToken> sentenceList = parser.sentenceNumToAugTokens.get(s_id);

		String sentence = "";
		for(KafAugToken t : sentenceList) {
			if(!t.getEvType().contains("ACT") || t.getMId().equals(m_id)) {
				if(lemma) 
					sentence += t.getLemma() + " ";
				else
					sentence += t.getTokenString() + " ";
			}
		}
		sentence = sentence.substring(0, sentence.length() -1);

		return sentence;
	}
	
	public static String getEvTrigger(HashMap<String, HashMap<String, List<DEPNode>>> chain) {
		
		String evTrigger = "";
		
		for(String m_id : chain.keySet()) {
			// fill triggers (they come in order)
			for(DEPNode n : chain.get(m_id).get("trigger")) {
				evTrigger += n.getWordForm() + " ";
			}
		}

		// remove trailing whitespace
		evTrigger = evTrigger.substring(0,evTrigger.length()-1);

		return evTrigger;
	}
	public static String getEvTrigger(HashMap<String, List<DEPNode>> ev, boolean lemma) {
		String evTrigger = "";

		// fill triggers (they come in order)
		for(DEPNode n : ev.get("trigger")) {
			if(lemma)
				evTrigger += n.getLemma() + " ";
			else
				evTrigger += n.getWordForm() + " ";
		}

		// remove trailing whitespace
		evTrigger = evTrigger.substring(0,evTrigger.length()-1);

		return evTrigger;
	}
	public static String getEvText(HashMap<String, List<DEPNode>> ev, boolean lemma) {
		String evText = "";

		TreeMap<Integer, String> textOrder = new TreeMap<Integer, String>();
		// fill text (first get in order)
		for(String role : ev.keySet()) {
			for(DEPNode n : ev.get(role)) {
				if(lemma)
					textOrder.put(n.getID(), n.getLemma());
				else
					textOrder.put(n.getID(), n.getWordForm());
			}
		}
		for(String w : textOrder.values()) {
			evText += w + " ";
		}

		// remove trailing whitespace
		evText = evText.substring(0,evText.length()-1);

		return evText;
	}
}
