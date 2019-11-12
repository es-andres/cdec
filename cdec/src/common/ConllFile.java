package common;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThat;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import clustering.EquivCluster;
import coref.CDEC;
import coref.WDEC;
import eu.kyotoproject.kaf.KafAugToken;
import eu.kyotoproject.kaf.KafSaxParser;
import naf.EventNode;
import naf.NafDoc;

public class ConllFile {
	static final Logger LOGGER = Logger.getLogger(ConllFile.class.getName());
	
	public ConllFile() {
		
	}
	public static File makeConllFile(ECBWrapper dataWrapper, ArrayList<HashSet<EventNode>> evClusters, List<File> files, String fName) {
		
		HashMap<String, String> clusterIdMap = new HashMap<String, String>();
		for(int i = 0; i < evClusters.size(); i++) {
			for(EventNode n : evClusters.get(i)) {
				if(fName.contains("gold")) {
					clusterIdMap.put(n.getGlobalKey(), n.ev_id.replace("ACT", ""));
				}
				else if (fName.contains("model")) {
					clusterIdMap.put(n.getGlobalKey(), String.valueOf(i));//String.valueOf(i)
				}
				else
					System.exit(-1);
			}
		}
		
		StringBuilder conllDoc = new StringBuilder("#begin document (cdec);\n");
		int extra = 0;
		for(int i =0; i < files.size(); i++) {
			File f = files.get(i);
			NafDoc doc = dataWrapper.docs.get(f.getName());

			for(String t_id : doc.inOrderToks) {
				HashMap<String, String> tok = doc.toks.get(t_id);
				String m_id = tok.get("m_id");
				String ev_id = tok.get("ev_id");
				String label = "-";

				if(Globals.cleanSentences.get(ECBWrapper.cleanFileName(f)).contains(tok.get("sentence"))){
					if(ev_id.contains("ACT")) {
						EventNode n = new EventNode(f, m_id, ev_id);
						String clustId = "-";
						if(!clusterIdMap.containsKey(n.getGlobalKey())) {
//							System.out.println(f.getName() + " " + m_id + " " + ev_id);
						}
						else {
							clustId = clusterIdMap.get(n.getGlobalKey());
						}
							
						int spanSize = doc.mentionIdToTokens.get(m_id).size();
						if(spanSize == 1)
							label = "(" + clustId + ")";
						else {
							if(t_id.equals(doc.mentionIdToTokens.get(m_id).get(0)))
								label = "(" + clustId;
							else if(t_id.equals(doc.mentionIdToTokens.get(m_id).get(spanSize - 1)))
								label = clustId + ")";
							else
								label = clustId;
						}
						
					}
				}
				//40_4ecb	0	6	update	-	-	-	-	-	-	-	-	(18091688366575792)
				String fKey = f.getName().replace("_aug.en.naf", "");
				String word = tok.get("text");
				String line = fKey + "\t" + "0" + "\t" + t_id + "\t" + word + "\t" + label;
				conllDoc.append(line + "\n");
			}
			conllDoc.append("\n");
		}
		conllDoc.append("#end document");
		File f = null;
		try {
			f = Paths.get(Globals.CONLL_DIR.toString(), fName).toFile();
			f.createNewFile();
			FileWriter writer = new FileWriter(f);
			writer.write(conllDoc.toString());
			writer.flush();
			writer.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
		LOGGER.info("Conll file written to " + f);
		return f;
		
	}
	
	public TreeMap<Integer, HashMap<String, LinkedList<File>>> getFileClusters(List<File> allFiles) {
		
		TreeMap<Integer, HashMap<String, LinkedList<File>>> clusters = new TreeMap<Integer, HashMap<String,LinkedList<File>>>();
		
		for(File f : allFiles) {
			int topic = Integer.parseInt(f.getName().split("_")[0]);
			String subTopic = f.getName().split("_")[1].replaceAll("[0-9]", "");
			
			if(!Globals.TEST_TOPICS.contains(String.valueOf(topic)))
				continue;
			if(!clusters.containsKey(topic))
				clusters.put(topic, new HashMap<String, LinkedList<File>>());
			if(!clusters.get(topic).containsKey(subTopic))
				clusters.get(topic).put(subTopic, new LinkedList<File>());
			clusters.get(topic).get(subTopic).add(f);
		}
		
		return clusters;
	}
	//========================================== CDEC ==========================================
	//==========================================================================================
	//==========================================================================================
	
	//########################################## Gold #########################################
	
	public File writeCDConllFromGold(List<File> files, String writeDir,HashMap<String, HashSet<String>> cleanSentences) {
		//1 (path)
		String document_id = "";
		//2 (index in cluster)
		String part_num = "";
		//3 (t_id)
		String word_num = "";
		//4 (surface form)
		String word_itself = "";
		//5 null
		String pos = "-";
		//6 null
		String parse_bit = "-";
		//7 null
		String predicate_lemma = "-";
		//8 null
		String predicate_frameset_id = "-";
		//9 null
		String word_sense = "-";
		//10 null
		String speaker = "-";
		//11 null
		String named_entities = "-";
		//12 null
		String predicate_args = "-";
		//13 (chain_id)
		String coreference = "";
		StringBuilder rows = new StringBuilder();
		int i = 0;
		
		rows.append("#begin document (CDEC_ECB+); part 0 \n\n");
		for(File f : files) {
			
			String topic = f.getName().split("_")[0];
			// file isn't test, continue
			if(!Globals.TEST_TOPICS.contains(topic))
				continue;
			String fileKey = f.getName().replace("_aug.en.naf", "");
			
			// file wasn't cleaned by ecb+ people
			if(!cleanSentences.containsKey(fileKey))
				continue;
			
			Document doc = null;
			try {
				DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
				DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
				doc = dBuilder.parse(f);
			}
			catch(Exception e) {
				e.printStackTrace();
			}
			
			
			//7_3ecbplus_aug.en.naf
			String generalFileName = f.getName().replace("_aug.en.naf","");
			
			//1
			document_id = generalFileName;
			//2
			part_num = Integer.toString(i++);
			
			NodeList augTokens = doc.getElementsByTagName("Augmented_Tokens").item(0).getChildNodes();
			KafSaxParser parser = new KafSaxParser();
			boolean augmented = true;
			parser.parseFile(f,augmented);
			//<m_id, tokens>
			HashMap<String,LinkedList<String>> actions = parser.getGoldActions();
			
			String curSentence = "0";
			int sentIndex = 0;
			for(int j = 0; j < augTokens.getLength(); j++) {
				if(augTokens.item(j).getNodeType() != Node.ELEMENT_NODE)
					continue;
				Element augToken = (Element)augTokens.item(j);
				
				if(!cleanSentences.get(fileKey).contains(augToken.getAttribute("sentence")))
					continue;
				
				if(!augToken.getAttribute("sentence").equals(curSentence)) {
					sentIndex = 0;
					curSentence = augToken.getAttribute("sentence");
					rows.append("\n");
				}
				//3
				word_num = augToken.getAttribute("t_id");
				//if want sentence indexes to start at 0 for every sentence
//						word_num = Integer.toString(sentIndex++);
				//4
				word_itself = augToken.getTextContent();
				//coreference
				if(augToken.getAttribute("ev_type").equals("") || !augToken.getAttribute("ev_type").contains("ACT")) {
					coreference = "-";
				}
				else {
					String curChain = augToken.getAttribute("ev_id");
					//getFirst() should just be a string (ie. make token:chain -> 1:1:), will deal with this later
					coreference = augToken.getAttribute("ev_id").replace("ACT", "");
					if(augToken.getAttribute("ev_id").contains("NEG"))
						coreference = augToken.getAttribute("ev_id").replace("NEG","");
					//cases:
					//first or middle: (no token before or token before corefers)
					if(j-2 > 0 && (!((Element)augTokens.item(j-2)).getAttribute("ev_id").equals("")
							|| ((Element)augTokens.item(j-2)).getAttribute("ev_id").equals(curChain))) {
						//middle (token before corefers)
						if(((Element)augTokens.item(j-2)).getAttribute("ev_id").equals(curChain)) {
							//do nothing
						}
						//first (token before doesn't corefer)
						else
							coreference = "(" + coreference;	
					}
					//first (first token in chain)
					else
						coreference = "("  +  coreference;
					//last or middle: (no token after or token after corefers)
					if(j + 2 < augTokens.getLength() && (!((Element)augTokens.item(j+2)).getAttribute("ev_id").equals("")
							|| ((Element)augTokens.item(j+2)).getAttribute("ev_id").equals(curChain))) {
						if(((Element)augTokens.item(j+2)).getAttribute("ev_id").equals(curChain)) {
							//do nothing
						}
						//last (token after doesn't corefer)
						else
							coreference = coreference + ")";	
					}
					//last (last token in chain)
					else
						coreference = coreference + ")";
				}
				String delim = "\t";

				rows.append(
						//1
						document_id + delim + 
						//2
						part_num + delim + 
						//3
						word_num + delim +
						//4	
						word_itself + delim +
						//5	
						pos + delim + 
						//6	
						parse_bit + delim +
						//7	
						predicate_lemma + delim +
						//8	
						predicate_frameset_id + delim +
						//9	
						word_sense + delim +
						//10	
						speaker + delim +
						//11	
						named_entities + delim +
						//12	
						predicate_args + delim +
						//13	
						coreference + "\n");
			}
			rows.append("\n");
//					rows.append("#end document");
//					rows.append("\n");
		}
		rows.append("#end document");
//		rows.setLength(rows.length() - 1);
		//append string buffer/builder to buffered writer
		File f = null;
		try {
			f = new File(writeDir + "/" + "CDEC_gold.conll");
			f.createNewFile();
			FileWriter writer = new FileWriter(f);
			writer.write(rows.toString());
			writer.flush();
			writer.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
		return f;
	}
	
	//######################################## Model #############################################
	
	public File writeCDConllFromPreds(WDEC wdec, CDEC cdec, String writeDir, HashMap<String, HashSet<String>> cleanSentences) {
		
		//1 (path)
		String document_id = "";
		//2 (index in cluster)
		String part_num = "";
		//3 (t_id)
		String word_num = "";
		//4 (surface form)
		String word_itself = "";
		//5 null
		String pos = "-";
		//6 null
		String parse_bit = "-";
		//7 null
		String predicate_lemma = "-";
		//8 null
		String predicate_frameset_id = "-";
		//9 null
		String word_sense = "-";
		//10 null
		String speaker = "-";
		//11 null
		String named_entities = "-";
		//12 null
		String predicate_args = "-";
		//13 (chain_id)
		String coreference = "";
		StringBuilder rows = new StringBuilder();
		int i = 0;
		rows.append("#begin document (CDEC_ECB+); part 0 \n\n");
		for(File f : wdec.evaluatedFiles) {
			
			String topic = f.getName().split("_")[0];
			// file isn't test, continue
			if(!Globals.TEST_TOPICS.contains(topic))
				continue;
			String fileKey = f.getName().replace("_aug.en.naf", "");
			
			// file wasn't cleaned by ecb+ people
			if(!cleanSentences.containsKey(fileKey))
				continue;
			
			HashMap<String,String> mIdToWD_id = wdec.fileToM_idToWD_id.get(f);
			
			//7_3ecbplus_aug.en.naf
			String generalFileName = f.getName().replace("_aug.en.naf","");
			
			//1
			document_id = generalFileName;
			//2
			part_num = Integer.toString(i++);
			
			KafSaxParser parser = new KafSaxParser();
			boolean augmented = true;
			parser.parseFile(f,augmented);

			String curSentence = "0";
			for(int j = 0; j < parser.kafAugTokensArrayList.size(); j++) {

				KafAugToken augToken = parser.kafAugTokensArrayList.get(j);
				
				if(!cleanSentences.get(fileKey).contains(augToken.getSentenceNum()))
					continue;
				
				if(!augToken.getSentenceNum().equals(curSentence)) {
					curSentence = augToken.getSentenceNum();
					rows.append("\n");
				}
				//3
				word_num = augToken.getTid().replace("t","");
				//if want sentence indexes to start at 0 for every sentence
//						word_num = Integer.toString(sentIndex++);
				//4
				word_itself = augToken.getTokenString();

				//coreference
				if(!mIdToWD_id.containsKey(augToken.getMId())) {
					coreference = "-";
				}
				else {
					String curChain = cdec.wd_idTocd_id.get(mIdToWD_id.get(augToken.getMId()));

					coreference = cdec.wd_idTocd_id.get(mIdToWD_id.get(augToken.getMId()));

					//cases:
					
					boolean containsKeyForPrevToken = j-1 > 0 && mIdToWD_id.containsKey(parser.kafAugTokensArrayList.get(j-1).getMId());
					//first or middle: (no token before or token before corefers)
					String chain = "";
					if (containsKeyForPrevToken) {
						chain = mIdToWD_id.get(parser.kafAugTokensArrayList.get(j-1).getMId());
					}
					if(j-1 > 0 && (!containsKeyForPrevToken || chain.equals(curChain))) {
						//middle (token before corefers)
						if(containsKeyForPrevToken && mIdToWD_id.get(parser.kafAugTokensArrayList.get(j-1).getMId()).equals(curChain)) {
							//do nothing
						}
						//first (token before doesn't corefer)
						else
							coreference = "(" + coreference;	
					}
					//first (first token in chain)
					else
						coreference = "("  +  coreference;
					boolean containsKeyForNextToken = j + 1 < parser.kafAugTokensArrayList.size() && mIdToWD_id.containsKey(parser.kafAugTokensArrayList.get(j+1).getMId());
					//last or middle: (no token after or token after corefers)
					if (containsKeyForNextToken) {
						chain = mIdToWD_id.get(parser.kafAugTokensArrayList.get(j+1).getMId());
					}

					if(j+1 < parser.kafAugTokensArrayList.size() && (!containsKeyForNextToken || chain.equals(curChain))) {

						if(containsKeyForNextToken && mIdToWD_id.get(parser.kafAugTokensArrayList.get(j+1).getMId()).equals(curChain)) {
							//do nothing
						}
						//last (token after doesn't corefer)
						else
							coreference = coreference + ")";	
					}
					//last (last token in chain)
					else
						coreference = coreference + ")";
				}
				String delim = "\t";

				rows.append(
						//1
						document_id + delim + 
						//2
						part_num + delim + 
						//3
						word_num + delim +
						//4	
						word_itself + delim +
						//5	
						pos + delim + 
						//6	
						parse_bit + delim +
						//7	
						predicate_lemma + delim +
						//8	
						predicate_frameset_id + delim +
						//9	
						word_sense + delim +
						//10	
						speaker + delim +
						//11	
						named_entities + delim +
						//12	
						predicate_args + delim +
						//13	
						coreference + "\n");
			}
			rows.append("\n");
//			rows.append("#end document");
//			rows.append("\n");
		}
		rows.append("#end document");
//		rows.setLength(rows.length() - 1);
		//append string buffer/builder to buffered writer
		File f = null;
		try {
			f = new File(writeDir + "/" + "CDEC_model.conll");
			f.createNewFile();
			FileWriter writer = new FileWriter(f);
			writer.write(rows.toString());
			writer.flush();
			writer.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
		return f;
	}
	
	
	//========================================== WDEC ==========================================
	//==========================================================================================
	//==========================================================================================
	
	//########################################## Gold ##########################################
	
	public File writeWDConllFromGold(List<File> files, String writeDir,HashMap<String, HashSet<String>> cleanSentences, boolean crossDoc) {
		//1 (path)
		String document_id = "";
		//2 (index in cluster)
		String part_num = "";
		//3 (t_id)
		String word_num = "";
		//4 (surface form)
		String word_itself = "";
		//5 null
		String pos = "-";
		//6 null
		String parse_bit = "-";
		//7 null
		String predicate_lemma = "-";
		//8 null
		String predicate_frameset_id = "-";
		//9 null
		String word_sense = "-";
		//10 null
		String speaker = "-";
		//11 null
		String named_entities = "-";
		//12 null
		String predicate_args = "-";
		//13 (chain_id)
		String coreference = "";
		StringBuilder rows = new StringBuilder();
		int i = 0;
		TreeMap<Integer, HashMap<String, LinkedList<File>>> clusters = getFileClusters(files);
		LinkedList<String> subTopics = Stream.of("ecbplus","ecb").collect(Collectors.toCollection(LinkedList::new));
		for(int topic : clusters.keySet()) {
			for(String subTopic : subTopics) {
				if(crossDoc)
					rows.append("#begin document (" + topic + "_" + subTopic + "); part " + i +"\n\n");
				for(File f : clusters.get(topic).get(subTopic)) {
					
					// file isn't test, continue
					if(!Globals.TEST_TOPICS.contains(String.valueOf(topic)))
						continue;
					if(Globals.DOING_DEV) {
						if(!Globals.DEV_TOPICS.contains(String.valueOf(topic)))
							continue;
					}
					String fileKey = f.getName().replace("_aug.en.naf", "");
					
					// file wasn't cleaned by ecb+ people
					if(!cleanSentences.containsKey(fileKey))
						continue;
					Document doc = null;
					try {
						DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
						DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
						doc = dBuilder.parse(f);
					}
					catch(Exception e) {
						e.printStackTrace();
					}
					
					
					//7_3ecbplus_aug.en.naf
					String generalFileName = f.getName().replace("_aug.en.naf","");
					if(!crossDoc)
						rows.append("#begin document (" +generalFileName + "); part " + i +"\n\n");
					
					//1
					document_id = generalFileName;
					//2
					part_num = Integer.toString(i++);
					
					NodeList augTokens = doc.getElementsByTagName("Augmented_Tokens").item(0).getChildNodes();
					KafSaxParser parser = new KafSaxParser();
					boolean augmented = true;
					parser.parseFile(f,augmented);
					//<m_id, tokens>
					HashMap<String,LinkedList<String>> actions = parser.getGoldActions();
					
					String curSentence = "0";
					int sentIndex = 0;
					for(int j = 0; j < augTokens.getLength(); j++) {
						
						if(augTokens.item(j).getNodeType() != Node.ELEMENT_NODE)
							continue;
						
						Element augToken = (Element)augTokens.item(j);
						// sentence not cleaned, continue
						if(!cleanSentences.get(fileKey).contains(augToken.getAttribute("sentence")))
							continue;
						
						if(!augToken.getAttribute("sentence").equals(curSentence)) {
							sentIndex = 0;
							curSentence = augToken.getAttribute("sentence");
							rows.append("\n");
						}
						//3
						word_num = augToken.getAttribute("t_id");
						//if want sentence indexes to start at 0 for every sentence
//						word_num = Integer.toString(sentIndex++);
						//4
						word_itself = augToken.getTextContent();
						//get coref chain num
						boolean multi_with_verb = false;
						int[] window = {-2,-1,0,1,2};
						multi_with_verb = false;
						for (int w : window) {
							try {
								multi_with_verb = multi_with_verb || parser.getTermForWordId("w" + Integer.toString(Integer.parseInt(word_num)+w)).getPos().equals("V");
							}
							catch(NullPointerException e) {
								//do nothing
							}
						}
						//coreference
						if(augToken.getAttribute("ev_type").equals("") || !augToken.getAttribute("ev_type").contains("ACT")) {
							coreference = "-";
						}
						else {
							String curChain = augToken.getAttribute("ev_id");
							//getFirst() should just be a string (ie. make token:chain -> 1:1:), will deal with this later
							coreference = augToken.getAttribute("ev_id").replace("ACT", "");
							if(augToken.getAttribute("ev_id").contains("NEG"))
								coreference = augToken.getAttribute("ev_id").replace("NEG","");
							//cases:
							//first or middle: (no token before or token before corefers)
							if(j-2 > 0 && (!((Element)augTokens.item(j-2)).getAttribute("ev_id").equals("")
									|| ((Element)augTokens.item(j-2)).getAttribute("ev_id").equals(curChain))) {
								//middle (token before corefers)
								if(((Element)augTokens.item(j-2)).getAttribute("ev_id").equals(curChain)) {
									//do nothing
								}
								//first (token before doesn't corefer)
								else
									coreference = "(" + coreference;	
							}
							//first (first token in chain)
							else
								coreference = "("  +  coreference;
							//last or middle: (no token after or token after corefers)
							if(j + 2 < augTokens.getLength() && (!((Element)augTokens.item(j+2)).getAttribute("ev_id").equals("")
									|| ((Element)augTokens.item(j+2)).getAttribute("ev_id").equals(curChain))) {
								if(((Element)augTokens.item(j+2)).getAttribute("ev_id").equals(curChain)) {
									//do nothing
								}
								//last (token after doesn't corefer)
								else
									coreference = coreference + ")";	
							}
							//last (last token in chain)
							else
								coreference = coreference + ")";
						}
						String delim = "\t";

						rows.append(
									//1
									document_id + delim + 
									//2
									part_num + delim + 
									//3
									word_num + delim +
									//4	
									word_itself + delim +
									//5	
									pos + delim + 
									//6	
									parse_bit + delim +
									//7	
									predicate_lemma + delim +
									//8	
									predicate_frameset_id + delim +
									//9	
									word_sense + delim +
									//10	
									speaker + delim +
									//11	
									named_entities + delim +
									//12	
									predicate_args + delim +
									//13	
									coreference + "\n");
					}
					if(crossDoc)
						rows.append("\n");
					else {
						rows.append("#end document");
						rows.append("\n");
					}
				}
				if(crossDoc) {
					rows.append("#end document");
					rows.append("\n");
				}
			}
		}
		if(!crossDoc)
			rows.setLength(rows.length() - 1);
		//append string buffer/builder to buffered writer
		File f = null;
		try {
			f = new File(writeDir + "/WDEC_gold.conll");
			f.createNewFile();
			FileWriter writer = new FileWriter(f);
			writer.write(rows.toString());
			writer.flush();
			writer.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
		return f;
	}


	//########################################## Model ##########################################

	
	public File writeWDConllFromPreds(List<File> files,WDEC wdec, String writeDir,HashMap<String, HashSet<String>> cleanSentences, boolean crossDoc) {
		
		//1 (path)
		String document_id = "";
		//2 (index in cluster)
		String part_num = "";
		//3 (t_id)
		String word_num = "";
		//4 (surface form)
		String word_itself = "";
		//5 null
		String pos = "-";
		//6 null
		String parse_bit = "-";
		//7 null
		String predicate_lemma = "-";
		//8 null
		String predicate_frameset_id = "-";
		//9 null
		String word_sense = "-";
		//10 null
		String speaker = "-";
		//11 null
		String named_entities = "-";
		//12 null
		String predicate_args = "-";
		//13 (chain_id)
		String coreference = "";
		StringBuilder rows = new StringBuilder();
		int i = 0;
		
		TreeMap<Integer, HashMap<String, LinkedList<File>>> clusters = getFileClusters(files);
		LinkedList<String> subTopics = Stream.of("ecbplus","ecb").collect(Collectors.toCollection(LinkedList::new));
		for(int topic : clusters.keySet()) {
			for(String subTopic : subTopics) {
				if(crossDoc)
					rows.append("#begin document (" + topic + "_" + subTopic + "); part " + i +"\n\n");

				for(File f : clusters.get(topic).get(subTopic)) {
					
					if(!Globals.TEST_TOPICS.contains(String.valueOf(topic)))
						continue;
					if(Globals.DOING_DEV) {
						if(!Globals.DEV_TOPICS.contains(String.valueOf(topic)))
							continue;
					}
					String fileKey = f.getName().replace("_aug.en.naf", "");
					
					// file wasn't cleaned by ecb+ people
					if(!cleanSentences.containsKey(fileKey))
						continue;

					HashMap<String,String> mIdToWD_id = wdec.fileToM_idToWD_id.get(f);
					
					//7_3ecbplus_aug.en.naf
					String generalFileName = f.getName().replace("_aug.en.naf","");
					if(!crossDoc)
						rows.append("#begin document (" +generalFileName + "); part " + i +"\n\n");
					
					//1
					document_id = generalFileName;
					//2
					part_num = Integer.toString(i++);
					
					KafSaxParser parser = new KafSaxParser();
					boolean augmented = true;
					parser.parseFile(f,augmented);

					String curSentence = "0";
					
					for(int j = 0; j < parser.kafAugTokensArrayList.size(); j++) {
						KafAugToken augToken = parser.kafAugTokensArrayList.get(j);
						
						// sentence not cleaned, continue
						if(!cleanSentences.get(fileKey).contains(augToken.getSentenceNum()))
							continue;

						
						if(!augToken.getSentenceNum().equals(curSentence)) {
							curSentence = augToken.getSentenceNum();
							rows.append("\n");
						}
						//3
						word_num = augToken.getTid().replace("t","");
						//if want sentence indexes to start at 0 for every sentence
//								word_num = Integer.toString(sentIndex++);
						//4
						word_itself = augToken.getTokenString();
						
						// annotation error
						if(mIdToWD_id == null)
							coreference = "-";
						//coreference
						else if(!mIdToWD_id.containsKey(augToken.getMId())) {
							coreference = "-";
						}
						else {
							String curChain = mIdToWD_id.get(augToken.getMId());

							coreference = mIdToWD_id.get(augToken.getMId());

							//cases:
							
							boolean containsKeyForPrevToken = j-1 > 0 && mIdToWD_id.containsKey(parser.kafAugTokensArrayList.get(j-1).getMId());
							//first or middle: (no token before or token before corefers)
							String chain = "";
							if (containsKeyForPrevToken) {
								chain = mIdToWD_id.get(parser.kafAugTokensArrayList.get(j-1).getMId());
							}
							if(j-1 > 0 && (!containsKeyForPrevToken || chain.equals(curChain))) {
								//middle (token before corefers)
								if(containsKeyForPrevToken && mIdToWD_id.get(parser.kafAugTokensArrayList.get(j-1).getMId()).equals(curChain)) {
									//do nothing
								}
								//first (token before doesn't corefer)
								else
									coreference = "(" + coreference;	
							}
							//first (first token in chain)
							else
								coreference = "("  +  coreference;
							boolean containsKeyForNextToken = j + 1 < parser.kafAugTokensArrayList.size() && mIdToWD_id.containsKey(parser.kafAugTokensArrayList.get(j+1).getMId());
							//last or middle: (no token after or token after corefers)
							if (containsKeyForNextToken) {
								chain = mIdToWD_id.get(parser.kafAugTokensArrayList.get(j+1).getMId());
							}

							if(j+1 < parser.kafAugTokensArrayList.size() && (!containsKeyForNextToken || chain.equals(curChain))) {

								if(containsKeyForNextToken && mIdToWD_id.get(parser.kafAugTokensArrayList.get(j+1).getMId()).equals(curChain)) {
									//do nothing
								}
								//last (token after doesn't corefer)
								else
									coreference = coreference + ")";	
							}
							//last (last token in chain)
							else
								coreference = coreference + ")";
						}
						String delim = "\t";

						rows.append(
									//1
									document_id + delim + 
									//2
									part_num + delim + 
									//3
									word_num + delim +
									//4	
									word_itself + delim +
									//5	
									pos + delim + 
									//6	
									parse_bit + delim +
									//7	
									predicate_lemma + delim +
									//8	
									predicate_frameset_id + delim +
									//9	
									word_sense + delim +
									//10	
									speaker + delim +
									//11	
									named_entities + delim +
									//12	
									predicate_args + delim +
									//13	
									coreference + "\n");
					}
					if(crossDoc)
						rows.append("\n");
					else {
						rows.append("#end document");
						rows.append("\n");
					}
				}
				if(crossDoc) {
					rows.append("#end document");
					rows.append("\n");
				}
			}
		}
		if(!crossDoc)
			rows.setLength(rows.length() - 1);
		//append string buffer/builder to buffered writer
		File f = null;
		try {
			f = new File(writeDir + "/WDEC_model.conll");
			f.createNewFile();
			FileWriter writer = new FileWriter(f);
			writer.write(rows.toString());
			writer.flush();
			writer.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
		return f;
	}
	
	
	
	
	
	
	///////////// maybe later ////////////////////
	
	public File writePiekWDEC(List<File> files, String writeDir,HashMap<String, HashSet<String>> cleanSentences) {
		//1 (path)
		String document_id = "";
		//2 (index in cluster)
		String part_num = "";
		//3 (t_id)
		String word_num = "";
		//4 (surface form)
		String word_itself = "";
		//5 null
		String pos = "-";
		//6 null
		String parse_bit = "-";
		//7 null
		String predicate_lemma = "-";
		//8 null
		String predicate_frameset_id = "-";
		//9 null
		String word_sense = "-";
		//10 null
		String speaker = "-";
		//11 null
		String named_entities = "-";
		//12 null
		String predicate_args = "-";
		//13 (chain_id)
		String coreference = "";
		StringBuilder rows = new StringBuilder();
		int i = 0;
		int corefId = 0;
		for(File f : files) {
			
			String topic = f.getName().split("_")[0];
			// file isn't test, continue
			if(!Globals.TEST_TOPICS.contains(topic))
				continue;
			String fileKey = f.getName().replace("_aug.en.naf", "");
			
			// file wasn't cleaned by ecb+ people
			if(!cleanSentences.containsKey(fileKey))
				continue;
			Document doc = null;
			try {
				DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
				DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
				doc = dBuilder.parse(f);
			}
			catch(Exception e) {
				e.printStackTrace();
			}
			
			
			//7_3ecbplus_aug.en.naf
			String generalFileName = f.getName().replace("_aug.en.naf","");
			rows.append("#begin document (" +generalFileName + "); part " + i +"\n\n");
			
			//1
			document_id = generalFileName;
			//2
			part_num = Integer.toString(i++);
			
			NodeList augTokens = doc.getElementsByTagName("Augmented_Tokens").item(0).getChildNodes();
			NodeList corefTags = doc.getElementsByTagName("coreferences").item(0).getChildNodes();
			for(int j = 0; j < corefTags.getLength(); j++) {
				if(corefTags.item(j).getNodeType() != Node.ELEMENT_NODE)
					continue;
				Element corefTag = (Element)corefTags.item(j);
				System.out.println(corefTag.getAttribute("id"));
			}
			System.exit(0);
			KafSaxParser parser = new KafSaxParser();
			boolean augmented = true;
			parser.parseFile(f,augmented);
			//<m_id, tokens>
			HashMap<String,LinkedList<String>> actions = parser.getGoldActions();
			
			String curSentence = "0";
			int sentIndex = 0;
			
			for(int j = 0; j < augTokens.getLength(); j++) {
				
				if(augTokens.item(j).getNodeType() != Node.ELEMENT_NODE)
					continue;
				
				Element augToken = (Element)augTokens.item(j);
				// sentence not cleaned, continue
				if(!cleanSentences.get(fileKey).contains(augToken.getAttribute("sentence")))
					continue;
				
				if(!augToken.getAttribute("sentence").equals(curSentence)) {
					sentIndex = 0;
					curSentence = augToken.getAttribute("sentence");
					rows.append("\n");
				}
				//3
				word_num = augToken.getAttribute("t_id");
				//if want sentence indexes to start at 0 for every sentence
//				word_num = Integer.toString(sentIndex++);
				//4
				word_itself = augToken.getTextContent();
				//get coref chain num
				boolean multi_with_verb = false;
				int[] window = {-2,-1,0,1,2};
				multi_with_verb = false;
				for (int w : window) {
					try {
						multi_with_verb = multi_with_verb || parser.getTermForWordId("w" + Integer.toString(Integer.parseInt(word_num)+w)).getPos().equals("V");
					}
					catch(NullPointerException e) {
						//do nothing
					}
				}
				//coreference
				if(augToken.getAttribute("ev_type").equals("") || !augToken.getAttribute("ev_type").contains("ACT")) {
					coreference = "-";
				}
				else {
					String curChain = augToken.getAttribute("ev_id");
					//getFirst() should just be a string (ie. make token:chain -> 1:1:), will deal with this later
					coreference = augToken.getAttribute("ev_id").replace("ACT", "");
					if(augToken.getAttribute("ev_id").contains("NEG"))
						coreference = augToken.getAttribute("ev_id").replace("NEG","");
					//cases:
					//first or middle: (no token before or token before corefers)
					if(j-2 > 0 && (!((Element)augTokens.item(j-2)).getAttribute("ev_id").equals("")
							|| ((Element)augTokens.item(j-2)).getAttribute("ev_id").equals(curChain))) {
						//middle (token before corefers)
						if(((Element)augTokens.item(j-2)).getAttribute("ev_id").equals(curChain)) {
							//do nothing
						}
						//first (token before doesn't corefer)
						else
							coreference = "(" + coreference;	
					}
					//first (first token in chain)
					else
						coreference = "("  +  coreference;
					//last or middle: (no token after or token after corefers)
					if(j + 2 < augTokens.getLength() && (!((Element)augTokens.item(j+2)).getAttribute("ev_id").equals("")
							|| ((Element)augTokens.item(j+2)).getAttribute("ev_id").equals(curChain))) {
						if(((Element)augTokens.item(j+2)).getAttribute("ev_id").equals(curChain)) {
							//do nothing
						}
						//last (token after doesn't corefer)
						else
							coreference = coreference + ")";	
					}
					//last (last token in chain)
					else
						coreference = coreference + ")";
				}
				String delim = "\t";

				rows.append(
						//1
						document_id + delim + 
						//2
						part_num + delim + 
						//3
						word_num + delim +
						//4	
						word_itself + delim +
						//5	
						pos + delim + 
						//6	
						parse_bit + delim +
						//7	
						predicate_lemma + delim +
						//8	
						predicate_frameset_id + delim +
						//9	
						word_sense + delim +
						//10	
						speaker + delim +
						//11	
						named_entities + delim +
						//12	
						predicate_args + delim +
						//13	
						coreference + "\n");
			}
			rows.append("\n");
			rows.append("#end document");
			rows.append("\n");
		}
//		rows.append("#end document");
		rows.setLength(rows.length() - 1);
		//append string buffer/builder to buffered writer
		File f = null;
		try {
			f = new File(writeDir + "/WDEC_gold.conll");
			f.createNewFile();
			FileWriter writer = new FileWriter(f);
			writer.write(rows.toString());
			writer.flush();
			writer.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
		return f;
		
		
	}
	
}
	
