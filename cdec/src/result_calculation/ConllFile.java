package result_calculation;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.logging.Logger;

import common.Globals;
import ecb_utils.ECBDoc;
import ecb_utils.ECBWrapper;
import ecb_utils.EventNode;

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
					clusterIdMap.put(n.getGlobalKey(), String.valueOf(i));
				}
				else
					System.exit(-1);
			}
		}
		
		StringBuilder conllDoc = new StringBuilder("#begin document (cdec);\n");
		int extra = evClusters.size();
		for(int i =0; i < files.size(); i++) {
			File f = files.get(i);
			ECBDoc doc = dataWrapper.docs.get(f.getName());

			for(String t_id : doc.inOrderToks) {
				HashMap<String, String> tok = doc.toks.get(t_id);
				String m_id = tok.get("m_id");
				String ev_id = tok.get("ev_id");
				String pred_m_id = tok.get("pred_ev");
				String label = "-";
				
				if(Globals.cleanSentences.get(ECBWrapper.cleanFileName(f)).contains(tok.get("sentence"))){
					String clustId = "-";
					if(fName.contains("gold")) {
						if(ev_id.contains("ACT")) {
							EventNode n = new EventNode(f, m_id, ev_id);
							clustId = ev_id.replace("ACT", "");
							if(clusterIdMap.containsKey(n.getGlobalKey()))
								clustId = clusterIdMap.get(n.getGlobalKey());
							label = getLabel(clustId, doc, m_id, t_id);
						}
					}
					else if (fName.contains("model")) {
						if(Globals.USE_TEST_PRED_EVS) {
							if(!pred_m_id.equals("")) {
								EventNode n = new EventNode(f, pred_m_id, ev_id);
								clustId = clusterIdMap.get(n.getGlobalKey());
								label = getLabel(clustId, doc, pred_m_id, t_id);
							}
							
						}
						else {
							if(ev_id.contains("ACT")) {
								EventNode n = new EventNode(f, m_id, ev_id);
								clustId = String.valueOf(extra);
								if(clusterIdMap.containsKey(n.getGlobalKey()))
									clustId = clusterIdMap.get(n.getGlobalKey());
								label = getLabel(clustId, doc, m_id, t_id);
							}
						}
					}
//					if(ev_id.contains("ACT")) {
//						String pred_or_gold_mid = true ? pred_m_id : m_id;
//						EventNode n = new EventNode(f, pred_or_gold_mid, ev_id);
//						String clustId = "-";
//						if(!clusterIdMap.containsKey(n.getGlobalKey())) {
////							System.out.println(f.getName() + " " + m_id + " " + ev_id);
//						}
//						else {
//							clustId = clusterIdMap.get(n.getGlobalKey());
//						}
//							
//						int spanSize = doc.mentionIdToTokens.get(pred_or_gold_mid).size();
//						if(spanSize == 1)
//							label = "(" + clustId + ")";
//						else {
//							if(t_id.equals(doc.mentionIdToTokens.get(pred_or_gold_mid).get(0)))
//								label = "(" + clustId;
//							else if(t_id.equals(doc.mentionIdToTokens.get(pred_or_gold_mid).get(spanSize - 1)))
//								label = clustId + ")";
//							else
//								label = clustId;
//						}
//						
//					}
				}
				String fKey = f.getName().replace("_aug.xml", "");
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
	
	private static String getLabel(String clustId, ECBDoc doc, String m_id, String t_id) {
		
		String label = "";
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
		return label;
	}
	
}
	
