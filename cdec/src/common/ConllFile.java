package common;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.logging.Logger;

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
					clusterIdMap.put(n.getGlobalKey(), String.valueOf(i));//String.valueOf(i)
				}
				else
					System.exit(-1);
			}
		}
		
		StringBuilder conllDoc = new StringBuilder("#begin document (cdec);\n");
		for(int i =0; i < files.size(); i++) {
			File f = files.get(i);
			ECBDoc doc = dataWrapper.docs.get(f.getName());

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
	
}
	
