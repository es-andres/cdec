package ecb_utils;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;
import java.util.stream.Collectors;

import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClientBuilder;

import com.google.common.graph.GraphBuilder;
import com.google.common.graph.MutableGraph;
import com.google.common.net.UrlEscapers;

import common.Globals;
import common.SerUtils;
import edu.emory.clir.clearnlp.util.StringUtils;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreNLPProtos;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer;
import edu.stanford.nlp.semgraph.SemanticGraph;
import feature_extraction.EventFeatures;

/**
 * - In memory representation of an ecb+ naf file.
 * - Belongs to a train or test set.
 * - Talks to globalCorefGraph to populate CDEC clusters 
 *   for its set.
 * 
 * @author Andres Cremisini
 *
 */
public class ECBDoc {
	
	public File file; // location of ecb+ file
	public int topic; // topic file belongs to
	public HashMap<String, List<String>> mentionIdToTokens; // mention_id to list of tokens
	public HashMap<String, List<String>> sentenceIdToTokens; // sentence_id to list of tokens
	public HashMap<String, HashMap<String, String>> toks; // token_id to Augmented_Tokens xml element
	public LinkedList<String> inOrderToks; // list of token ids as they appear in the document
	public HashMap<String, HashSet<String>> actionToMentions; // global ecb+ CDEC cluster label (action_id) to local mention_id members
	public MutableGraph<EventNode> evCorefGraph; // binary event coreference graph (edge denotes coreference)
	public HashMap<String, HashMap<String, TreeMap<Integer, IndexedWord>>> mIdToEventText;
	public CoreDocument coreDoc; // CoreNLP parse


	public ECBDoc(File f) {
		this.file = f;
		this.topic = Integer.parseInt(f.getName().split("_")[0]);
		this.toks = new HashMap<String, HashMap<String, String>>();
		this.inOrderToks = new LinkedList<String>();
		this.mentionIdToTokens = new HashMap<String, List<String>>();
		this.sentenceIdToTokens = new HashMap<String, List<String>>();
		this.actionToMentions = new HashMap<String, HashSet<String>>();
		this.evCorefGraph = GraphBuilder.undirected().<EventNode>build();
		this.mIdToEventText = new HashMap<String, HashMap<String, TreeMap<Integer,IndexedWord>>>();
		
		List<HashMap<String, String>> tokens = this.readXML(f);
		this.parseTokens(tokens);
		if(Globals.USED_CACHED_PARSES)
			coreDoc = this.getSerializedCoreDoc();
		else
			coreDoc = this.parseDocWithCoreNLP();
		this.getCoreSemanticDeps();
	}
	
	public String getTopic() {
		return this.file.getName().split("_")[0];
	}
	
	public String getSubTopic() {
		String sub = "";
		if(this.file.getName().contains("ecbplus"))
			sub = "ecbplus";
		else
			sub = "ecb";
		return sub;
	}
	
	public String getDocText() {
		String doc = "";
		for(CoreLabel tok : coreDoc.tokens()) {
			String clean = EventFeatures.cleanTok(tok, false, new HashSet<String>(Arrays.asList("N", "V")));
			if(clean!= null)
				doc += clean + " ";
		}
		doc = doc.substring(0, doc.length() - 1);
		return doc;
	}
	
	public String getHeadText(String m_id) {
		List<String> t_ids = this.mentionIdToTokens.get(m_id);
		String s_num = this.toks.get(t_ids.get(0)).get("sentence");
		int offset = 0;
		if(this.toks.get("1").get("sentence").equals("1")) {
			System.out.println(this.file);
			offset++;
		}
		Integer s_idx = Integer.parseInt(s_num) + offset;
		CoreSentence sent = this.coreDoc.sentences().get(s_idx);
		String head = "";
		for(String t_id : t_ids) {
			CoreLabel coreTok = sent.tokens().get(Integer.parseInt(this.toks.get(t_id).get("number")));
			String ecbTok = this.toks.get(t_id).get("text");
			if(this.file.getName().equals("24_10ecb_aug.xml") && t_id.equals("14")) { // weird corenlp bug
				coreTok.setOriginalText(ecbTok);
			}
			if(this.file.getName().equals("2_7ecbplus_aug.xml") && t_id.equals("56")) { // weird corenlp encoding bug
				coreTok.setOriginalText(ecbTok);
			}
			assertEquals(coreTok.originalText(), ecbTok);
			String clean = EventFeatures.cleanTok(coreTok, false, new HashSet<String>());
			if(clean != null)
				head += clean + "+";
		}
		return head.substring(0, head.length() - 1);
	}
	
	private List<HashMap<String,String>> readXML(File f) {
		XMLInputFactory xmlInputFactory = XMLInputFactory.newInstance();
		List<HashMap<String, String>> tokens = new LinkedList<HashMap<String,String>>();
		FileInputStream fileReader;
		XMLEventReader xmlEventReader;
		try {
			fileReader = new FileInputStream(f);
			xmlEventReader = xmlInputFactory.createXMLEventReader(fileReader);
			while (xmlEventReader.hasNext()) {
				XMLEvent xmlEvent = xmlEventReader.nextEvent();
				if (xmlEvent.isStartElement()) {
					StartElement startElement = xmlEvent.asStartElement();
					if (startElement.getName().getLocalPart().equals("aug_token")) {
						HashMap<String,String> token = new HashMap<String, String>();
						Iterator<?> it = startElement.getAttributes();
						while(it.hasNext()) {
							Attribute att = (Attribute)it.next();
							// just for now before changing this in Augmented_Tokens python script
							if(att.getName().toString().equals("pred_ev")) {
								String pred_mid = att.getValue();
								if(!pred_mid.equals("") && !pred_mid.contains("pred_"))
									pred_mid = att.getValue(); 
								token.put(att.getName().toString(), pred_mid);
							}
							else
								token.put(att.getName().toString(), att.getValue());
						}
						String txt = xmlEventReader.nextEvent().toString().replaceAll("\\s+","");
						String clean = txt.replaceAll("[^a-zA-Z0-9]", "");
						// filter out spurious (non-word and single letter) event trigger predictions
						if(clean.length() <= 1)
							token.put("pred_ev", "");
						token.put("text", txt);
						tokens.add(token);
					}
				}
			}
			xmlEventReader.close();
			fileReader.close();

		} 
		catch (XMLStreamException | IOException e) {
			e.printStackTrace();
		}
		finally {
			
		}
		return tokens;
	}
	
	/**
	 * Populate token, mention, action and sentence hashmaps
	 * @param nodes: NodeList of xml Augmented_Tokens elements
	 */
	private void parseTokens(List<HashMap<String,String>> tokens) {
		
		for(HashMap<String, String> token : tokens) {
			/*
			 * token
			 */
			String t_id = token.get("t_id");
			this.inOrderToks.add(t_id);
			this.toks.put(t_id, token);
			/*
			 * sentence
			 */
			String s_id = token.get("sentence");
			if(!this.sentenceIdToTokens.containsKey(s_id))
				this.sentenceIdToTokens.put(s_id, new LinkedList<>());
			this.sentenceIdToTokens.get(s_id).add(t_id);
			
			/*
			 * mention, action, graph nodes
			 */
			// only log mentions if the sentence was cleaned by ecb
			if(Globals.cleanSentences.get(ECBWrapper.cleanFileName(this.file)).contains(s_id)) {
				
				/*
				 * gold
				 */
				String m_id = token.get("m_id"); 
				String ev_id = token.get("ev_id");
				if(ev_id.contains("ACT")) { // only add action mentions (event triggers)
					// mention
					if(!this.mentionIdToTokens.containsKey(m_id))
						this.mentionIdToTokens.put(m_id, new LinkedList<String>());
					this.mentionIdToTokens.get(m_id).add(t_id);
					
					// action
					if(!this.actionToMentions.containsKey(ev_id))
						this.actionToMentions.put(ev_id, new HashSet<String>());
					this.actionToMentions.get(ev_id).add(m_id);
				}
				
				/*
				 * pred
				 */
				String pred_m_id = token.get("pred_ev");
				if(!pred_m_id.equals("")) {
					// mention
					if(!this.mentionIdToTokens.containsKey(pred_m_id))
						this.mentionIdToTokens.put(pred_m_id, new LinkedList<String>());
					this.mentionIdToTokens.get(pred_m_id).add(t_id);
					
					// action
					if(!this.actionToMentions.containsKey("preds"))
						this.actionToMentions.put("preds", new HashSet<String>());
					this.actionToMentions.get("preds").add(pred_m_id);
				}
				
			}
		}
	}
	
	private void getCoreSemanticDeps() {

		for(String m_id : this.mentionIdToTokens.keySet()) {
			String s_id = this.toks.get(this.mentionIdToTokens.get(m_id).get(0)).get("sentence");
			CoreSentence coreSent = coreDoc.sentences().get(Integer.parseInt(s_id));
			SemanticGraph semGraph = coreSent.dependencyParse();
			TreeMap<Integer, IndexedWord> evHeads = new TreeMap<Integer, IndexedWord>();
			TreeMap<Integer, IndexedWord> evDeps = new TreeMap<Integer, IndexedWord>();
			
			for(String t_id : this.mentionIdToTokens.get(m_id)) {
				CoreLabel coreTok = coreSent.tokens().get(Integer.parseInt(this.toks.get(t_id).get("number")));
				IndexedWord headNode = semGraph.getNodeByIndex(coreTok.index());
				evHeads.put(headNode.index(), headNode);
//				assertEquals(headNode.originalText(), this.toks.get(String.valueOf(firstIdx + headNode.index() - 1)).getTextContent());
				for(IndexedWord depNode : semGraph.descendants(headNode)) {
					if(StringUtils.containsPunctuation(depNode.tag()))
						continue;
					evDeps.put(depNode.index(), depNode);
				}
			}

			this.mIdToEventText.put(m_id, new HashMap<String,TreeMap<Integer, IndexedWord>>());
			this.mIdToEventText.get(m_id).put("trigger", evHeads);
			this.mIdToEventText.get(m_id).put("deps", evDeps);
		}
	}
	
	private CoreDocument parseDocWithCoreNLP() {
		String doc = "";
		List<Integer> sentIdx = this.sentenceIdToTokens.keySet().stream().map(s -> Integer.parseInt(s)).collect(Collectors.toList());
		Collections.sort(sentIdx);

		for(int s_id: sentIdx) {
			for(String t_id : this.sentenceIdToTokens.get(String.valueOf(s_id)))
				doc += this.toks.get(t_id).get("text") + " ";
			doc = doc.substring(0, doc.length() - 1);
			doc += "\n";
		}
		doc = doc.substring(0, doc.length() - 2);
		HttpClient httpClient = HttpClientBuilder.create().build();
		String args = "properties={" +
                "\"annotators\": \"tokenize,ssplit,parse,coref\", " +
                "\"ssplit.eolonly\": \"True\"," +
				"\"tokenize.whitespace\": \"True\"," +
                "\"tokenize.tokenize_pretokenized\": \"True\"," +
				"\"coref.algorithm\": \"neural\"," + 
                "\"outputFormat\": \"serialized\","
                + "\"serializer\": \"edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer\"}";
        StringEntity requestEntity = new StringEntity(doc,ContentType.APPLICATION_JSON);

    	HttpPost postMethod = new HttpPost(Globals.CORE_NLP_SERVER + "/?" + UrlEscapers.urlFormParameterEscaper().escape(args));
    	postMethod.setEntity(requestEntity);
    	HttpResponse rawResponse = null;
    	ProtobufAnnotationSerializer serial = new ProtobufAnnotationSerializer();
    	Annotation coreAnnotation = null;
    	try {
			rawResponse = httpClient.execute(postMethod);
			CoreNLPProtos.Document protoDoc = CoreNLPProtos.Document.parseDelimitedFrom(rawResponse.getEntity().getContent());
			coreAnnotation = serial.fromProto(protoDoc);
			
		} 
    	catch (IOException e) {
			e.printStackTrace();
		}
    	
    	CoreDocument coreDoc = new CoreDocument(coreAnnotation);

		return coreDoc;
	}
	
	private CoreDocument getSerializedCoreDoc() {
		ProtobufAnnotationSerializer ser = new ProtobufAnnotationSerializer();
		Object doc = SerUtils.loadObj(Paths.get(Globals.CACHED_CORE_DOCS.toString(), this.file.getName() + ".ser").toFile());
		return new CoreDocument(ser.fromProto((CoreNLPProtos.Document)doc));
	}
	
	@SuppressWarnings("unused")
	private void printCorefs(String type) {
		for(String m_id : this.mIdToEventText.keySet()) {
			for(int i : this.mIdToEventText.get(m_id).get("trigger").navigableKeySet()) {
				IndexedWord n = this.mIdToEventText.get(m_id).get("trigger").get(i);
				System.out.println(n.originalText()+ ": " + n.tag());
			}
			for(int i : this.mIdToEventText.get(m_id).get("deps").navigableKeySet()) {
				IndexedWord n = this.mIdToEventText.get(m_id).get("deps").get(i);
				System.out.println("\t"+n.originalText() + ": " + n.tag());
			}
			System.out.println("---");
		}
		
	}
}
