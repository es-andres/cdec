package naf;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.net.URL;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.EndElement;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

import org.apache.commons.collections4.iterators.NodeListIterator;
import org.apache.http.HttpResponse;
import org.apache.http.NameValuePair;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.util.EntityUtils;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import com.google.common.collect.Sets;
import com.google.common.graph.Graph;
import com.google.common.graph.GraphBuilder;
import com.google.common.graph.Graphs;
import com.google.common.graph.MutableGraph;
import com.google.common.net.UrlEscapers;

import common.Globals;
import common.SerUtils;
import comparer.TFIDF;
import common.ECBWrapper;
import edu.emory.clir.clearnlp.dependency.DEPNode;
import edu.emory.clir.clearnlp.dependency.DEPTree;
import edu.emory.clir.clearnlp.srl.SRLTree;
import edu.emory.clir.clearnlp.util.StringUtils;
import edu.emory.clir.clearnlp.util.arc.DEPArc;
import edu.emory.clir.clearnlp.util.arc.SRLArc;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.CorefChain.CorefMention;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreNLPProtos;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.IntPair;
import vu.wntools.lmf.Gloss;

/**
 * - In memory representation of an ecb+ naf file.
 * - Belongs to a train or test set.
 * - Talks to globalCorefGraph to populate CDEC clusters 
 *   for its set.
 * 
 * @author Andres Cremisini
 *
 */
public class NafDoc {
	
	public File file; // location of ecb+ file
	public int topic; // topic file belongs to
	public HashMap<String, List<String>> mentionIdToTokens; // mention_id to list of tokens
	public HashMap<String, List<String>> sentenceIdToTokens; // sentence_id to list of tokens
	public HashMap<String, HashMap<String, String>> toks; // token_id to Augmented_Tokens xml element
	public LinkedList<String> inOrderToks;
	public HashMap<String, HashSet<String>> actionToMentions; // global ecb+ CDEC cluster label (action_id) to local mention_id members
	public HashMap<String,  CorefChain> tokIdToEntCorefChain;
	public MutableGraph<EventNode> evCorefGraph; // binary event coreference graph (edge denotes coreference)
	public HashMap<String, DEPTree> sentenceToSRLParse;
	public HashMap<String, DEPNode[]> sentenceToSRLNodes;
	public HashMap<String, HashMap<String, TreeMap<Integer,DEPNode>>> mIdToEvText;
	public HashMap<String, HashMap<String, TreeMap<Integer, IndexedWord>>> mIdToCoreEvText;
	public HashMap<String, HashMap<String, TreeMap<Integer, CorefChain>>> mIdToEntCorefs;
	public CoreDocument coreDoc;
	public NDArray tfidfVec;
	// m_id to srl roles (toks)
	// entity coref graph (stanford)
	// 


	public NafDoc(File f) {
		this.file = f;
		this.topic = Integer.parseInt(f.getName().split("_")[0]);
		this.toks = new HashMap<String, HashMap<String, String>>();
		this.inOrderToks = new LinkedList<String>();
		this.mentionIdToTokens = new HashMap<String, List<String>>();
		this.sentenceIdToTokens = new HashMap<String, List<String>>();
		this.actionToMentions = new HashMap<String, HashSet<String>>();
		this.tokIdToEntCorefChain = new HashMap<String, CorefChain>();
		this.evCorefGraph = GraphBuilder.undirected().<EventNode>build();
		this.sentenceToSRLParse = new HashMap<String, DEPTree>();
		this.sentenceToSRLNodes = new HashMap<String, DEPNode[]>();
		this.mIdToEvText = new HashMap<String, HashMap<String,TreeMap<Integer, DEPNode>>>();
		this.mIdToCoreEvText = new HashMap<String, HashMap<String, TreeMap<Integer, IndexedWord>>>();
		this.mIdToEntCorefs = new HashMap<String, HashMap<String, TreeMap<Integer, CorefChain>>>();
		
		List<HashMap<String, String>> tokens = this.readXML(f);
		this.parseTokens(tokens);
		if(Globals.USED_CACHED_PARSES)
			coreDoc = this.getSerializedCoreDoc();
		else
			coreDoc = this.parseDocWithCoreNLP();
		this.recordEntCorefs();
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
			String clean = TFIDF.cleanTok(tok, false, new HashSet<String>(Arrays.asList("N", "V")));
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
			String ecbTok = this.toks.get(t_id).get("text").replace(" ", "");
			if(this.file.getName().equals("24_10ecb_aug.en.naf") && t_id.equals("14")) // weird corenlp bug
				coreTok.setOriginalText(ecbTok);
			assertEquals(coreTok.originalText(), ecbTok);
			String clean = TFIDF.cleanTok(coreTok, false, new HashSet<String>());
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
					if (startElement.getName().getLocalPart().equals("token")) {
						HashMap<String,String> token = new HashMap<String, String>();
						Iterator it = startElement.getAttributes();
						while(it.hasNext()) {
							Attribute att = (Attribute)it.next();
							token.put(att.getName().toString(), att.getValue());
						}
						token.put("text", xmlEventReader.nextEvent().toString());
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
			TreeMap<Integer, CorefChain> headCorefs = new TreeMap<Integer, CorefChain>();
			TreeMap<Integer, CorefChain> depCorefs = new TreeMap<Integer, CorefChain>();
			
			for(String t_id : this.mentionIdToTokens.get(m_id)) {
				int firstIdx = Integer.parseInt(this.toks.get(this.sentenceIdToTokens.get(s_id).get(0)).get("t_id"));
				CoreLabel coreTok = coreSent.tokens().get(Integer.parseInt(this.toks.get(t_id).get("number")));
				IndexedWord headNode = semGraph.getNodeByIndex(coreTok.index());
				evHeads.put(headNode.index(), headNode);
//				assertEquals(headNode.originalText(), this.toks.get(String.valueOf(firstIdx + headNode.index() - 1)).getTextContent());
				if(this.tokIdToEntCorefChain.containsKey(String.valueOf(firstIdx + headNode.index() - 1))) {
					CorefChain chain = this.tokIdToEntCorefChain.get(String.valueOf(firstIdx + headNode.index() - 1));
					headCorefs.put(headNode.index(), chain);
				}
				for(IndexedWord depNode : semGraph.descendants(headNode)) {
					if(StringUtils.containsPunctuation(depNode.tag()))
						continue;
					evDeps.put(depNode.index(), depNode);
					if(this.tokIdToEntCorefChain.containsKey(String.valueOf(firstIdx + depNode.index() - 1))) {
						CorefChain chain = this.tokIdToEntCorefChain.get(String.valueOf(firstIdx + depNode.index() - 1));
						depCorefs.put(depNode.index(), chain);
					}
				}
			}

			this.mIdToCoreEvText.put(m_id, new HashMap<String,TreeMap<Integer, IndexedWord>>());
			this.mIdToCoreEvText.get(m_id).put("trigger", evHeads);
			this.mIdToCoreEvText.get(m_id).put("deps", evDeps);
			this.mIdToEntCorefs.put(m_id, new HashMap<String, TreeMap<Integer, CorefChain>>());
			this.mIdToEntCorefs.get(m_id).put("trigger", headCorefs);
			this.mIdToEntCorefs.get(m_id).put("deps", depCorefs);
		}
	}
	
	private CoreDocument getSerializedCoreDoc() {
		ProtobufAnnotationSerializer ser = new ProtobufAnnotationSerializer();
		Object doc = SerUtils.loadObj(Paths.get(Globals.CACHED_CORE_DOCS.toString(), this.file.getName() + ".ser").toFile());
		return new CoreDocument(ser.fromProto((CoreNLPProtos.Document)doc));
	}
	
	private void recordEntCorefs() {
		
    	for(int i : coreDoc.corefChains().keySet()) {
    		CorefChain chain = coreDoc.corefChains().get(i);
    		for(CorefMention m : chain.getMentionsInTextualOrder()) {
    			for(int idx : IntStream.range(m.headIndex - 1, m.headIndex).toArray()) {
    				List<String> tokSent = this.sentenceIdToTokens.get(String.valueOf(m.sentNum - 1));
    				int firstIdx = Integer.parseInt(this.toks.get(tokSent.get(0)).get("t_id"));
    				this.tokIdToEntCorefChain.put(String.valueOf(firstIdx + idx), chain);
    			}
    		}
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
        StringEntity requestEntity = new StringEntity(
        	    doc,
        	    ContentType.APPLICATION_JSON);

    	HttpPost postMethod = new HttpPost("http://localhost:9000/?" + UrlEscapers.urlFormParameterEscaper().escape(args));
    	postMethod.setEntity(requestEntity);
    	HttpResponse rawResponse = null;
    	ProtobufAnnotationSerializer serial = new ProtobufAnnotationSerializer();
    	Annotation coreAnnotation = null;
    	try {
			rawResponse = httpClient.execute(postMethod);
			CoreNLPProtos.Document protoDoc = CoreNLPProtos.Document.parseDelimitedFrom(rawResponse.getEntity().getContent());
			coreAnnotation = serial.fromProto(protoDoc);
			
		} catch (ClientProtocolException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
    	CoreDocument coreDoc = new CoreDocument(coreAnnotation);

		return coreDoc;
	}

	
	private void printCorefs(String type) {
		if(type.equals("clear")) {
			for(String m_id : this.mIdToEvText.keySet()) {
				String s_id = this.toks.get(this.mentionIdToTokens.get(m_id).get(0)).get("text");
				int firstIdx = Integer.parseInt(this.toks.get(this.sentenceIdToTokens.get(s_id).get(0)).get("t_id"));
				for(int i : this.mIdToEvText.get(m_id).get("trigger").navigableKeySet()) {
					DEPNode n = this.mIdToEvText.get(m_id).get("trigger").get(i);
					System.out.println(n.getWordForm()+ ": " + n.getPOSTag()+ " || " +  
										this.tokIdToEntCorefChain.get(String.valueOf(firstIdx + n.getID() - 1)));
				}
				for(int i : this.mIdToEvText.get(m_id).get("deps").navigableKeySet()) {
					DEPNode n = this.mIdToEvText.get(m_id).get("deps").get(i);
					System.out.println("\t"+n.getWordForm()+ ": " + n.getPOSTag() + " || " +  
							this.tokIdToEntCorefChain.get(String.valueOf(firstIdx + n.getID() - 1)));
				}
				System.out.println("---");
			}
		}
		else {
			for(String m_id : this.mIdToCoreEvText.keySet()) {
				for(int i : this.mIdToCoreEvText.get(m_id).get("trigger").navigableKeySet()) {
					IndexedWord n = this.mIdToCoreEvText.get(m_id).get("trigger").get(i);
					System.out.println(n.originalText()+ ": " + n.tag() + " || " +  
										this.mIdToEntCorefs.get(m_id).get("trigger").get(i));
				}
				for(int i : this.mIdToCoreEvText.get(m_id).get("deps").navigableKeySet()) {
					IndexedWord n = this.mIdToCoreEvText.get(m_id).get("deps").get(i);
					System.out.println("\t"+n.originalText() + ": " + n.tag() + " || " +  
										this.mIdToEntCorefs.get(m_id).get("deps").get(i));
				}
				System.out.println("---");
			}
		}
		
	}
}
