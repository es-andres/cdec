package common;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.StringWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.io.IOUtils;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import ecb_utils.ECBDoc;
import ecb_utils.ECBWrapper;
import edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer;
import me.tongfei.progressbar.ProgressBar;

/**
 * Utility for serializing objects
 * @author acrem003
 *asCharSink(to, charset).write(from)
 */
public class SerUtils {
	private static final Logger LOGGER = Logger.getLogger(SerUtils.class.getName());
	
	/**
	 * Generate a corenlp CoreDoc object for all ECB+ documents and serialize them
	 */
	public static void cacheSemanticParses() {
		Globals.CACHED_CORE_DOCS.toFile().mkdir();
		startCoreNLPServer();
		
		Globals.USED_CACHED_PARSES = false;
		List<Integer> topics = IntStream.rangeClosed(1, 45).boxed().collect(Collectors.toList());
	    List<File> allFiles = ECBWrapper.getFilesFromTopics(topics);

		LOGGER.info("loading all files...");
		for(File f : ProgressBar.wrap(allFiles, "Load all files")) {
			File serExists = Paths.get(Globals.CACHED_CORE_DOCS.toString(), f.getName() + ".ser").toFile();
			if(serExists.exists())
				continue;
			ECBDoc doc = new ECBDoc(f);
			ProtobufAnnotationSerializer ser = new ProtobufAnnotationSerializer();
			serializeObj(ser.toProto(doc.coreDoc.annotation()), Paths.get(Globals.CACHED_CORE_DOCS.toString(), f.getName() + ".ser").toFile());
		}
		stopCoreNLPServer();
		Globals.USED_CACHED_PARSES = true;
		LOGGER.info("cache built ");
	}
	
	private static void startCoreNLPServer() {
		Path nlpDir = null;
		try {
			FileReader reader = new FileReader(Globals.EXTERNAL_PATHS.toFile());
			JSONParser jsonParser = new JSONParser();
			JSONObject json = (JSONObject)jsonParser.parse(reader);
			nlpDir = Paths.get((String) json.get("core_nlp"));
			ProcessBuilder procBuilder = new ProcessBuilder("nohup", "java", "-mx8g", "-cp", "*",
											"edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
											"-port", "9000", "-timeout", "1500000");
			procBuilder.directory(nlpDir.toFile());
			Process proc = procBuilder.start();

		    final StringWriter writer = new StringWriter();
		    new Thread(new Runnable() {
		        public void run() {
		            try {
						IOUtils.copy(proc.getErrorStream(), writer);
					    final int exitValue = proc.waitFor();
					    final String processOutput = writer.toString();
						
					} catch (IOException | InterruptedException e) {
						e.printStackTrace();
					}
		        }
		    }).start();
		} catch (IOException| ParseException e) {
			e.printStackTrace();
		} 
	}
	
	private static void stopCoreNLPServer() {
		try {
			/*
			 * 	apparently this key at corenlp.shutdown is the nice way to shutdown corenlp but it's being
			 * 	annoying so just killing the port
			 */
//			wget "localhost:9000/shutdown?key=`cat /tmp/corenlp.shutdown`" -O -
//		    List<String> shutdown = Files.readAllLines(Paths.get("/tmp/corenlp.shutdown"));
//		    String key = shutdown.get(0);
//			System.out.println(System.getProperty("java.io.tmpdir"));
//			fuser -k 9000/tcp
//			Process proc = new ProcessBuilder("wget", "localhost:9000/shutdown?key=" + key,
//											   "-O", "-").start();
			
			/*
			 * kill the port
			 */
			Process proc = new ProcessBuilder("fuser", "-k", "9000/tcp").start();

		} catch (IOException e) {
			e.printStackTrace();
		} 
	}
	
	/**
	 * Serialize an object to a location
	 * @param o: object to serialize
	 * @param loc: location
	 */
	public static void serializeObj(Object o, File loc) {
		try {
			FileOutputStream fos = new FileOutputStream(loc);
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(o);
			oos.close();
			fos.close();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
	/**
	 * Load a previously serialized object
	 * @param path: where the serialized object is stored
	 * @return
	 */
	public static Object loadObj(File path) {
		
		Object obj = null;
		try {
			// Reading the object from a file
			FileInputStream file = new FileInputStream(path);
			ObjectInputStream in = new ObjectInputStream(file);

			// Method for deserialization of object
			obj = in.readObject();

			in.close();
			file.close();
		}

		catch (IOException | ClassNotFoundException ex) {
			System.out.println(ex);
		}
		
		return obj;
  
    } 

}
