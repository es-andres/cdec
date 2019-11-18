package common;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ecb_utils.ECBDoc;
import ecb_utils.ECBWrapper;
import edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer;
import me.tongfei.progressbar.ProgressBar;

public class SerUtils {
	private static final Logger LOGGER = Logger.getLogger(SerUtils.class.getName());
	
	public static void cacheSemanticParses() {
		Globals.USED_CACHED_PARSES = false;
		List<Integer> topics = IntStream.rangeClosed(1, 45).boxed().collect(Collectors.toList());
	    List<File> allFiles = ECBWrapper.getFilesFromTopics(topics);

		Globals.nafDocs = new HashMap<String, ECBDoc>();
		LOGGER.info("loading all files...");
		for(File f : ProgressBar.wrap(allFiles, "Load all files")) {
			ECBDoc doc = new ECBDoc(f);
			ProtobufAnnotationSerializer ser = new ProtobufAnnotationSerializer();
			serializeObj(ser.toProto(doc.coreDoc.annotation()), Paths.get(Globals.CACHED_CORE_DOCS.toString(), f.getName() + ".ser").toFile());
			serializeObj(doc.sentenceToSRLNodes, Paths.get(Globals.CACHED_SRL.toString(), f.getName() + ".ser").toFile());
		}

		LOGGER.info("cache built ");
		System.exit(0);
		
	}
	
	public static void serializeObj(Object o, File loc) {
		try {
			FileOutputStream fos = new FileOutputStream(loc);
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(o); // updated with the creation of each NafDoc
			oos.close();
			fos.close();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
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

		catch (IOException ex) {
			System.out.println(ex);
		}

		catch (ClassNotFoundException ex) {
			System.out.println(ex);
		}
		
		return obj;
  
    } 

}
