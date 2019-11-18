package common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;

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

}
