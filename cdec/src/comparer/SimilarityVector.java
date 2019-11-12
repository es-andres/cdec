package comparer;

public abstract class SimilarityVector {
	
	/*
	 * unless otherwise noted, all features apply to both pairs and chains
	 */
	
	//1
	public abstract double getLemmaOverlap();
	//2
	public abstract double getSynsetOverlap();
	//3
	public abstract double getSynsetDistance();
	//4
	public abstract double getParticipantOverlap();
	//5
	public abstract double getTimeOverlap();
	//6
	public abstract double getLocationOverlap();
	//7 pair
	public abstract double getDiscourseDistance();
	//8 pair
	public abstract double getSentenceDistance();
	//9
	public abstract double getWord2VecSimilarity();
	//10 chain
	public abstract double getRelativeChainStartPosition();
	//11 chain
	public abstract double getRelativeChainEndPosition();
	//12 chain
	public abstract double getRelativeSentenceStartPosition();
	//13 chain
	public abstract double getRelativeSentenceEndPosition();
	//14 chain
	public abstract double getRelativeChainSize();
	//15
	public abstract double getTemporalRelation();
	//16
	public abstract double getSts();
	//class
	public abstract boolean getCorefPrediction();

	

}
