package common;


public class ComparableTuple extends GeneralTuple<Object,Object> implements Comparable<ComparableTuple> {

	private int first;
	private Object second;
	
	public ComparableTuple(int first, Object second) {
		super(first, second);
	}

	@Override
	public int compareTo(ComparableTuple t) {
		return Integer.compare((int)this.first, (int)t.first);
	}}
