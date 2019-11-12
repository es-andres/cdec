package common;

public class GeneralTuple<X, Y>{ 
	
  public final X first; 
  public final Y second; 
  
  public GeneralTuple(X x, Y y) 
  { 
    this.first = x; 
    this.second = y; 
  } 
  
  @Override
  public String toString() {
	  return "first: " + this.first + ",second: " + this.second;
  }
  
} 
