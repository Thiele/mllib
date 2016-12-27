package nu.thiele.mllib.data;

public class DataSet {
	public double[][] x;
	public double[] y;
	
	public DataSet(){}
	
	public DataSet(double[][] x, double[] y){
		this.x = x;
		this.y = y;
	}
	
	public int size(){
		return this.x.length;
	}
}