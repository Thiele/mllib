package nu.thiele.mllib.kernelestimators;

import java.util.ArrayList;

import nu.thiele.mllib.utils.Statistics;

public class GaussianKernelEstimator implements IEstimator{
	private ArrayList<Double> ar = new ArrayList<Double>();
	private double sum = 0.0;
	public void addValue(double x) {
		sum = sum+x;
		this.ar.add(x);
	}
	public double probability(double x) {
		double n = this.ar.size();
		double gennemsnit = this.sum/n;
		double h = Statistics.standardDeviation(ar, gennemsnit)*Math.pow(80/(1.1283791670955125739*n), -1/5);
		double total = 0.0;
		//Beregn kurver
		for(double d : this.ar){
			total = total+(new Curve(d, h)).probability(x);
		}
		return total/(h*n);
	}
	
	private class Curve {
		private double h;
		private double x;
		public Curve(double x, double h){
			this.h = h;
			this.x = x;
		}
		
		public double probability(double input){
			double dat = (input-this.x)/h;
			double base = 1.0/(Math.sqrt(2*Math.PI));
			return base*Math.pow(Math.E, -0.5*dat*dat);
		}
	}
}
