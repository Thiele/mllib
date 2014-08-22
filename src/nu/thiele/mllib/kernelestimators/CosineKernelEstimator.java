package nu.thiele.mllib.kernelestimators;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import nu.thiele.mllib.utils.Statistics;

public class CosineKernelEstimator extends KernelEstimator{
	private ArrayList<Double> ar = new ArrayList<Double>();
	private double sum = 0.0;
	
	@Override
	protected void addValueToEstimator(double x) {
		sum = sum+x;
		this.ar.add(x);
	}
	@Override
	protected double estimatedProbability(Object[] parameters, double x) {
		double v = 0.5;
		if(x < (double) parameters[0]) return 0.000001;
		if(x > (double) parameters[1]) return 0.000001;
		
		//Create copy of array and rescale it to lie between 0 and 1
		List<Double> brug = new LinkedList<Double>();
		for(Double d : this.ar) brug.add(d);
		brug.add(0, x);
		Statistics.rescale(brug);
		brug.remove(x);
		
		double n = this.ar.size();
		double gennemsnit = this.sum/n;
		double k = Math.PI*Math.PI-8;
		double h = Statistics.standardDeviation(ar, gennemsnit)*Math.pow(Math.PI,0.65)*Math.pow(6*k*k*n, 1/5);
		double total = 0.0;
		//Calculate curves
		for(double d : brug){
			total = total+(new Curve(d, h)).probability(v);
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
			double dat = (input-this.x)/this.h;
			return (Math.PI/4.0)*Math.cos(Math.PI*dat/2.0);
		}
	}

	@Override
	protected Object[] getEstimatorParametersForEstimation() {
		double min = Statistics.min(this.ar);
		double max = Statistics.max(this.ar);
		return new Object[]{min,max};
	}
}
