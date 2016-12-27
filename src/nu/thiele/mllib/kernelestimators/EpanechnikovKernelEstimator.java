package nu.thiele.mllib.kernelestimators;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import nu.thiele.mllib.utils.Statistics;

public class EpanechnikovKernelEstimator extends KernelEstimator implements KernelEstimatorFactory{
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
		
		/*
		 * Create a copy
		 */
		List<Double> brug = new LinkedList<Double>();
		for(double d : this.ar) brug.add(d);
		
		/*
		 * Rescale it
		 */
		
		brug.add(0, x);
		brug = Statistics.rescale(brug);
		v = brug.get(0);
		brug.remove(0);
		
		
		double n = this.ar.size();
		double gennemsnit = this.sum/n;
		double h = Statistics.standardDeviation(ar, gennemsnit)*Math.pow(80/(1.1283791670955125739*n), 1/5);
		double total = 0.0;
		//Beregn kurver
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
			return (1-dat*dat)*0.75;
		}
	}

	@Override
	protected Object[] getEstimatorParametersForEstimation() {
		double min = Statistics.min(this.ar);
		double max = Statistics.max(this.ar);
		return new Object[]{min,max};
	}
	@Override
	public KernelEstimator newInstance() {
		// TODO Auto-generated method stub
		return new EpanechnikovKernelEstimator();
	}
}
