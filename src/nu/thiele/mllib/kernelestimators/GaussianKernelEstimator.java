package nu.thiele.mllib.kernelestimators;

import java.util.ArrayList;

import nu.thiele.mllib.utils.Statistics;

public class GaussianKernelEstimator extends KernelEstimator implements KernelEstimatorFactory{
	private ArrayList<Double> ar = new ArrayList<Double>();
	private double sum = 0.0;
	
	@Override
	protected void addValueToEstimator(double x) {
		sum = sum+x;
		this.ar.add(x);
	}
	
	@Override
	protected double estimatedProbability(Object[] parameters, double x) {
		double h = ((double)parameters[1])*Math.pow(80/(1.1283791670955125739*this.ar.size()), -1/5);
		double total = 0.0;
		//Beregn kurver
		for(double d : this.ar){
			total = total+(new Curve(d, h)).probability(x);
		}
		return total/(h*this.ar.size());
	}
	
	@Override
	protected Object[] getEstimatorParametersForEstimation() {
		double avg = this.sum/((double)this.ar.size());
		double std = Statistics.standardDeviation(this.ar, avg);
		return new Object[]{avg, std};
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

	@Override
	public KernelEstimator newInstance() {
		// TODO Auto-generated method stub
		return new GaussianKernelEstimator();
	}
}
