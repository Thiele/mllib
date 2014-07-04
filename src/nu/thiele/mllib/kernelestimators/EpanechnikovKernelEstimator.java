package nu.thiele.mllib.kernelestimators;

import java.util.ArrayList;

import nu.thiele.mllib.utils.Statistics;

public class EpanechnikovKernelEstimator extends KernelEstimator{
	private ArrayList<Double> ar = new ArrayList<Double>();
	private double sum = 0.0;
	
	@Override
	protected void addValueToEstimator(double x) {
		sum = sum+x;
		this.ar.add(x);
	}
	@Override
	protected double estimatedProbability(double[] parameters, double x) {
		double v = 0.5;
		if(x > Statistics.max(this.ar)) return 0.000001;
		else if(x < Statistics.min(this.ar)) return 0.000001;
		
		ArrayList<Double> brug = new ArrayList<Double>(this.ar);
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
	protected double[] getEstimatorParametersForEstimation() {
		// TODO Auto-generated method stub
		return null;
	}
}
