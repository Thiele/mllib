package nu.thiele.mllib.kernelestimators;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import nu.thiele.mllib.utils.Statistics;

public class TriweightKernelEstimator extends KernelEstimator{
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
		
		List<Double> brug = new LinkedList<Double>(this.ar);
		brug.add(0, x);
		brug = Statistics.rescale(brug);
		v = brug.get(0);
		brug.remove(0);
		
		
		double n = this.ar.size();
		double gennemsnit = this.sum/n;
		double h = Statistics.standardDeviation(ar, gennemsnit)*Math.pow(50400/(1.1283791670955125739*143*n), 1/5);
		double total = 0.0;
		//Beregn kurver
		for(double d : brug){
			double dat = (v-d)/h;
			total = total+(35.0/32.0)*Math.pow((1-dat*dat),3.0);
		}
		return total/(h*n);
	}
	
	@Override
	protected Object[] getEstimatorParametersForEstimation() {
		double min = Statistics.min(this.ar);
		double max = Statistics.max(this.ar);
		return new Object[]{min,max};
	}
}
