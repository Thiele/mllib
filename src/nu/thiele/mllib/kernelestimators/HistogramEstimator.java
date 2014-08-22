package nu.thiele.mllib.kernelestimators;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.NavigableSet;
import java.util.TreeSet;

import nu.thiele.mllib.utils.Statistics;

public class HistogramEstimator extends KernelEstimator{
	private ArrayList<Double> ar;
	private double total = 0;
	public HistogramEstimator(){
		this.ar = new ArrayList<Double>();
	}
	
	@Override
	protected void addValueToEstimator(double x) {
		this.ar.add(x);
		this.total = this.total+x;
	}
	
	@Override
	protected double estimatedProbability(Object[] parameters, double x) {
		double h = 0;
		double k = 0;
		//Scott's choice, hvis man får lyst
		//h = 3.5*Statistics.standardDeviation(this.ar, this.total/((double)this.ar.size()))/(Math.pow(this.ar.size(), 1.0/3.0));
		//k = (Statistics.max(this.ar)-Statistics.min(this.ar))/h;
		if(k == 0){
			k = Math.ceil(Math.log(this.ar.size())/Math.log(2.0)+1); //Sturges formel til standard
		}
		if(h == 0){
			h = (Statistics.max(this.ar)-Statistics.min(this.ar))/k;
		}
		HashMap<Double,Double> v = new HashMap<Double,Double>();
		NavigableSet<Double> t = new TreeSet<Double>(v.keySet());
		for(double i = -k; i <= k+10; i++){
			t.add(i*h);
			v.put(i*h, 0.0);
		}
		//Indlæs samtlige this.ar værdier
		for(double d : this.ar){
			double lavere = t.floor(d) == null ? Double.MIN_VALUE : t.floor(d) ;
			double højere = t.ceiling(d) == null ? Double.MAX_VALUE : t.ceiling(d);
			if(lavere < højere){
				try{
					v.put(lavere, v.get(lavere)+1.0);	
				}catch(Exception e){v.put(lavere, 1.0);} //Hvis nu der gik noget galt
			}
			else{
				try{
					v.put(højere, v.get(højere)+1.0);	
				}
				catch(Exception e){v.put(højere, 1.0);}
			}
		}		
		//Find nærmeste entry
		double lavere = t.floor(x) == null ? Double.MIN_VALUE : t.floor(x) ;
		double højere = t.ceiling(x) == null ? Double.MAX_VALUE : t.ceiling(x);
		double antal = 0.0;
		try{
			if(Math.abs(lavere-x) < Math.abs(højere-x)){ //Hvis begge er sat
				antal = v.get(lavere);
			}
			else{
				antal = v.get(højere);
			}
		}
		catch(Exception e){//Returnerer standard normal, hvis der sker fejl
			double var = Statistics.variance(this.ar, this.total/((double)this.ar.size()));
			return Statistics.normalDistributionProbability(x, this.total/((double)this.ar.size()), var);
		}
		
		return antal/((double)this.ar.size());
	}

	@Override
	protected Object[] getEstimatorParametersForEstimation() {
		// TODO Auto-generated method stub
		return null;
	}
}