package nu.thiele.mllib.data;

import java.util.LinkedList;
import java.util.List;

import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.utils.Statistics;

public class Normalizer {
	private double[] maxs;
	private double[] means;
	private double[] mins;
	private double[] stdds;
	private Method method;
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public Normalizer(List v, Method m){
		this.method = m;
		List<double[]> vals = null;
		if(v == null || v.size() == 0) return; //Parameter check
		if(v.get(0) instanceof float[]){ //Why can't I match on instanceof List<DataEntry>? Annoying...
			vals = v;
		}
		else if(v.get(0) instanceof DataEntry){
			vals = new LinkedList<double[]>();
			for(DataEntry de : (List<DataEntry>) v){
				vals.add(de.getX());
			}
		}
		else return; //Cannot be used
		
		
		switch(m){
		case RESCALING:
			//And find max and min for each feature
			this.maxs = new double[vals.get(0).length];
			this.mins = new double[vals.get(0).length];
			for(int i = 0; i < this.mins.length; i++){ //Fill with dummy values to be overwritten
				this.mins[i] = Float.MAX_VALUE;
				this.maxs[i] = Float.MIN_VALUE;
			}
			for(double[] xes : vals){
				for(int i = 0; i < xes.length; i++){
					if(xes[i] > this.maxs[i]) this.maxs[i] = xes[i];
					else if(xes[i] < this.mins[i]) this.mins[i] = xes[i];
				}
			}
		break;
		case NORMALIZATION:
			this.means = new double[vals.get(0).length];
			this.stdds = new double[vals.get(0).length];
			for(int i = 0; i < this.means.length; i++){
				if(v.get(0) instanceof DataEntry){
					this.means[i] = Statistics.mean(v, i);
					this.stdds[i] = Statistics.standardDeviation(v, i, this.means[i]);
				}
			}
		break;
		}
	}
	
	public void apply(List<DataEntry> data){
		for(DataEntry entry : data){
			apply(entry);
		}
	}
	
	public void apply(DataEntry entry){
		this.apply(entry.getX());
	}
	
	public void apply(double[] data){
		for(int i = 0; i < data.length; i++){
			switch(this.method){
			case NORMALIZATION:
				data[i] = (data[i]-this.means[i])/(this.stdds[i]);
				break;
			case RESCALING:
				data[i] = (data[i]-this.mins[i])/(this.maxs[i]-this.mins[i]);
				break;
			}
		}
	}
	
	public void printFunctions(){
		switch(this.method){
		case RESCALING:
			for(int i = 0; i < this.mins.length; i++){
				System.out.println("For feature "+i+", the rescaling function is: f(x) = (x-"+this.mins[i]+")/("+this.maxs[i]+"-"+this.mins[i]+")");
			}
		break;
		case NORMALIZATION:
			for(int i = 0; i < this.stdds.length; i++){
				System.out.println("For feature "+i+", the normalization function is: f(x) = (x-"+this.means[i]+")/("+this.stdds[i]+")");
			}
			break;
		}
	}
	
	public static enum Method{
		NORMALIZATION,RESCALING
	}
}
