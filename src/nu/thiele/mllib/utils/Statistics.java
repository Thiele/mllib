package nu.thiele.mllib.utils;

import java.util.ArrayList;
import java.util.List;

import nu.thiele.mllib.data.Data.DataEntry;

public class Statistics {
	
	public static double max(ArrayList<Double> t){
		double maks = Double.MIN_VALUE;
		for(double d : t){
			if(d > maks) maks = d;
		}
		return maks;
	}
	
	
	public static double mean(List<DataEntry> data, int index){
		float retval = 0;
		for(DataEntry entry : data){
			retval += entry.getX()[index];
		}
		return retval/((float)data.size());
	}
	
	public static double min(ArrayList<Double> t){
		double min = Double.MAX_VALUE;
		for(double d : t){
			if(d < min) min = d;
		}
		return min;
	}
	
	public static double normalDistributionProbability(double x, double mean, double var){
		double exp = (x-mean)*(x-mean)/(2*var);
		double base = 1.0/(Math.sqrt(2*Math.PI*var));
		double retur = base*Math.pow(Math.E, -exp);
		return retur;
	}
	
	public static ArrayList<Double> rescale(ArrayList<Double> t){
		return rescale(t, min(t), max(t));
	}
	
	public static ArrayList<Double> rescale(ArrayList<Double> t, double min, double maks){
		maks = maks-min;
		ArrayList<Double> retur = new ArrayList<Double>();
		for(double d : t){
			retur.add((d-min)/maks);
		}
		return retur;
	}
	
	public static double standardDeviation(List<DataEntry> data, int index, double mean){
		return (float) Math.sqrt(variance(data,index,mean));
	}
	
	public static double standardDeviation(List<Double> data, double mean){
		return (float) Math.sqrt(variance(data,mean));
	}
	
	public static double standardDeviation(List<DataEntry> data, int index){
		return standardDeviation(data,index,mean(data,index));
	}
	
	public static double variance(List<DataEntry> data, int index, double mean){
		float retval = 0;
		for(DataEntry entry : data){
			double n = (entry.getX()[index]-mean);
			retval += n*n;
		}
		return retval/((float) (data.size()-1));
	}
	
	public static double variance(List<Double> data, double mean){
		float retval = 0;
		for(double entry : data){
			double n = (entry-mean);
			retval += n*n;
		}
		return retval/((float) (data.size()-1));
	}
	
	public static double variance(List<DataEntry> data, int index){
		return variance(data,index,mean(data,index));
	}
	
	public static double mean(List<Double> d){
		double sum = 0;
		for(double dd : d) sum += dd;
		return sum / ((double)d.size());
	}
	
	public static double variance(List<Double> data){
		double retval = 0;
		double mean = mean(data);
		for(double d : data){
			double n = mean-d;
			retval += n*n;
		}
		return retval/((double)(data.size()-1));
	}
}
