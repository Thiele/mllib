package nu.thiele.ailib.utils;

import java.util.List;

import nu.thiele.ailib.data.Data.DataEntry;

public class Statistics {
	
	
	public static double mean(List<DataEntry> data, int index){
		float retval = 0;
		for(DataEntry entry : data){
			retval += entry.getX()[index];
		}
		return retval/((float)data.size());
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
