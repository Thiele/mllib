package nu.thiele.mllib.filters;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import nu.thiele.mllib.data.Data.DataEntry;
import nu.thiele.mllib.utils.Statistics;

public class NormalDistributionFilter implements IFilter{
	private float threshold;
	public NormalDistributionFilter(float threshold){
		this.threshold = threshold;
	}

	@Override
	public Set<DataEntry> getRemovals(List<DataEntry> inputlist) {
		HashSet<DataEntry> toRemove = new HashSet<DataEntry>();
		//Check input
		if(inputlist == null || inputlist.size() == 0) return toRemove;
		//Start by finding 
		HashSet<Object> ys = new HashSet<Object>();
		for(DataEntry d : inputlist){
			ys.add(d.getY());
		}
		for(Object o : ys){
			List<DataEntry> input = new LinkedList<DataEntry>();
			for(DataEntry tse : inputlist){
				if(tse.getY() == o) input.add(tse);
			}
			//And now do stuff with the list
			if(input == null || input.size() == 0) return toRemove;
			double[] avgs = new double[input.get(0).getX().length];
			double[] stds = new double[input.get(0).getX().length];
			ArrayList<LinkedList<Double>> vals = new ArrayList<LinkedList<Double>>();
			for(int i = 0; i < input.get(0).getX().length; i++){
				vals.add(new LinkedList<Double>());
			}
			for(DataEntry tse : input){
				for(int i = 0; i < tse.getX().length; i++){
					vals.get(i).add(tse.getX()[i]);
				}
			}
			for(int i = 0; i < input.get(0).getX().length; i++){
				avgs[i] = Statistics.mean(vals.get(i));
				stds[i] = Statistics.standardDeviation(vals.get(i), avgs[i]);
			}
			for(DataEntry tse : input){
				for(int i = 0; i < tse.getX().length; i++){
					 if(Math.abs(avgs[i]-tse.getX()[i]) > this.threshold*stds[i]){
						 toRemove.add(tse);
						 break;
					 }
				}
			}
		}
		return toRemove;
	}
}