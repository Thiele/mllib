package nu.thiele.mllib.filters;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import nu.thiele.mllib.classifiers.NearestNeighbour;
import nu.thiele.mllib.data.Data.DataEntry;


public class NearestNeighbourFilter implements IFilter{
	private int neighbours;
	private int threshold;
	public NearestNeighbourFilter(int neighbours, int threshold){
		this.neighbours = neighbours;
		this.threshold = threshold;
	}
	
	@Override
	public Set<DataEntry> getRemovals(List<DataEntry> input) {		
		HashSet<DataEntry> toRemove = new HashSet<DataEntry>();
		
		//First, check to see if there are even enough neighbours to perform filtering
		if(input.size() < this.neighbours) throw new IllegalArgumentException();
		
		for(DataEntry tse : input){
			DataEntry[] nearest = NearestNeighbour.getKNearestNeighbours(input, tse.getX(), this.neighbours+1); //+1, since it looks at k nearest, excluding itself, which will always be found 
			
			int correct = -1; //Start at -1, since the match itself will always be closest
			for(DataEntry near : nearest){
				if(near != null && near.getY() == tse.getY()) correct++;
			}
			if(correct<= this.threshold) toRemove.add(tse);
		}
		return toRemove;

	}
}