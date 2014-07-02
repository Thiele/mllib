package nu.thiele.mllib.filters;

import java.util.HashSet;
import java.util.List;
import nu.thiele.mllib.data.Data.DataEntry;


public class FilterHandler {
	public static class FilterMode{
		public static final int REMOVE_ALL = 0; //Just remove anything
		public static final int REMOVE_UNANIMOUS = 1; //Remove entries that all filters mark as outliers
	}

	private int filterMode;
	private IFilter[] filters;
	public FilterHandler(int filterMode, IFilter[] filters){
		this.filterMode = filterMode;
		this.filters = filters;
	}

	public void applyFilters(List<DataEntry> input){
		if(this.filters == null || this.filters.length == 0 || input == null || input.size() == 0) return;
		HashSet<DataEntry> toRemove = new HashSet<DataEntry>();
		boolean firstRun = true;
		for(IFilter f : this.filters){
			HashSet<DataEntry> vals = (HashSet<DataEntry>) f.getRemovals(input);
			if(this.filterMode == FilterMode.REMOVE_ALL) toRemove.addAll(vals);
			else if(this.filterMode == FilterMode.REMOVE_UNANIMOUS){
				if(firstRun){
					toRemove.addAll(vals);
				}
				else{
					HashSet<DataEntry> newSet = new HashSet<DataEntry>();
					for(DataEntry a : toRemove){
						if(vals.contains(a)) newSet.add(a);
					}
					toRemove = newSet;
				}
			}
			firstRun = false;
		}
		input.removeAll(toRemove);
	}
}