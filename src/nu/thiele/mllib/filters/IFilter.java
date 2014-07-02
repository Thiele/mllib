package nu.thiele.mllib.filters;

import java.util.List;
import java.util.Set;

import nu.thiele.mllib.data.Data.DataEntry;

public interface IFilter {
	public static enum Filters{
		NEAREST_NEIGHBOUR_FILTER,
		NORMAL_DISTRIBUTION_FILTER
	}

	public Set<DataEntry> getRemovals(List<DataEntry> input);
}