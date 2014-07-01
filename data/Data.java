package nu.thiele.ailib.data;

import java.util.TreeMap;

public class Data {
	public static class DataEntry{
		private TreeMap<String, Object> extra;
		private double[] x;
		private Object y;
		
		public DataEntry(double[] x, Object y){
			this.extra = new TreeMap<String, Object>();
			this.x = x;
			this.y = y;
		}
		
		public void addExtra(String id, Object o){
			this.extra.put(id, o);
		}
		
		@Override
		public DataEntry clone(){
			DataEntry retval = new DataEntry(this.x.clone(), this.y);
			retval.extra = this.extra;
			return retval;
		}
		
		public Object getExtra(String s){
			return this.extra.get(s);
		}
		
		public double[] getX(){
			return this.x;
		}
		
		public Object getY(){
			return this.y;
		}
		
		public void setY(Object y){
			this.y = y;
		}
	}
}
