package knn;

public class Classifier {

	//the label of train that is one of k nearest neighbor..
	public int label = 0;
	//The times of label occurrence.
	public int times = 0;
	
	public Classifier(int label, int times){
		this.label = label;
		this.times = times;
	}
}
