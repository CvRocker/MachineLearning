package knn;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * @version 1.0
 * @author liuzhi
 * @category machine learning, classification
 * @since 2015.9.17
 */
public class KNN {
	//k is the number of nearest neighbor. The default value is 1.
	private int k = 1;
	//the train data which key is one element label array and value is 784 elements image array...
	private Map<int[], int[]> trainData = null;
	
	public void setK(int k){
		this.k = k;
	}
	
	public int getK(){
		return this.k;
	}
	/**
	 * create a KNN class with train data file path.
	 * @param path
	 * @throws Exception
	 */
	public KNN(String path) throws Exception {
		loadTrainData(path);
	}
	
	/**
	 * 
	 * @param path
	 * @param k the set number of nearest neighbor.
	 * @throws Exception
	 */
	public KNN(String path, int k) throws Exception {
		this(path);
		this.k = k;
	}
	
	/**
	 * Load train data automatically...
	 * @param path
	 * @throws Exception
	 */
	protected Map<int[], int[]> loadTrainData(String path) throws Exception{
		File file = new File(path);
		if (!file.exists()){
			throw new Exception("the train data file is not exists!!!");
		}
		
		try(ObjectInputStream in = new ObjectInputStream(new FileInputStream(file))){
			trainData = (Map<int[], int[]>)in.readObject();
		}
		Map<int[], int[]> result = trainData;
		return result;
	}
	
	public Map<int[], int[]> loadData(String path) throws Exception{
		return loadTrainData(path);
	}
	
	public void classificationAndTestPrecision(Map<int[], int[]> testData) throws Exception{
		if (testData == null || testData.size() == 0){
			throw new Exception("Test data is null, please assure test data won't be null!!");
		}
		int right = 0, len = testData.size();
		for (Map.Entry<int[], int[]> entry : testData.entrySet()){
			//Get the true label of a image
			int trueLabel = entry.getKey()[0];
			//Get the image features array that will be tested.
			int[] features = entry.getValue();
			//Compute the current image distances with all the train data's image.
			Distance[] distancesK = computeDistance(features);
			Map<Integer, Integer> collect = new HashMap<>();
			for (int i = 0; i < k; i++){
				if (collect.size() == 0){
					collect.put(distancesK[i].label, 1);
				} else {
					int label = distancesK[i].label;
					if (collect.containsKey(label)){
						int time = collect.get(label);
						time++;
						collect.put(label, time);
						continue;
					}
					collect.put(label, 1);
				}
			}
			int max = 0, label = 0;
			for (Map.Entry<Integer, Integer> record : collect.entrySet()){
				int rlabel = record.getKey(), times = record.getValue();
				if (max <= times){
					max = times;
					label = rlabel;
				}
			}
			
			//judge whether the predictive label equals true label.
			if (label == trueLabel){
				right++;
				System.out.println(right+"/"+len);
			} 
			System.gc();
		}
		
		System.out.println("The ratio of precision: " + ((double) right / (double)len));
	}
	
	public Distance[] computeDistance(int[] values){
		double distance = 0; int featureLen = 28 * 28, count = 0;
		Distance[] distances = new Distance[trainData.size()];
		// compute the distance between given test image and all images in train data.
		for (Map.Entry<int[], int[]> entry : trainData.entrySet()){
			int[] trainLabel = entry.getKey();
			int[] trainFeatures = entry.getValue();
			//System.out.println("test: "+values.length + ", train: "+ trainFeatures.length);
			//使用欧氏距离计算测试数据与训练数据之间的距离
			for (int i = 0; i < featureLen; i++){
				if (values[i] == 0 && trainFeatures[i] == 0){
					continue;
				}
				distance +=Math.pow(Math.abs(values[i] - trainFeatures[i]), 2);
			}
			distance =  Math.sqrt(distance);
			distances[count] = new Distance(trainLabel[0], distance);
			count++;
			distance = 0;
		}
		Arrays.sort(distances, new Comparator<Distance>(){
			@Override
			public int compare(Distance d1, Distance d2) {
				return d1.distance <= d2.distance ? -1 : 1;
			}
		});
		distances = Arrays.copyOfRange(distances, 0, k);
		/*for (int i = 0; i < distances.length; i++){
			System.out.print(distances[i].label + " = " + distances[i].distance + " ");
			if (i % 28 == 0 && i != 0){
				System.out.println();
			}
		}*/
		System.gc();
		return distances;
	}
	
}
