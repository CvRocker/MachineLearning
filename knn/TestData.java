package knn;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import com.sun.org.apache.bcel.internal.util.ByteSequence;

public class TestData {

	public static void main(String[] args) throws ClassNotFoundException {
		/*byte val = -32;
		System.out.println(val&0xff);
		ByteBuffer buffer = ByteBuffer.allocate(2).order(ByteOrder.BIG_ENDIAN);
		buffer.put(new byte[]{0x00, -32});
		buffer.flip();
		System.out.println(buffer.getShort());*/
		String labelfile = "/home/leo/testdata/testDataLabel";
		String datafile = "/home/leo/testdata/testData";
		String[] pathes = {labelfile, datafile};
		int[] starts = {8, 15};
		int len = 5000;
		int[] labels = null;
		int[][] images = null;
		
		try {
				System.out.println("file: "+pathes[0]+", start: "+starts[0]);
				labels = readLabel(pathes[0], starts[0], len);
				images = readData(pathes[1], starts[1], len);
				Map<int[], int[]> data = new HashMap<>(len);
				for (int i = 0; i < len; i++){
					int[] label = new int[1];
					label[0] = labels[i];
					data.put(label, images[i]);
				}
				ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(new File("data/testdata_5000")));
				out.writeObject(data);
				System.out.println(data.size());
			/*
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(new File("data/traindata")));
			Map<int[], int[]> testdata = (Map<int[], int[]>)in.readObject();
			for (Map.Entry<int[], int[]> entry : testdata.entrySet()){
				int label = entry.getKey()[0];
				int[] image = entry.getValue();
				System.out.println("label: " + label);
				for (int i = 0; i < 784; i++){
					System.out.print(image[i] + "\t");
					if (i % 28 == 0 && i != 0){
						System.out.println();
					}
				}
				System.out.println("\none image finished!");
				TimeUnit.SECONDS.sleep(5);
			}
			System.out.println("read data: " + testdata.size());
		*/
		} catch (IOException e) {
			e.printStackTrace();
		} catch(InterruptedException e){
			e.printStackTrace();
		}
	}

	public static int[] readLabel(String path, int start, int size) throws FileNotFoundException, IOException{
		//List<Integer> data = new ArrayList<>(size);
		int[] label = new int[size];
		try(FileInputStream input = new FileInputStream(path);
				FileChannel channel = input.getChannel();){
				ByteBuffer buffer = ByteBuffer.allocate(1024).order(ByteOrder.BIG_ENDIAN);
				int i = start, j = 0;
				channel.read(buffer);
				buffer.flip();
				for (; i < buffer.limit(); i++){
					Byte b = buffer.get(i);
					if (b < 0){
						int val = b & 0xff;
						label[j] = (val);
						//System.out.println(j +", data: " + label[j]);
						j++; 
						continue;
					}
					label[j] = (b.intValue());
					//System.out.println(j +", data: " + label[j]);
					j++;
				}
				buffer.clear();
				//System.out.println("label position: " + labelBuffer.position()+", data position: " + dataBuffer.position());
				while ((channel.read(buffer)) != -1){
					buffer.flip();
					for (i = 0; i < buffer.limit(); i++){
						Byte b = buffer.get(i);
						if (b < 0){
							int val = b & 0xff;
							//data.add(val);
							label[j] = (val);
							//System.out.println(j +", data: " + label[j]);
							j++;
							if (j == size){
								return label;
							}
							continue;
						}
						//data.add(b.intValue());
						label[j] = (b.intValue());
						//System.out.println(j +", data: " + label[j]);
						j++;
						if (j == size){
							return label;
						}
					}
					buffer.clear();
				}
				//System.out.println("total records: " + j);
				return label;
			}
		} 
	
	public static int[][] readData(String path, int start, int size) throws FileNotFoundException, IOException, InterruptedException{
		int[][] data = new int[size][];
		int imageLen = 28 * 28;
		int[] image = new int[imageLen];
		try(FileInputStream input = new FileInputStream(path);){
			long num = input.skip(start);
			System.out.println("skip bytes: " + num);
			FileChannel channel = input.getChannel();
			ByteBuffer buffer = ByteBuffer.allocate(1024).order(ByteOrder.BIG_ENDIAN);
			int j = 0, pixel = 0;
			while ((channel.read(buffer)) != -1){
				buffer.flip();
				for (int i = 0; i < buffer.limit(); i++){
					byte b = buffer.get(i);
					if (b < 0){
						int val = b & 0xff;
						image[pixel] = (val);
						pixel++;
						if (pixel == imageLen){
							data[j] = image;
							j++;
							if (j == size){
								return data;
							}
							//if pixel equals 784, we need create a new array for another image;
							image = new int[imageLen];
							pixel = 0;
						}
						continue;
					}
					image[pixel] = b;
					pixel++;
					if (pixel == imageLen){
						data[j] = image;
						j++;
						if (j == size){
							return data;
						}
						//if pixel equals 784, we need create a new array for another image;
						image = new int[imageLen];
						pixel = 0;
					}
				}
				buffer.clear();
			}
			System.out.println("totoal records : " + j);
		}
		return data;
	}
}
