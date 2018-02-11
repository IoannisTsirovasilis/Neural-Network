package neuralNetwork;

import java.util.Random;
import java.util.ArrayList;

public class Network {
	protected Layer[] layers;
	protected int L; // index of output layer
	protected double learningRate = 0.5; // IMPORTANT
	protected int N = 10000; // size of training set
	protected int m = 10; // size of mini-batch
	
	public Network(int[] neurons) {		
		layers = new Layer[neurons.length];
		L = layers.length-1;
		for (int i = 0; i < layers.length; i++) {
			layers[i] = new Layer(neurons[i]);			
		}
		for (int i = 1; i < layers.length; i++) {
			layers[i].previousLayer = layers[i-1];
			connectLayers(layers[i]);
		}
		startTraining();
	}
	
	private void connectLayers(Layer l) {
		Random rand = new Random();
		if (l.previousLayer != null) {
			l.conns = new Conn[l.neurons.length][l.previousLayer.neurons.length];
			for (int j = 0; j < l.neurons.length; j++) {
				for (int k = 0; k < l.previousLayer.neurons.length; k++) {
					double r = rand.nextGaussian(); // number with normal distribution (mean 0, deviation 1)
					r /= Math.sqrt(l.previousLayer.neurons.length);
					l.conns[j][k] = new Conn(r, l.previousLayer.neurons[k], l.neurons[j]);
				}
			}
		}
	}	
	
	private void startTraining() {
		int step = 0;
		for (int epoch = 1; epoch <= 100; epoch++) { // number of training epochs
			int label = 0; 
			int evaluated = 0; // number of correct evaluations
			int[] labels = MnistReader.getLabels("C:/Mnist/labels.idx1-ubyte");
			ArrayList<int[][]> images = (ArrayList<int[][]>) MnistReader.getImages("C:/Mnist/images.idx3-ubyte");	
			for (int num = 0; num < (int) (N/m); num++) {
				for (int image = step; image < step+m; image++) {
					double[] y = new double[layers[L].neurons.length]; // desired output
					double[] a = new double[layers[L].neurons.length]; // actual output
					for (int i = 0; i < layers[L].neurons.length; i++) {
						y[i] = (i == labels[label]) ? 1 : 0;
					}
					label++; // next loop, next image
					int neuron = 0;
					for (int i = 0; i < images.get(image).length; i++) {
						for (int j = 0; j < images.get(image)[0].length; j++) {
							layers[0].neurons[neuron].activation = images.get(image)[i][j]/255.0; // normalized input (double)
							neuron++;
						}
					}
					
					// feed forward
					for (int i = 1; i < layers.length; i++) {
						for (int j = 0; j < layers[i].neurons.length; j++) {
							double temp = 0;
							for (int k = 0; k < layers[i-1].neurons.length; k++) {
								temp += layers[i].conns[j][k].weight*layers[i-1].neurons[k].activation;
							}
							temp += layers[i].neurons[j].bias;
							temp = MathTool.sigmoid(temp);
							layers[i].neurons[j].activation = temp;
							if (i == L) {
								a[j] = temp;
							}
						}
						if (i == L) {
							for (int j = 0; j < layers[L].neurons.length; j++) {
								layers[L].neurons[j].deltaBias = a[j] - y[j];
								layers[L].neurons[j].sDeltaBias += layers[L].neurons[j].deltaBias;
								for (int k = 0; k < layers[L-1].neurons.length; k++) {
									layers[L].conns[j][k].deltaWeight = layers[L].neurons[j].deltaBias * layers[L-1].neurons[k].activation;
									layers[L].conns[j][k].sDeltaWeight += layers[L].conns[j][k].deltaWeight;
								}
							}
						}
					}
					
					// back propagation
					for (int i = L-1; i > 0; i--) {
						for (int k = 0; k < layers[i].neurons.length; k++) {
							layers[i].neurons[k].deltaBias = 0;
							for (int j = 0; j < layers[i+1].neurons.length; j++) {
								layers[i].neurons[k].deltaBias += layers[i+1].conns[j][k].weight * layers[i+1].neurons[j].deltaBias;
								layers[i].neurons[k].deltaBias *= layers[i].neurons[k].activation * (1 - layers[i].neurons[j].activation);
								layers[i].neurons[k].sDeltaBias += layers[i].neurons[k].deltaBias;
							}
							for (int n = 0; n < layers[i-1].neurons.length; n++) {
								layers[i].conns[k][n].deltaWeight = layers[i-1].neurons[n].activation * layers[i].neurons[k].deltaBias;
								layers[i].conns[k][n].sDeltaWeight += layers[i].conns[k][n].deltaWeight;
							}
						}
					}
					
					// evaluation
					double max = layers[L].neurons[0].activation;
					int index = 0;
					for (int i = 1; i < layers[L].neurons.length; i++) {
						if (layers[L].neurons[i].activation > max) {
							max = layers[L].neurons[i].activation;
							index = i;
						}
					}
					evaluated += (y[index] == 1) ? 1 : 0;
					
					// update biases and weights
					if ((image+1) % m == 0) {
						for (int i = 1; i < layers.length; i++) {
							for (Neuron n : layers[i].neurons) {
								n.updateBias(learningRate, m);
							}
							for (int j = 0; j < layers[i].neurons.length; j++) {
								for (int k = 0; k < layers[i-1].neurons.length; k++) {
									layers[i].conns[j][k].updateWeight(learningRate, m);
								}
							}
						}
	 				}									
				}
				step = (step+m == N) ? 0 : step+m;
			}
			System.out.println("Epoch: " + epoch + "\n" + evaluated + " / " + N);	
		}
	}
}