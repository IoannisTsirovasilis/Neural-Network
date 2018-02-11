package neuralNetwork;

import java.util.Random;

public class Layer {
	protected Layer previousLayer;
	protected Neuron[] neurons;
	protected Conn[][] conns;
	
	public Layer(int nNeurons) {
		neurons = new Neuron[nNeurons];
		previousLayer = null;
		conns = null;
		Random rand = new Random();
		for (int i = 0; i < neurons.length; i++) {
			double r = rand.nextGaussian();
			neurons[i] = new Neuron(r);
		}		
	}
}