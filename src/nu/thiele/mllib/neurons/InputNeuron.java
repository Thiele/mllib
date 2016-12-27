package nu.thiele.mllib.neurons;

public class InputNeuron extends Neuron{
	public InputNeuron(String id) {
		super(id);
	}

	@Override
	public void addConnectionBackward(Neuron n) {}
	
	@Override
	public void fire(){
		this.latestOutput = this.sumOfInputs;
		for(Neuron n : forwardConnections) n.input(latestOutput*weights.get(n));
	}
}