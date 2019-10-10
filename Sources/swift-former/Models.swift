import TensorFlow

struct SelfAttentionWide: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    let emb: Int
    let heads: Int
    let mask: Int
    let toKeys: Dense<Float>
    let toQueries: Dense<Float>
    let toValues: Dense<Float>

    init(emb: Int, heads: Int, mask: Int) {
	self.emb = emb
	self.heads = heads
	self.mask = mask
	toKeys = Dense<Float>(inputSize: emb, outputSize: emb * heads)
	toQueries = Dense<Float>(inputSize: emb, outputSize: emb * heads)
	toValues = Dense<Float>(inputSize: emb, outputSize: emb * heads)
    }


    var conv1 = Conv2D<Float>(filterShape: (5, 5, 3, 6), activation: relu)
    var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var conv2 = Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    var flatten = Flatten<Float>()
    var dense1 = Dense<Float>(inputSize: 16 * 5 * 5, outputSize: 120, activation: relu)
    var dense2 = Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    var dense3 = Dense<Float>(inputSize: 84, outputSize: 10, activation: identity)

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
	let convolved = input.sequenced(through: conv1, pool1, conv2, pool2)
	return convolved.sequenced(through: flatten, dense1, dense2, dense3)
    }

}


