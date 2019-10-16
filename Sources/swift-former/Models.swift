import TensorFlow
import Foundation
import TensorFlow
import Python

let np = Python.import("numpy")

struct SelfAttentionWide: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    @noDerivative let emb: Int
    @noDerivative let heads: Int
    @noDerivative let mask: Bool
    var toKeys: Linear<Float>
    var toQueries: Linear<Float>
    var toValues: Linear<Float>
    var unifyHeads: Dense<Float>

    init(emb: Int, heads: Int, mask: Bool) {
	self.emb = emb
	self.heads = heads
	self.mask = mask
	toKeys = Linear<Float>(inputSize: emb, outputSize: emb * heads, useBias: false)
	toQueries = Linear<Float>(inputSize: emb, outputSize: emb * heads, useBias: false)
	toValues = Linear<Float>(inputSize: emb, outputSize: emb * heads, useBias: false)
        unifyHeads = Dense<Float>(inputSize: heads * emb, outputSize: emb)
    }

    @differentiable
    func callAsFunction(_ x: Input) -> Output {
	let b = x.shape[0]
	let t = x.shape[1]
	let e = x.shape[2]
	let h = self.heads

	// assert  is not differentiable
	// assert(e == self.emb, "Input embedding \(e) should match layer embedding dim \(self.emb)")

	let keys = toKeys(x)
	    .reshaped(to: [b, t, h, e])
	    .transposed(withPermutations: [0, 2, 1, 3])
	    .reshaped(to: [b * h, t, e])
	    .transposed(withPermutations: [0, 2, 1]) / Float(pow(Double(e), 0.25))
	let queries = toQueries(x)
	    .reshaped(to: [b, t, h, e])
	    .transposed(withPermutations: [0, 2, 1, 3])
	    .reshaped(to: [b * h, t, e]) / Float(pow(Double(e), 0.25))
	let values = toValues(x)
	    .reshaped(to: [b, t, h, e])
	    .transposed(withPermutations: [0, 2, 1, 3])
	    .reshaped(to: [b * h, t, e])

	let dot = softmax(matmul(queries, keys), alongAxis: 2)

	// assert  is not differentiable
	// assert(dot.shape == [b*h, t, t])

	// todo implement masking

	let out = matmul(dot, values)
	    .reshaped(to: [b, h, t, e])
	    .transposed(withPermutations: [0, 2, 1, 3])
	    .reshaped(to: [b, t, h * e])

	return unifyHeads(out)
    }
}

struct TransformerBlock: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    @noDerivative let emb: Int
    @noDerivative let heads: Int
    @noDerivative let mask: Bool
    @noDerivative let seqLength: Int
    @noDerivative let ffHiddenMult: Int
    @noDerivative let dropout: Double
    @noDerivative let wide: Bool

    var attention: SelfAttentionWide
    var norm1: LayerNorm<Float>
    var norm2: LayerNorm<Float>
    var ff: Sequential<Dense<Float>,Dense<Float>>
    var drpt: Dropout<Float>

    init(emb: Int, heads: Int, mask: Bool, seqLength: Int, ffHiddenMult: Int = 4, dropout: Double = 0.0, wide: Bool = true) {

	self.emb = emb
	self.heads = heads
	self.mask = mask
	self.seqLength = seqLength
	self.ffHiddenMult = ffHiddenMult
	self.dropout = dropout
	self.wide = wide

	attention = SelfAttentionWide(emb: emb, heads: heads, mask: mask)
	norm1 = LayerNorm<Float>(featureCount: emb, axis: 0, epsilon:  Tensor(1e-5))
	norm2 = LayerNorm<Float>(featureCount: emb, axis: 0, epsilon:  Tensor(1e-5))

	ff = Sequential(
	    Dense<Float>(inputSize: emb, outputSize: ffHiddenMult * emb, activation: relu),
	    Dense<Float>(inputSize: ffHiddenMult * emb, outputSize: emb)
	)

	drpt = Dropout<Float>(probability: dropout)
    }

    @differentiable
    func callAsFunction(_ x: Input) -> Output {
	let x0 = attention(x)
	let x1 = norm1(x0 + x)
	let x2 = drpt(x1)
	let x3 = ff(x2)
	let x4 = norm2(x3 + x2)
	let x5 = drpt(x4)

	return x5
    }
}

// copied from fast ai
extension Array: Module where Element: Layer, Element.Input == Element.Output {
    public typealias Input = Element.Input
    public typealias Output = Element.Output

    @differentiable(wrt: (self, input))
    public func callAsFunction(_ input: Input) -> Output {
          return self.differentiableReduce(input) { $1($0) }
    }
}
extension Array: Layer where Element: Layer, Element.Input == Element.Output {}


struct GTransformer: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    @noDerivative let emb: Int
    @noDerivative let heads: Int
    @noDerivative let depth: Int
    @noDerivative let seqLength: Int
    @noDerivative let numTokens: Int
    @noDerivative let wide: Bool

    var tokenEmbedding: Embedding<Float>
    var posEmbedding: Embedding<Float>
    var toProbs: Dense<Float>
    var tblocks: Array<TransformerBlock>

    init(emb: Int, heads: Int, depth: Int, seqLength: Int, numTokens: Int, wide: Bool = false) {
	self.emb = emb
	self.heads = heads
	self.depth = depth
	self.seqLength = seqLength
	self.numTokens = numTokens
	self.wide = wide

	tokenEmbedding = Embedding<Float>(vocabularySize: emb, embeddingSize: numTokens)
	posEmbedding = Embedding<Float>(vocabularySize: emb, embeddingSize: seqLength)

	tblocks = (1 ... depth).map { d in
	    print("tblocks depth: \(d)")
	    return TransformerBlock(emb:emb, heads:heads, mask:false, seqLength:seqLength, wide:wide)
	}

	toProbs = Dense<Float>(inputSize: emb, outputSize: numTokens)

    }

    @differentiable
    func callAsFunction(_ x: Input) -> Output {
	let tokens = tokenEmbedding(Tensor<Int32>(x))
	print("tokens.shape: \(tokens.shape)")
	let b = tokens.shape[0]
	let t = tokens.shape[1]
	let e = tokens.shape[2]

	let a = np.arange(t, dtype:"int32")
	let p = Tensor<Int32>(numpy: a)!
	let positions = posEmbedding(p).expandingShape(at:0).broadcasted(to: [b, t, e])
	print("positions.shape: \(positions.shape)")

	let x0 = tokens + positions
	let x1 = tblocks(x0)
	// let x2 = toProbs(x1.reshaped(to: [b*t, e])).reshaped(to: [b, t, numTokens])

	// return logSoftmax(x2)

	print(x1)

	return x1

    }
}
