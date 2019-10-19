import TensorFlow
import Foundation
import Python

struct GTransformer: Module {
    typealias Input = Tensor<Int32>
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

	tokenEmbedding = Embedding<Float>(vocabularySize: numTokens, embeddingSize: emb)
	posEmbedding = Embedding<Float>(vocabularySize: seqLength, embeddingSize: emb)

	tblocks = (1 ... depth).map { d in
	    TransformerBlock(emb:emb, heads:heads, mask:true, seqLength:seqLength, wide:wide)
	}

	toProbs = Dense<Float>(inputSize: emb, outputSize: numTokens)
    }

    @differentiable(wrt: self)
    func callAsFunction(_ x: Input) -> Output {
	let tokens = tokenEmbedding(x)
	// print("tokens.shape: \(tokens.shape)")
	let (b, t, e) = (tokens.shape[0], tokens.shape[1], tokens.shape[2])

	let p = Tensor<Int32>(rangeFrom: Int32(0), to: Int32(t), stride: Int32(1))
	let positions = posEmbedding(p).expandingShape(at:0).broadcasted(to: [b, t, e])
	// print("positions.shape: \(positions.shape)")

	let x0 = tokens + positions
	let x1 = tblocks(x0)

	// in the pytorch implementation, we return a 3D tensor
	// in s4tf, we need to reshape that back to a 2D one
	// let x2 = toProbs(x1.reshaped(to: [b*t, e])).reshaped(to: [b, t, numTokens])

	let x2 = toProbs(x1.reshaped(to: [b*t, e]))

	// we're using softmaxCrossEntropy as the loss function, no need for logSoftmax here
	// return logSoftmax(x2)

	return x2
    }
}
