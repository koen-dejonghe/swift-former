import Foundation
import TensorFlow
import Python

extension Tensor: TensorProtocol {
    func transpose(_ d1: Int, _ d2: Int) -> Tensor {
	var r = Array<Int>(0 ..< self.rank)
	r[d1] = d2
	r[d2] = d1
        return self.transposed(withPermutations: r)
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

public struct Linear<Scalar: TensorFlowFloatingPoint>: Layer {
    /// The weight matrix.
    public var weight: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative public let activation: Activation
    /// Indicates whether this is a batched dense layer.
    @noDerivative internal let batched: Bool
    /// Indicates whether to use bias or not.
    @noDerivative public let useBias: Bool

    /// The element-wise activation function type.
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    public init(
        weight: Tensor<Scalar>,
        bias: Tensor<Scalar>,
        activation: @escaping Activation,
        useBias: Bool = true
    ) {
        precondition(weight.rank <= 3, "The rank of the 'weight' tensor must be less than 4.")
        precondition(bias.rank <= 2, "The rank of the 'bias' tensor must be less than 3.")
        self.weight = weight
        self.bias = bias
        self.activation = activation
        self.batched = weight.rank == 3
        self.useBias = useBias
    }

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        var hidden: Tensor<Scalar>
        if batched {
            hidden = matmul(input.expandingShape(at: 1), weight).squeezingShape(at: 1)
        } else {
            hidden = matmul(input, weight)
        }
        if useBias {
            hidden = hidden + bias
        }
        return activation(hidden)
    }
}

public extension Linear {
    init(
        inputSize: Int,
        outputSize: Int,
        activation: @escaping Activation = identity,
        weightInitializer: ParameterInitializer<Scalar> = glorotUniform(),
        biasInitializer: ParameterInitializer<Scalar> = zeros(),
        useBias: Bool = true
    ) {
	let bias = useBias ? biasInitializer([outputSize]) : Tensor<Scalar>(repeating: Scalar(0), shape: [0])
	self.init(
	    weight: weightInitializer([inputSize, outputSize]),
	    bias: bias,
	    activation: activation,
	    useBias: useBias)
    }
}

// copied from https://github.com/tensorflow/swift-models/blob/master/Transformer/Model.swift
// mask matrix elements above diagonal with large negative number
@differentiable(wrt: dotProducts, vjp: _vjpCausallyMasked)
func causallyMasked(_ dotProducts: Tensor<Float>, enable: Bool = false) -> Tensor<Float> {
    if !enable {
        return dotProducts
    }
    let (queryTimeSteps, keyTimeSteps) = (dotProducts.shape[1], dotProducts.shape[2])
    let ones = Tensor<Float>(ones: [1, queryTimeSteps, keyTimeSteps])
    let mask = Raw.matrixBandPart(
        ones,
        numLower: Tensor(Int32(-1)),
        numUpper: Tensor(Int32(queryTimeSteps - keyTimeSteps)))
    // return dotProducts * mask - .infinity * (1 - mask)
    return dotProducts * mask - 1e10 * (1 - mask)
}

// causal mask is intentionally invisible to differentiation
func _vjpCausallyMasked(_ dotProducts: Tensor<Float>, enable: Bool)
    -> (Tensor<Float>, (Tensor<Float>) -> Tensor<Float>) {
    // bug in copied code: must pass enable to diff call
    return (causallyMasked(dotProducts, enable: enable), identity)
}


protocol AttentionProtocol: Layer {
}

struct SelfAttentionWide: AttentionProtocol {
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
	let (b, t, e) = (x.shape[0], x.shape[1], x.shape[2])
	let h = self.heads

	precondition(e == self.emb, "Input embedding \(e) should match layer embedding dim \(self.emb)")

	let keys = toKeys(x)
	    .reshaped(to: [b, t, h, e])
	    .transpose(1, 2)
	    .reshaped(to: [b * h, t, e])
	    .transpose(1, 2)
	    / Float(pow(Double(e), 0.25))

	let queries = toQueries(x)
	    .reshaped(to: [b, t, h, e])
	    .transpose(1, 2)
	    .reshaped(to: [b * h, t, e]) 
	    / Float(pow(Double(e), 0.25))

	let values = toValues(x)
	    .reshaped(to: [b, t, h, e])
	    .transpose(1, 2)
	    .reshaped(to: [b * h, t, e])

	let dot = matmul(queries, keys)

	precondition(dot.shape == [b*h, t, t])

	let maskedDot = causallyMasked(dot, enable:mask)

	let smDot = softmax(maskedDot, alongAxis: 2)

	let out = matmul(smDot, values)
	    .reshaped(to: [b, h, t, e])
	    .transpose(1, 2)
	    .reshaped(to: [b, t, h * e])

	return unifyHeads(out)
    }
}


struct SelfAttentionNarrow: Layer {
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
	precondition(
	  emb % heads == 0, 
	  "Embedding dimension \(emb) should be divisible by nr. of heads \(heads)")
	self.emb = emb
	self.heads = heads
	self.mask = mask

	let s = emb / heads

	toKeys = Linear<Float>(inputSize: s, outputSize: s, useBias: false)
	toQueries = Linear<Float>(inputSize: s, outputSize: s, useBias: false)
	toValues = Linear<Float>(inputSize: s, outputSize: s, useBias: false)

        unifyHeads = Dense<Float>(inputSize: heads * s, outputSize: emb)
    }

    @differentiable
    func callAsFunction(_ x: Input) -> Output {

	let (b, t, e) = (x.shape[0], x.shape[1], x.shape[2])
	let h = heads
	precondition(e == emb, "Input embedding \(e) should match layer embedding dim \(emb)")

	let s = e / h
	let x0 = x.reshaped(to: [b, t, h, s])

	let keys = toKeys(x0)
	let queries = toQueries(x0)
	let values = toValues(x0)

	precondition(keys.shape == [b, t, h, s])
	precondition(queries.shape == [b, t, h, s])
	precondition(values.shape == [b, t, h, s])

	let keys1 = keys.transpose(2, 1).reshaped(to: [b * h, t, s])
	let queries1 = queries.transpose(2, 1).reshaped(to: [b * h, t, s])
	let values1 = values.transpose(2, 1).reshaped(to: [b * h, t, s])

	let keys2 = keys1 / Float(pow(Double(e), 0.25))
	let queries2 = queries1 / Float(pow(Double(e), 0.25))

	let dot = matmul(queries2, keys2.transpose(2, 1)) 

	precondition(dot.shape == [b*h, t, t])

	let maskedDot = causallyMasked(dot, enable:mask)

	let smDot = softmax(maskedDot, alongAxis: 2)

	let out = matmul(smDot, values1)
	    .reshaped(to: [b, h, t, s])
	    .transpose(2, 1)
	    .reshaped(to: [b, t, h * s])

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

    // todo generalize this
    var attentionWide: SelfAttentionWide
    var attentionNarrow: SelfAttentionNarrow

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

	attentionWide = SelfAttentionWide(emb: emb, heads: heads, mask: mask)
	attentionNarrow = SelfAttentionNarrow(emb: emb, heads: heads, mask: mask) 

	norm1 = LayerNorm<Float>(featureCount: emb, axis: -1, epsilon: Tensor(1e-5))
	norm2 = LayerNorm<Float>(featureCount: emb, axis: -1, epsilon: Tensor(1e-5))

	ff = Sequential(
	    Dense<Float>(inputSize: emb, outputSize: ffHiddenMult * emb, activation: relu),
	    Dense<Float>(inputSize: ffHiddenMult * emb, outputSize: emb)
	)

	drpt = Dropout<Float>(probability: dropout)
    }

    @differentiable
    func callAsFunction(_ x: Input) -> Output {
	// let x0 = self.wide ? attentionWide(x) : attentionNarrow(x)
	let x0 = attentionNarrow(x)
	// let x0 = attentionWide(x)
	let x1 = norm1(x0 + x)
	let x2 = drpt(x1)
	let x3 = ff(x2)
	let x4 = norm2(x3 + x2)
	let x5 = drpt(x4)

	return x5
    }
}

