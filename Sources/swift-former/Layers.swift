import TensorFlow

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

