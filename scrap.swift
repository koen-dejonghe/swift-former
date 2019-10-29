
import TensorFlow
import Python

// let np = Python.import("numpy")
// let pi = np.random.randint(low: 0, high: 10, size: 10, dtype: "int32")
// let y = Tensor<Int32>(numpy: pi)!

withDevice(.cpu) {
    let x = Tensor<Float>(rangeFrom: 0, to: 30, stride: 1).reshaped(to: [10, 3])
    let y = Tensor<Int32>(rangeFrom: 0, to: 10, stride: 1) % 3

    let idx: Tensor<Int32> = Raw.range(start: Tensor(0), 
        limit: Tensor(numericCast(y.shape[0])), 
        delta: Tensor(1))

    let indices = Raw.concat(concatDim: Tensor(1), 
        [idx.expandingShape(at: -1), y.expandingShape(at: -1)])

    let r = Raw.gatherNd(params: x, indices: indices)

    print(r)
    print(r.shape)

}

