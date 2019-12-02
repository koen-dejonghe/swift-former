import TensorFlow
import Python

extension Array where Element: Numeric {
  func product() -> Element { return self.reduce(1, *) }
  func sum() -> Element { return self.reduce(0, +) }

  static func fill(filler: () -> Element, count: Int) -> Array {
    Array(filler: filler, count: count)
  }

  init(filler: () -> Element, count: Int) {
    self.init()
    self.reserveCapacity(count)
    for _ in 0 ..< count {
      self.append(filler())
    }
  }
}

struct Ndarray {
  var data: [Double]
  let shape: [Int]

  init(data: [Double], shape: [Int]) {
    self.shape = shape
    self.data = data
  }

  init(ones: [Int]) {
    self.data = Array(repeating: 1.0, count: ones.product())
    self.shape = ones
  }

  init(zeros: [Int]) {
    self.data = Array(repeating: 0.0, count: zeros.product())
    self.shape = zeros
  }

  init(randn: [Int], in range: ClosedRange<Double> = 0.0 ... 1.0) {
    self.data = Array.fill(filler: { Double.random(in: range) }, count: randn.product())
    self.shape = randn
  }

  init(_ data: [Double]) {
    self.init(data: data, shape: [1])
  }

  init(_ data: Double...) {
    self.init(data: data, shape: [1])
  }

  static func arange(start: Double = 0.0, _ stop: Double, step: Double = 1.0) -> Ndarray {
    let data = stride(from: start, to: stop, by: step)
    return Ndarray(data: Array(data), shape: [1])
  }

  func rank() -> Int {
    return shape.count
  }

  // remove dim=1
  func squeeze() -> Ndarray {
    let newShape = shape.filter { $0 != 1 }
    return Ndarray(data: data, shape: newShape)
  }

  func reshape(newShape: [Int]) -> Ndarray {
    assert(newShape.product() == data.count)
    return Ndarray(data: data, shape: newShape)
  }

  subscript(index: Int...) -> Ndarray {
    get {
      var newShape = shape
      newShape.removeFirst(index.count)
      return Ndarray(randn: newShape)
    }
  }
}

let a = Ndarray(randn: [3, 4, 3])
print(a)
print(a.rank())
print(a[2].shape)

print(Ndarray(ones: [2, 3]))
print(Ndarray.arange(10).reshape(newShape: [5, 2]))
