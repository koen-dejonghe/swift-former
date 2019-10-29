print("Hello, world!")

import Python
import TensorFlow
let np = Python.import("numpy")
let gzip = Python.import("gzip")

func testSaw() {
    let emb = 4
    let heads = 3
    let mask = false
    let bs = 4

    let saw = SelfAttention(emb: emb, heads: heads, mask: mask, wide: true)
    // let x = Tensor<Float>(randomNormal: [10, 3, 4])
    let x = Tensor<Float>(linearSpaceFrom: 0.0, to: 0.999, count: bs * heads * emb)
        .reshaped(to: [bs, heads, emb])
    // (randomNormal: [10, 3, 4])
    let ŷ = saw(x)
    print(ŷ.shape)

    let 𝛁m1 = saw.gradient { classifier -> Tensor<Float> in
        let ŷ = classifier(x).sum()
        // let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
        // print("Loss: \(loss)")
        // return loss
        return ŷ
    }
    print(𝛁m1)
}

func testTf() {
    let emb = 4
    let heads = 3
    let mask = false
    let bs = 4
    let seqLength = 8
    let ffHiddenMult = 4
    let dropout = 0.0
    let wide = true
    let x = Tensor<Float>(linearSpaceFrom: 0.0, to: 0.999, count: bs * heads * emb)
        .reshaped(to: [bs, heads, emb])

    let transformer = TransformerBlock(emb: emb,
                                       heads: heads,
                                       mask: mask,
                                       seqLength: seqLength,
                                       ffHiddenMult: ffHiddenMult,
                                       dropout: dropout,
                                       wide: wide)

    let m2 = transformer.gradient { classifier -> Tensor<Float> in
        let r = classifier(x).sum()
        // let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
        // print("Loss: \(loss)")
        // return loss
        return r
    }
    print(m2)
}

func testTBlocks() {
    print("testing tblocks")

    let bs = 32
    let heads = 8
    let emb = 128
    let depth = 12
    let context = 256
    // let numTokens = 256

    let tblocks: [TransformerBlock] = (1 ... depth).map { d in
        print("tblocks depth: \(d)")
        return TransformerBlock(emb: emb, heads: heads, mask: false, seqLength: context, wide: true)
    }

    let x = Tensor<Float>(randomNormal: [bs, context, emb])
    let d = tblocks.gradient { classifier -> Tensor<Float> in
        let r = classifier(x).sum()
        return r
    }
    print(d)
}

// testTBlocks()

func testGT() {
    print("testing gt")
    let bs = 32
    let heads = 8
    let emb = 128
    let depth = 12
    let context = 256
    let numTokens = 256

    let gt = GTransformer(emb: emb,
                          heads: heads,
                          depth: depth,
                          seqLength: context,
                          numTokens: numTokens,
                          wide: false)

    let x = Tensor<Int32>(randomUniform: [bs, context],
                          lowerBound: Tensor<Int32>(0),
                          upperBound: Tensor<Int32>(Int32(numTokens)))

    let m = gt.gradient { classifier -> Tensor<Float> in
        let r = classifier(x).sum()
        return r
    }
    print(type(of: m))
}

// testGT()

func enwik8(path: String,
            nTrain: Int32 = Int32(90e6),
            nValid: Int32 = Int32(5e6),
            nTest: Int32 = Int32(5e6)) -> [Tensor<UInt8>] {
    let file = gzip.open(path).read(nTrain + nValid + nTest)
    let x = np.fromstring(file, dtype: "uint8")
    let xt = Tensor<UInt8>(numpy: x)!
    print(xt.shape)
    let split = xt.split(sizes: Tensor<Int32>([nTrain, nValid, nTest]), alongAxis: 0)
    print(split.count)
    return split
}

/*
 def nll_gaussian(y_pred_mean,y_pred_sd,y_test):

 ## element wise square
 square = tf.square(y_pred_mean - y_test)## preserve the same shape as y_pred.shape
 ms = tf.add(tf.divide(square,y_pred_sd), tf.log(y_pred_sd))
 ## axis = -1 means that we take mean across the last dimension
 ## the output keeps all but the last dimension
 ## ms = tf.reduce_mean(ms,axis=-1)
 ## return scalar
 ms = tf.reduce_mean(ms)
 return(ms)
 }
 */

/*
 # dim == 3 or dim > 4
 n = input.size(0)
 c = input.size(1)
 out_size = (n,) + input.size()[2:]
 if target.size()[1:] != input.size()[2:]:
    raise ValueError('Expected target size {}, got {}'.format( out_size, target.size()))
 input = input.contiguous().view(n, c, 1, -1)
 target = target.contiguous().view(n, 1, -1)
 reduction_enum = _Reduction.get_enum(reduction)
 if reduction != 'none':
    ret = torch._C._nn.nll_loss2d(input, target, weight, reduction_enum, ignore_index)
 else:
    out = torch._C._nn.nll_loss2d(input, target, weight, reduction_enum, ignore_index)
    ret = out.view(out_size)
 */

/*
 def NLLLoss(logs, targets):
 out = logs[range(len(targets)), targets]
 return -out.sum()/len(out)
 */

/*
 func nllLoss(source: Tensor<Float>, target: Tensor<Int32>) -> Float {

 let n = source.shape[0]
 let c = source.shape[1]
 let z = source.shape[2]

 let s = source.reshaped(to: [n, c, 1, z])
 let t = target.reshaped(to: [n, 1, z])

 let out = s[0 ..< n, t]

 return 0.0
 }
 */

extension Array {
    func grouped(size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

func testLoad() {
    let bs = 32
    let heads = 8
    let emb = 128
    let depth = 12
    let context = 256
    let numTokens = 256
    // let lr: Float = 1e-4
    let lr: Float = 1e-3
    let numBatches = 1_000_000
    // let testEvery = 1500
    let testEvery = 5
    let testSubset = 100_000
    let testBatchSize = 64

    let data = enwik8(path: "data/enwik8.gz")
    let dataTrain = data[0]
    let dataTest = data[1]
    var model = GTransformer(emb: emb,
                             heads: heads,
                             depth: depth,
                             seqLength: context,
                             numTokens: numTokens,
                             wide: false)

    let optimizer = Adam(for: model, learningRate: lr)

    for b in 0 ..< numBatches {
        let starts = Tensor<Int32>(
            randomUniform: [bs],
            lowerBound: Tensor<Int32>(0),
            upperBound: Tensor<Int32>(Int32(dataTrain.shape[0] - context - 1))
        )

        let seqSource: [Tensor<UInt8>] = (0 ..< bs).map { i in
            let range = Int(starts[i].scalarized()) ..< Int(starts[i].scalarized()) + context
            return dataTrain[range]
        }

        let seqTarget: [Tensor<UInt8>] = (0 ..< bs).map { i in
            let range = Int(starts[i].scalarized()) + 1 ..< Int(starts[i].scalarized()) + context + 1
            return dataTrain[range]
        }

        let source = Tensor<Int32>(Tensor(concatenating: seqSource)).reshaped(to: [bs, context])
        let target = Tensor<Int32>(Tensor(concatenating: seqTarget)).reshaped(to: [bs * context])

        let (_, grad) = model.valueWithGradient { generator -> Tensor<Float> in
            let output = generator(source)
            // print(output)
            let loss = softmaxCrossEntropy(logits: output, labels: target)
            print("batch \(b): \(loss)")
            return loss
        }

        optimizer.update(&model, along: grad)

        if b != 0, b % testEvery == 0 || b == numBatches {





        }
    }

/* ==========================================================
*/
    func bitsPerByte(b: Int) {
        let upto = b == numBatches - 1 ? dataTest.shape[0] : testSubset
        let dataSub = dataTest[..<upto]
        let (bits, tot) = (0.0, 0)
        Array(0 ..< dataSub.shape[0])
            .grouped(size: testBatchSize)
            .forEach { chunk in
                let batch: [Tensor<Int32>] = chunk.map { current in
                    let fr = max(0, current - context)
                    let to = current + 1
                    var localContext = Tensor<Int32>(dataSub[fr ..< to])
                    if localContext.shape[0] < context + 1 {
                        let pad = Tensor<Int32>(zeros: [context + 1 - localContext.shape[0]])
                        localContext = Tensor(concatenating: [pad, localContext])
                        assert(localContext.shape[0] == context + 1)
                    }
                    return localContext.reshaped(to: [1, -1])
                }

                let all = Tensor<Int32>(concatenating: batch, alongAxis: 0)
                // print(all.shape) 64 x 257
                let source = all.slice(lowerBounds: [0, 0], upperBounds: [all.shape[0], all.shape[1] - 1])
                let target = all
                    .slice(lowerBounds: [0, all.shape[1] - 1], upperBounds: [all.shape[0], all.shape[1]])
                    .squeezingShape()
                // print(target.shape) // 64

                let output = model(source)
                    .reshaped(to: [batch.count, -1, source.shape[1]])
                // print(output.shape) // 64 x 256 x 256

                let r = ..<batch.count
                let lnProbs = output[r, -1, -1].gathering(atIndices: target)
                print(lnProbs)
                // let lnProbs = output.gathering(atIndices: target, alongAxis: 2)
                print(lnProbs.shape)

                // let lnprobs = output[torch.arange(b, device=d()), -1, target]
                // print(output.shape)
            }
    }
}

// testLoad()

func testIndexing() {
    let x = Tensor<Float>(rangeFrom: 0, to: 100, stride: 1).reshaped(to: [10, 10])
}

testIndexing()