print("Hello, world!")

import TensorFlow
import Python
let np = Python.import("numpy")
let gzip = Python.import("gzip")

func testSaw() {
    let emb = 4
    let heads = 3
    let mask = false
    let bs = 4

    let saw = SelfAttention(emb: emb, heads: heads, mask: mask, wide: true)
    // let x = Tensor<Float>(randomNormal: [10, 3, 4])
    let x = Tensor<Float>(linearSpaceFrom: 0.0, to: 0.999, count: bs*heads*emb)
	     .reshaped(to: [bs, heads, emb])
    //(randomNormal: [10, 3, 4])
    let ŷ = saw(x)
    print(ŷ.shape)

    let 𝛁m1 = saw.gradient { classifier -> Tensor<Float> in
	    let ŷ = classifier(x).sum()
	    //let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
	    //print("Loss: \(loss)")
	    //return loss
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
    let x = Tensor<Float>(linearSpaceFrom: 0.0, to: 0.999, count: bs*heads*emb)
	     .reshaped(to: [bs, heads, emb])

    let transformer = TransformerBlock(emb:emb, 
				       heads:heads, 
				       mask:mask, 
				       seqLength:seqLength, 
				       ffHiddenMult:ffHiddenMult, 
				       dropout:dropout, 
				       wide:wide)

    let m2 = transformer.gradient { classifier -> Tensor<Float> in
	    let r = classifier(x).sum()
	    //let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
	    //print("Loss: \(loss)")
	    //return loss
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

    let tblocks:Array<TransformerBlock> = (1 ... depth).map { d in
	print("tblocks depth: \(d)")
	return TransformerBlock(emb:emb, heads:heads, mask:false, seqLength:context, wide:true)
    }

    let x = Tensor<Float>(randomNormal: [bs, context, emb])
    let d = tblocks.gradient { classifier -> Tensor<Float> in
	let r = classifier(x).sum()
	return r
    }
    print(d)
}
// testTBlocks()


func testGT()
{
    print("testing gt")
    let bs = 32
    let heads = 8
    let emb = 128
    let depth = 12
    let context = 256
    let numTokens = 256

    let gt = GTransformer(emb:emb, 
			  heads:heads, 
			  depth:depth, 
			  seqLength:context, 
			  numTokens:numTokens, 
			  wide:false) 

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
    let x = np.fromstring(file, dtype:"uint8")
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


func testLoad() {
    let bs = 32
    let heads = 8
    let emb = 128
    let depth = 12
    let context = 256
    let numTokens = 256
    // let lr: Float = 1e-4
    let lr: Float = 1e-3
    let numBatches = 1000000

    let data = enwik8(path:"data/enwik8.gz")
    let dataTrain = data[0]
    var model = GTransformer(emb:emb, 
			  heads:heads, 
			  depth:depth, 
			  seqLength:context, 
			  numTokens:numTokens, 
			  wide:false) 

    let optimizer = Adam(for: model, learningRate: lr)

    for b in 0 ..< numBatches {
        let starts = Tensor<Int32>(
            randomUniform: [bs],
            lowerBound: Tensor<Int32>(0),
            upperBound: Tensor<Int32>(Int32(dataTrain.shape[0] - context - 1)))

        let seqSource: Array<Tensor<UInt8>> = (0 ..< bs).map { i in
            let range = Int(starts[i].scalarized()) ..< Int(starts[i].scalarized()) + context
            return dataTrain[range]
        }

        let seqTarget: Array<Tensor<UInt8>> = (0 ..< bs).map { i in
            let range = Int(starts[i].scalarized()) + 1 ..< Int(starts[i].scalarized()) + context + 1
            return dataTrain[range]
        }

        let source = Tensor<Int32>(Tensor(concatenating: seqSource)).reshaped(to:[bs, context])
        let target = Tensor<Int32>(Tensor(concatenating: seqTarget)).reshaped(to:[bs*context])

        let (_, grad) = model.valueWithGradient { generator -> Tensor<Float> in
            let output = generator(source)
	    // print(output)
	    let loss = softmaxCrossEntropy(logits: output, //.reshaped(to: [bs*context, -1]), 
				           labels: target)
	    print("batch \(b): \(loss)")
	    return loss
        }

	optimizer.update(&model, along: grad)

    }
}
testLoad()


