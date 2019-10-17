print("Hello, world!")

import TensorFlow
import Python

func testSaw() {
    let emb = 4
    let heads = 3
    let mask = false
    let bs = 4

    var saw = SelfAttentionWide(emb: emb, heads: heads, mask: mask)
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
    let numTokens = 256

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

let np = Python.import("numpy")
let gzip = Python.import("gzip")
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


func testLoad() {
    let bs = 32
    let heads = 8
    let emb = 128
    let depth = 12
    let context = 256
    let numTokens = 256
    let lr: Float = 1e-4
    let numBatches = 1000000

    let data = enwik8(path:"data/enwik8.gz")
    let dataTrain = data[0]
    let model = GTransformer(emb:emb, 
			  heads:heads, 
			  depth:depth, 
			  seqLength:context, 
			  numTokens:numTokens, 
			  wide:true) 

    let opt = Adam(for: model, learningRate: lr)

    for i in 0 ..< numBatches {
	// starts = torch.randint(size=(arg.batch_size, ), low=0, high=data_train.size(0) - arg.context - 1)
	let starts = Tensor<Int32>(
	    randomUniform: [bs], 
	    lowerBound: Tensor<Int32>(0), 
	    upperBound: Tensor<Int32>(Int32(dataTrain.shape[0] - context - 1)))

	// seqs_source = [data_train[start  :start+arg.context  ] for start in starts]
	let seqSource: Array<Tensor<UInt8>> = (1 ... bs).map { i in 
	    let range = Int(starts[i].scalarized()) ... Int(starts[i].scalarized()) + context
	    return dataTrain[range]
	}

    }
}
testLoad()


