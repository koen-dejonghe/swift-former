print("Hello, world!")

import TensorFlow

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
    // print(m2)
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

    let gt = GTransformer(emb:emb, heads:heads, depth:depth, seqLength:context, numTokens:numTokens, wide:false) 
    // let x = Tensor<Float>(linearSpaceFrom: 0.0, to: 0.999, count: bs*context).reshaped(to: [bs, context])
    let x = Tensor<Float>(randomNormal: [bs, context])

    let m = gt.gradient { classifier -> Tensor<Float> in
	let r = classifier(x).sum()
	return r
    }
    print(m)
}

testGT()


