print("Hello, world!")

import TensorFlow

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
// print(𝛁m1)

let seqLength = 8
let ffHiddenMult = 4
let dropout = 0.0
let wide = true

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

let tblocks = (1 ... 12).map { _ in
    TransformerBlock(emb:emb, heads:heads, mask:false, seqLength:256, wide:true)
}

// let x3 = Tensor<Float>(randomNormal: [4, 8, 256])
let d3 = tblocks.gradient { classifier -> Tensor<Float> in
        let r = classifier(x).sum()
        //let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
        //print("Loss: \(loss)")
        //return loss
        return r
}
print(d3)



let gt = GTransformer(emb: emb, heads: heads, depth: 12, seqLength: 256, numTokens: 256, wide: false) 

// let xg = Tensor<Float>(linearSpaceFrom: 0.0, to: 0.999, count: bs*seqLength)
	 // .reshaped(to: [bs, seqLength])

let xg = Tensor<Float>(linearSpaceFrom: 0.0, to: 0.999, count: bs*emb)
	 .reshaped(to: [bs, emb])

let m3 = gt.gradient { classifier -> Tensor<Float> in
        let r = classifier(xg).sum()
        return r
}
print(m3)


