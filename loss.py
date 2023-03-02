import mindspore as ms
import mindspore.ops as ops
def NL_loss(f, labels):
    Q_1 = 1 - ops.Softmax(1)(f)
    Q = ops.Softmax(1)(Q_1)
    weight = 1-Q
    out = weight*Q.log()
    return ops.NLLLoss()(logits=out, labels=labels)
def entropy_loss(p):
    p = ops.Softmax(axis=1)(p)
    epsilon = 1e-5
    return -1 * ops.ReduceSum()(p * ops.Log()(p + epsilon)) / p.shape(0)