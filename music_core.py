import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

from models import linear_model


def get_negative_labels(unlabel_out, position, _postion, thres=0.2):
    length = len(position)
    r = []
    un_idx = []
    for idx in range(length):
        pos = position[idx]
        _pos = _postion[idx]
        _out = unlabel_out[idx][pos]
        softmax = ops.softmax()
        out = softmax(_out)
        if len(pos)==1:
            r.append(_pos[-1])
            un_idx.append(idx)
            continue
        conf = out.min()
        if conf>thres:
            un_idx.append(idx)
            if len(_pos)==0:
                r.append(ops.Argmin()(out).asnumpy())
            else:
                r.append(_pos[-1])
            continue
        t, _ = get_preds(out)
        a = pos[t]
        _postion[idx].append(a)
        position[idx].remove(a)
        r.append(a)
    return np.asarray(r), un_idx

def train_NL(model, pseudo_nl_embeddings, nl_label, epoch, optimizer, test_embeddings, test_targets, train_targets):
    def forward_fn(data, label):
        out = model(data)
        loss_neg = negative_crossentropy(out, ms.Tensor.from_numpy(label))
        loss_entropy = entropy_loss(out)
        loss = loss_neg + loss_entropy
        return loss, out
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    for epc in range(epoch):
        (loss, _), grads = grad_fn(pseudo_nl_embeddings, nl_label)
        loss = ops.depend(loss, optimizer(grads))

def negative_crossentropy(f, labels):
    softmax = ops.softmax(dim=1)
    out = 1 - softmax(f)
    out = softmax(out)
    nllloss = ops.NLLLoss()
    loss, _ = nllloss(ops.log(out), labels)
    return loss

def train(args):
    _POSITION = [[] for _ in range(250)]
    POSITION = [[0, 1, 2, 3, 4] for _ in range(250)]

    crieterion = nn.SoftmaxCrossEntropyWithLogits()
    clf = linear_model()

    optimizer = nn.SGD(clf.trainable_params(), learning_rate=0.1, momentum=0.9, weight_decay=0.0005)
    train(clf, train_embeddings, train_targets, 100, optimizer, crieterion, test_embeddings, test_targets)

    optimizer_NL = nn.SGD(clf.trainable_params(), learning_rate=0.02, weight_decay=0.0005, momentum=0.9)
    while True:
        print('********************************************NL')
        unlabel_out = clf(unlabel_embeddings)
        nl_pred, unselect_idx = get_negative_labels(unlabel_out, POSITION, _POSITION, thres=args.nl_thres)
        select_idx = [x for x in range(250) if x not in unselect_idx]
        if len(select_idx)<1: break
        nl_logits = unlabel_embeddings[select_idx]
        nl_label = unlabel_targets[select_idx]
        nl_pred = nl_pred[select_idx]
        train_NL(clf, nl_logits, nl_pred, 10, optimizer_NL, test_embeddings, test_targets, nl_label)
        pseudo_label = []
        index_pl = []
        for idx in range(len(POSITION)):
            item = POSITION[idx]
            if len(item)==1:
                pseudo_label.append(item[-1])
                index_pl.append(idx)
        pseudo_label = np.asarray(pseudo_label)
        inputs = unlabel_embeddings[index_pl]
        concat_op = ops.Cat(dim=0)
        inputs = concat_op((train_embeddings, inputs))
        pseudo_label = ms.Tensor(pseudo_label).astype(np.float32)
        pseudo_label = concat_op((train_targets, pseudo_label))
        train(clf, inputs, pseudo_label, 100, optimizer, crieterion, test_embeddings, test_targets)



