from datasets import *
from mindspore import nn
from model import resnet50
import mindspore.numpy as msnp
from loss import *
def get_preds_position_(unlabel_out, position, _postion, thres=0.001):
    length = len(position)
    r = []
    un_idx = []
    for idx in range(length):
        pos = position[idx]
        _pos = _postion[idx]
        _out = unlabel_out[idx][pos]
        out = ops.Softmax()(_out)
        if len(pos)==1:
            un_idx.append(idx)
            continue
        conf = ops.ArgMinWithValue()(out)[1]
        if conf>thres:
            un_idx.append(idx)
            if len(_pos)==0:
                r.append(ops.Argmin(axis=0)(out).asnumpy())
            else:
                r.append(_pos[-1])
            continue
        t, _ = get_preds(out)
        a = pos[t]
        _postion[idx].append(a)
        position[idx].remove(a)
        r.append(a)
    return np.asarray(r), un_idx
def get_preds(out):
    # out = F.softmax(out)
    preds = ops.Argmin(axis=0)(out).asnumpy()
    return preds, preds
def get_embedding(model, inputs, type=False):
    batch_size = 64
    inputs = Tensor(inputs)
    embed = model(inputs)
    assert embed.shape[0] == inputs.shape[0]
    if type:return embed
    return embed.numpy()
def train_loop(model, dataset, loss_fn, optimizer, inputs, targets):
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label) + entropy_loss(logits)
        # loss = loss_fn(logits, label)
        return loss, logits
    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # Define function of one-step training
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits
    model.set_train()
    inputs = Tensor(inputs)
    loss, logits = train_step(inputs, targets)
    return loss, logits
def train_loop_NL(model, dataset, loss_fn, optimizer, inputs, targets):
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label) + entropy_loss(logits)
        # loss = loss_fn(logits, label)
        return loss, logits
    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # Define function of one-step training
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits
    model.set_train()
    inputs = Tensor(inputs)
    loss, logits = train_step(inputs, Tensor(targets))
    return loss, logits
def meta_test_FC(args):
    if args.dataset == 'cub':
        num_classes = 100
    elif args.dataset == 'tieredimagenet':
        num_classes = 351
    else:
        num_classes = 64
    model = resnet50(num_classes=num_classes, pretrained=False)
    a = DataSet(data_root=args.folder)
    sampler = ds.IterSampler(sampler=CategoriesSampler(a.label, args.num_batches,
                                                       args.num_test_ways, (args.num_shots, 15, args.unlabel)))
    dataset = ds.GeneratorDataset(source=a, column_names=["image", "holder", "path"], sampler=sampler)
    loader = dataset.create_dict_iterator()
    k = args.num_shots * args.num_test_ways
    for ia, check in enumerate(loader):
        targets = msnp.tile(msnp.arange(args.num_test_ways), (args.num_shots + 15 + args.unlabel))
        paths = check['path'].asnumpy()
        data = []
        c1 = transforms.Resize([84, 84], Inter.BICUBIC)
        c2 = py_transforms.ToTensor()
        for path in paths:
            im = Image.open(path)
            im = c1(im)
            im = c2(im)
            data.append(im)
        train_inputs = data[:k]
        train_targets = targets[:k]
        test_inputs = data[k:k+15*args.num_test_ways]
        test_targets = targets[k:k+15*args.num_test_ways].asnumpy()
        unlabel_targets = targets[k + 15 * args.num_test_ways:].asnumpy()
        train_embeddings = get_embedding(model=model, inputs=train_inputs, type=True)
        test_embeddings = get_embedding(model, inputs=test_inputs, type=True)
        if args.unlabel != 0:
            unlabel_inputs = data[k + 15 * args.num_test_ways:]
            unlabel_embeddings = get_embedding(model, unlabel_inputs, type=True)
        if args.classifier =='linear':
            ori_index = [x for x in range(250)]
            _POSITION = [[] for _ in range(250)]
            POSITION = [[0, 1, 2, 3, 4] for _ in range(250)]
            criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            clf = nn.Dense(in_channels=18432, out_channels=5)
            optimizer = nn.SGD(clf.trainable_params(), learning_rate = 1e-3)
            print('\n********************************************PL')
            for epoch in range(100):
                loss=train_loop(clf, None, criterion, optimizer, train_embeddings, train_targets)
                print(f"Epoch: {epoch}  Loss: {loss}")
            unlabel_out = clf(unlabel_embeddings)
            nl_pred, unselect_idx = get_preds_position_(unlabel_out, POSITION, _POSITION, thres = 0.2)
            select_idx = [x for x in ori_index if x not in unselect_idx]
            _unlabel_embeddings = unlabel_embeddings[select_idx]
            _unlabel_t = unlabel_targets[select_idx]
            nl_pred = nl_pred[select_idx]
            print('********************************************SelectNL')
            optimizer_NL = nn.SGD(clf.trainable_params(), learning_rate = 1e-3)
            for epoch in range(10):
                loss = train_loop_NL(clf, None, NL_loss, optimizer_NL, _unlabel_embeddings, nl_pred)
                print(f"Epoch: {epoch}  Loss: {loss}")
            unlabel_out = clf(unlabel_embeddings)
            nl_pred2, unselect_idx2 = get_preds_position_(unlabel_out, POSITION, _POSITION, thres=0.2)
            # nl_pred2 = np.asarray(nl_pred2)
            select_idx2 = [x for x in ori_index if x not in unselect_idx2]
            _unlabel_embeddings = unlabel_embeddings[select_idx2]
            _unlabel_t = unlabel_targets[select_idx2]
            nl_pred2 = nl_pred2[select_idx2]
            for epoch in range(10):
                loss = train_loop_NL(clf, None, NL_loss, optimizer_NL, _unlabel_embeddings, nl_pred2)
                print(f"Epoch: {epoch}  Loss: {loss}")
            unlabel_out = clf(unlabel_embeddings)
            # nl_pred3, nl_conf = get_preds_third(unlabel_out)

            nl_pred3, unselect_idx3 = get_preds_position_(unlabel_out, POSITION, _POSITION, thres=0.2)
            nl_pred3 = np.asarray(nl_pred3)
            select_idx3 = [x for x in ori_index if x not in unselect_idx3]
            # select_idx3 = [x for x in ori_index if x not in _unselect_idx]
            _unlabel_targets = unlabel_targets[select_idx3]
            _unlabel_embeddings = unlabel_embeddings[select_idx3]

            indexes_nl3 = [x for x in range(len(_unlabel_targets))]
            nl_pred3 = nl_pred3[select_idx3]
            for epoch in range(10):
                loss = train_loop_NL(clf, None, NL_loss, optimizer_NL, _unlabel_embeddings, nl_pred3)
                print(f"Epoch: {epoch}  Loss: {loss}")
            unlabel_out = clf(unlabel_embeddings)

            nl_pred4, unselect_idx4 = get_preds_position_(unlabel_out, POSITION, _POSITION, thres=0.2)
            # nl_pred4 = np.asarray(nl_pred4)
            select_idx4 = [x for x in ori_index if x not in unselect_idx4]
            # select_idx4 = [x for x in ori_index if x not in _unselect_idx]
            _unlabel_targets = unlabel_targets[select_idx4]
            _unlabel_embeddings = unlabel_embeddings[select_idx4]
            nl_pred4 = nl_pred4[select_idx4]
            indexes_nl4 = [x for x in range(len(_unlabel_targets))]
            for epoch in range(10):
                loss = train_loop_NL(clf, None, NL_loss, optimizer_NL, _unlabel_embeddings, nl_pred3)
                print(f"Epoch: {epoch}  Loss: {loss}")

            class_num = [0 for _ in range(5)]
            pseudo_label = []
            index_pl = []
            for idx in range(len(POSITION)):
                item = POSITION[idx]
                if len(item) == 1:
                    lab = item[0]
                    pseudo_label.append(item[-1])
                    class_num[lab] += 1
                    index_pl.append(idx)
            class_num = [item + 8 for item in class_num]
            max_ = max(class_num) * 1.0
            pseudo_label = np.asarray(pseudo_label)
            t1_ = unlabel_embeddings[index_pl]
            t2_ = Tensor(pseudo_label, dtype=mstype.int64)
            for epoch in range(100):
                loss=train_loop(clf, None, criterion, optimizer, t1_, t2_)
                print(f"Epoch: {epoch}  Loss: {loss}")