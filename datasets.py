import os
import os.path as osp
import csv
from mindspore.dataset.transforms.py_transforms import Compose
from config import config
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
import numpy as np
import PIL.Image as Image
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as transforms
from mindspore.dataset.vision import Inter
from mindspore.dataset.vision import py_transforms
import mindspore as ms
from mindspore.dataset import PKSampler
class DataSet:
    def __init__(self, data_root):
        self.img_size = 84
        csv_path = osp.join('./data', 'test' + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1
        self.wnids = []
        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(data_root, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        path, label = np.asarray(self.data)[index], np.asarray(self.label)[index]
        # paths = path.asnumpy()
        image =1
        return image, label, path
class CategoriesSampler(ds.Sampler):
    def  __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # num_batches
        self.n_cls = n_cls  # test_ways
        self.n_per = np.sum(n_per)  # num_per_class
        self.number_distract = n_per[-1]
        label = np.array(label)
        self.m_ind = []  # 20*600 表示每个类别的index，下面的iter方便取
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = Tensor.from_numpy(ind)
            self.m_ind.append(ind)
    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            indicator_batch = []
            classes = ops.Randperm(max_length=25)(Tensor([len(self.m_ind)], dtype=mstype.int32))
            trad_classes = classes[:self.n_cls]
            for c in trad_classes:
                # episode class
                l = self.m_ind[c]
                pos = ops.Randperm(max_length=620)(Tensor([len(l)], dtype = mstype.int32))[:self.n_per]
                # episode data
                cls_batch = l[pos]
                cls_indicator = np.zeros(self.n_per)
                cls_indicator[:cls_batch.shape[0]] = 1
                if cls_batch.shape[0] != self.n_per:
                    # cls_batch = torch.cat([cls_batch, -1 * torch.ones([self.n_per - cls_batch.shape[0]]).long()], 0)
                    cls_batch = ops.Concat()([cls_batch, -1 * ops.Ones()([self.n_per - cls_batch.shape[0]]).long()], 0)
                batch.append(cls_batch)
                indicator_batch.append(cls_indicator)
            # batch = torch.stack(batch).t().reshape(-1)
            # batch = ops.Stack()(batch).T().reshape(-1)
            batch = ops.Stack()(batch)
            batch = ops.Transpose()(batch, (0,1))
            reshape = ops.Reshape()
            batch = reshape(batch, (-1,))
            batch = Tensor.asnumpy(batch)
            yield batch
filenameToPILImage = lambda x: Image.open(x).convert('RGB')
if __name__=='__main__':
    args = config()
    ms.set_context(device_target="GPU")
    a = DataSet(data_root=args.folder)
    sampler = ds.IterSampler(sampler=CategoriesSampler(a.label, args.num_batches,
                                args.num_test_ways, (args.num_shots, 15, args.unlabel)))
    dataset = ds.GeneratorDataset(source=a, column_names=["image", "holder", "path"], sampler=sampler)
    transforms_list = Compose([transforms.Resize([84, 84], Inter.BICUBIC), py_transforms.ToTensor()])
    dataset.map(transforms_list, "image")
    iterator = dataset.create_dict_iterator()
    check = next(iter(iterator))

    paths = check['path'].asnumpy()
    check['image'] = []
    c1 = transforms.Resize([84,84], Inter.BICUBIC)
    c2 = py_transforms.ToTensor()
    for path in paths:
        im = Image.open(path)
        im = c1(im)
        im = c2(im)
        check['image'].append(im)
    dd = 1

