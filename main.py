from datasets import *
from loss import *
from utils import *
def main(args):
    ms.set_context(device_target="GPU")
    if args.mode == 'meta_test_FC':
        meta_test_FC(args)
    else:
        raise NameError
    return
if __name__=='__main__':
    args = config()
    main(args)
