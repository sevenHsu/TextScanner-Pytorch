import string

digit = [str(i) for i in range(10)]
upper = [i for i in string.ascii_uppercase]
lower = [i for i in string.ascii_lowercase]
alp = upper
alp.remove('I')
alp.remove('O')
license = [i for i in '皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云西陕甘宁青新'] + digit + alp


class Config(object):
    model = 'txt_scan_res18'
    load_model_path = None

    input_size = (96, 384)
    max_seq = 9
    attn = False
    num_classes = 38
    batch_size = 2
    epoches = 50
    lr = 1e-3
    lr_decay = 0.5
    lr_immediate_decay = False
    betas = (0.9, 0.999)
    weight_decay = 0.0005
    save_folder = 'checkpoints'
    gpus = [0]
    print_freq = 2

    num_works = 2
    chars_list = ['_'] + license
    train_txt = open('./cfgs/train.txt', 'r').read().splitlines()[1:]
    valid_txt = open('./cfgs/valid.txt', 'r').read().splitlines()[1:]
    test_txt = open('./cfgs/test.txt', 'r').read().splitlines()[1:]

    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise IOError("Error: opt has not attribut %s" % k)
            setattr(self, k, v)
        self.num_classes = len(self.chars_list)
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = Config()
