import torch
import models
from utils import load_state
from cfgs.config import opt


def export(**kwargs):
    opt.parse(kwargs)

    # load model
    model = getattr(models, 'txt_scan_res18')(opt.input_size, opt.max_seq, opt.num_classes, mode='test', attn=opt.attn)
    load_state(model, 'checkpoints/txt_scan_res18_lr_1e-5_batch_16/best_val_error.pth', 'cpu')
    # model.cuda()
    model.eval()
    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, 96, 384)
    h0 = torch.zeros(1, 1, 384)
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, (example, h0))

    inputs = torch.rand(1, 3, 96, 384).cuda()
    h0 = h0.cuda()
    # output = traced_script_module(inputs)

    traced_script_module.save("plug_in/txt_scan_res18_lr_1e-5_batch_16.pt")
    model_traced = torch.jit.load('plug_in/txt_scan_res18_lr_1e-5_batch_16.pt')

    model.cuda()
    model_traced.cuda()
    model_traced.eval()

    output1 = model(inputs, h0)
    output2 = model_traced(inputs, h0)
    for i in range(3):
        print('diff of outputs: %f' % ((output1[i] - output2[i]).sum().item()))


if __name__ == "__main__":
    import fire

    fire.Fire(export)
