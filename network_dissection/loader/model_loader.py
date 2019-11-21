import settings
import torch
import torchvision
import sys
import os
print(os.path.pardir)
sys.path.append(os.path.pardir)
import models



def loadmodel(hook_fn):
    if settings.IS_TORCHVISION:
        if settings.MODEL_FILE is None:
            model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
        else:
            checkpoint = torch.load(settings.MODEL_FILE)
            if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
                model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
                if settings.MODEL_PARALLEL:
                    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                        'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            else:
                model = checkpoint
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        net = getattr(models, settings.MODEL)
        model = net(num_classes=settings.NUM_CLASSES)
        param_dict = {}
        for k, v in zip(model.state_dict().keys(), checkpoint['net'].keys()):
            param_dict[k] = checkpoint['net'][v]
        model.load_state_dict(param_dict)

    for name in settings.FEATURE_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
