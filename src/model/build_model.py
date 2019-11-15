from .densenet import DenseNet
from .inception import GoogLeNet
import torch

def getModel(training=False,model_to_use='densenet',**kwargs):

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    
    if model_to_use == 'inception':
        model = GoogLeNet()
        model = torch.nn.DataParallel(model)
    else:
        model = DenseNet(**kwargs)
    
    model.eval()
    
    '''
    
    if training :
        model.train()
        model = torch.nn.DataParallel(model)
        print("The model is in training mode")
    '''

    print("No of params in model is " , sum(p.numel() for p in model.parameters() if p.requires_grad))
    model  = model.to(device)
    print("model is loaded on GPU {next(model.parameters()).is_cuda}")
    return model
