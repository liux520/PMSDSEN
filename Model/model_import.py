import torch


def model_import(model_name: str = 'PMSDSEN',
                 cls: int = 19,
                 act_name: str = 'gelu',
                 load: bool = False):
    if model_name == 'PMSDSEN':
        from Model.PMSDSEN import PMSDSEN
        model = PMSDSEN(cls=cls, act_name=act_name)
        if load:
            load_func(
                path=r'../Weights/PMSDSEN_city_75.02.pth',
                model=model,
                model_name=model_name
            )

    else:
        NotImplementedError('Not implemented.')
        exit()

    return model


def load_func(path, model, model_name):
    checkpoint = torch.load(path, map_location='cpu')
    ###
    model_dict = model.state_dict()
    overlap = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(overlap)
    print(f'{(len(overlap) * 1.0 / len(checkpoint["state_dict"]) * 100):.4f}% weights from checkpoint is loaded!')
    print(f'{(len(overlap) * 1.0 / len(model_dict) * 100):.4f}% model params is init!')
    print(f'Drop keys:{[k for k, v in checkpoint["state_dict"].items() if k not in model_dict]}')
    ###
    model.load_state_dict(model_dict)
    print(f'{model_name}_ckpt_pred: {checkpoint["pred"]}')