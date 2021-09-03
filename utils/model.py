import torch


def get_model(repo_or_dir:str, model:str, pretrained:bool=True, *args, **kwargs):
    
    model =\
        torch.hub.load(
            repo_or_dir=repo_or_dir,
            model=model,
            pretrained=pretrained,
            *args, **kwargs
    )   
    return model

    


