from utils.model import get_model

def test_build_model():
    repo_or_dir = "pytorch/vision"
    model       = "resnet50" 
    get_model(
        repo_or_dir=repo_or_dir,
        model=model
    )