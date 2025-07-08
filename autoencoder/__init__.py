def load_autoencoder():
    """
    Load the VGG AutoEncoder model.
    
    Returns:
        VGGAutoEncoder: An instance of the VGGAutoEncoder model.
    """
    from autoencoder.vgg import VGGAutoEncoder
    model = VGGAutoEncoder()

    # load pretrained weights if available
    state = torch.load("autoencoder/imagenet.pth")['state_dict']
    # remove 'module.' prefix if it exists
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model
