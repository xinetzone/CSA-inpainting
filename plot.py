def array2image(x):
    x *= 255
    x = x.detach().cpu().numpy()
    return x.astype('uint8').transpose((1, 2, 0))