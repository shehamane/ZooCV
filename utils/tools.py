def get_conv_out_shape(w, k, s=1, p=0):
    return int((w - k + 2 * p) / s + 1)