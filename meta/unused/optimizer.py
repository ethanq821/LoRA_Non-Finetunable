# import torch

# class meta_adamw():
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0: