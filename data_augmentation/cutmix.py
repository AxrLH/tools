import torch
import numpy as np


def random_bbox(shape, lam):
    w = shape[2]
    h = shape[3]

    sqr = np.sqrt(1.0 - lam)

    r_x = np.random.randint(w)
    r_y = np.random.randint(h)
    r_w = np.int(w * sqr)
    r_h = np.int(h * sqr)

    bbox_x1 = np.clip(r_x - r_w // 2, 0, w)
    bbox_x2 = np.clip(r_x + r_w // 2, 0, w)
    bbox_y1 = np.clip(r_y - r_h // 2, 0, h)
    bbox_y2 = np.clip(r_y + r_h // 2, 0, h)

    return bbox_x1, bbox_x2, bbox_y1, bbox_y2


def cutmix(train_loader, model, beta, cutmix_prob, loss_func):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = 0.0
    for i, (input, label) in enumerate(train_loader):
        input = input.to(device)
        label = label.to(device)

        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # 初始化lam
            lam = np.random.beta(beta, beta)
            # 打乱label顺序
            rand_label = torch.randperm(input.size()[0]).to(device)
            label_a = label
            label_b = label[rand_label]

            bbox_x1, bbox_x2, bbox_y1, bbox_y2 = random_bbox(input.size(), lam)
            # begin cutmix operation
            input[:, :, bbox_x1:bbox_x2, bbox_y1:bbox_y2] = input[label_b, :, bbox_x1:bbox_x2, bbox_y1:bbox_y2]

            lam = 1 - ((bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)) / (input.size()[2] * input.size()[3])

            # loss
            output = model(input)
            loss = loss_func(output, label_a) * lam + loss_func(output, label_b) * (1 - lam)

        else:
            output = model(input)
            loss = loss_func(output, label)

    return loss.item()
