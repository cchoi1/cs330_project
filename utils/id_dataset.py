import torch
from torch.utils.data import Dataset
import numpy as np

to_np = lambda x: x.data.cpu().numpy()

def get_misclassified_examples(net, loader):
    assert loader.batch_size == 1
    misclassified_idxs = []
    net.eval()
    with torch.no_grad():
        for example_idx, (data, target) in enumerate(loader):
            data = data.cuda()

            output = net(data)
            smax = to_np(torch.nn.functional.softmax(output, dim=1))

            pred = np.argmax(smax, axis=1)[0]
            target = target.numpy()[0]

            if not pred == target:
                misclassified_idxs.append(example_idx)

    return misclassified_idxs