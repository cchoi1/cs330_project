import numpy as np
import sklearn.metrics as sk
import torch

recall_level_default = 0.95
cons_level_default = 0.99


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def process_dataset(model, dataloader):
    model.eval()
    preds = []
    targets = []
    all_scores = []

    with torch.no_grad():
        for img, label in dataloader:
            if torch.cuda.is_available():
                img = img.cuda()

            scores = (
                torch.nn.functional.softmax(model.forward(img), dim=1).detach().cpu().numpy()
            )
            max_scores = np.max(a=scores, axis=1, keepdims=False)
            all_scores.append(max_scores)
            targets.append(label.detach().cpu().numpy())

            preds.append(np.argmax(scores, axis=1))

    preds = np.concatenate(preds, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    return preds, all_scores, targets


def calculate_selective_classification_accuracy(model, id_test_dataloader, threshold=0.98):
    preds, all_scores, targets = process_dataset(
        model=model,
        dataloader=id_test_dataloader,
    )

    assert len(all_scores.shape) == 1
    sorted_indices = np.flip(np.argsort(all_scores))
    preds = preds[sorted_indices]
    all_scores = all_scores[sorted_indices]
    targets = targets[sorted_indices]

    cutoff_index = threshold * len(id_test_dataloader.dataset)
    cutoff_index = int(cutoff_index)
    pred_slice = preds[:cutoff_index]
    target_slice = targets[:cutoff_index]

    num_correct = pred_slice[pred_slice == target_slice].shape[0]
    selective_accuracy = float(num_correct) / cutoff_index

    return selective_accuracy

def calculate_selective_classification_coverage(model, id_test_dataloader, threshold=0.98, grid_size=1000):
    preds, all_scores, targets = process_dataset(
        model=model,
        dataloader=id_test_dataloader,
    )

    assert len(all_scores.shape) == 1
    sorted_indices = np.flip(np.argsort(all_scores))
    preds = preds[sorted_indices]
    all_scores = all_scores[sorted_indices]
    targets = targets[sorted_indices]

    coverages = np.array([i for i in range(1, grid_size + 1)], dtype=np.float32)
    coverages /= grid_size
    indices = coverages * len(id_test_dataloader.dataset)

    for i in range(grid_size - 1, -1, -1):
        cutoff_index = int(indices[i])
        pred_slice = preds[:cutoff_index]
        target_slice = targets[:cutoff_index]

        num_correct = pred_slice[pred_slice == target_slice].shape[0]
        selective_accuracy = float(num_correct) / cutoff_index
        if selective_accuracy >= threshold:
            return coverages[i]

    raise ValueError(f"Selective accuracy never reaches threshold {threshold}")


def calculate_selective_classification_auc(
        model, id_test_dataloader, grid_size=1000
):
    preds, all_scores, targets = process_dataset(
        model=model,
        dataloader=id_test_dataloader,
    )

    # auc for accuracy vs coverage
    assert len(all_scores.shape) == 1
    sorted_indices = np.flip(np.argsort(all_scores))
    preds = preds[sorted_indices]
    all_scores = all_scores[sorted_indices]
    targets = targets[sorted_indices]

    coverages = np.array([i for i in range(1, grid_size + 1)], dtype=np.float32)
    coverages /= grid_size
    indices = coverages * len(id_test_dataloader.dataset)

    selective_accuracies = []
    for i in range(grid_size):
        cutoff_index = int(indices[i])
        pred_slice = preds[:cutoff_index]
        target_slice = targets[:cutoff_index]

        num_correct = pred_slice[pred_slice == target_slice].shape[0]
        selective_accuracy = float(num_correct) / cutoff_index
        selective_accuracies.append(selective_accuracy)

    area_under_curve = sk.auc(x=coverages, y=selective_accuracies)

    return area_under_curve


def calculate_fpr_cutoff(id_scores, recall_level):
    # Sort ID scores in increasing order
    id_scores = np.sort(id_scores)
    cutoff_idx = int(recall_level * len(id_scores))
    if recall_level == 1.0:
        cutoff_score = id_scores[-1]
    else:
        cutoff_score = id_scores[cutoff_idx]

    return cutoff_score


def get_measures(
        _pos,
        _neg,
        model,
        id_test_dataloader,
        recall_levels
):
    pos = np.array(_pos[:]).reshape((-1, 1)) # out_score
    neg = np.array(_neg[:]).reshape((-1, 1)) # in_score
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)

    fprs = {}
    for recall_level in recall_levels:
        fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
        fpr_cutoff = calculate_fpr_cutoff(neg, recall_level)
        fprs[recall_level] = [fpr, fpr_cutoff]

    id_scores, ood_scores = neg, pos
    max_ood_score = np.max(ood_scores)
    num_correct = 0
    for i in range(id_scores.shape[0]):
        if id_scores[i] > max_ood_score:
            num_correct += 1

    cons_id_acc = float(num_correct) / id_scores.shape[0]
    sel_class_acc = calculate_selective_classification_accuracy(model, id_test_dataloader)
    sel_class_cov = calculate_selective_classification_coverage(model, id_test_dataloader)
    sel_class_auc = calculate_selective_classification_auc(model, id_test_dataloader)

    return auroc, aupr, fprs, cons_id_acc, sel_class_acc, sel_class_cov, sel_class_auc


def show_performance(pos, neg, model, id_test_dataloader, recall_levels, method_name='Ours'):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fprs, cons_id_acc, sel_class_acc, sel_class_cov, sel_class_auc = get_measures(pos[:], neg[:], model, id_test_dataloader, recall_levels)

    print('\t\t\t' + method_name)
    for recall_level in recall_levels:
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fprs[recall_level][0]))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))


def print_measures(auroc, aupr, fprs, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t\t' + method_name)
    for recall_level in fprs.keys():
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fprs[recall_level][0]))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


# def print_measures_with_std(aurocs, auprs, fprs, method_name='Ours', recall_level=recall_level_default):
#     print('\t\t\t\t' + method_name)
#     print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
#     print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)))
#     print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)))

def print_measures_with_std(aurocs, auprs, fprs, cons_id_accs, sel_class_accs, sel_class_covs, sel_class_aucs, method_name='Ours'):
    print('\t\t\t\t' + method_name)
    for recall_level in fprs[0].keys():
        fpr_list = [fprs[i][recall_level][0] for i in range(len(fprs))]
        print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(fpr_list), 100 * np.std(fpr_list)))
    print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)))
    print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)))
    print('Conservative ID Accuracy:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(cons_id_accs), 100 * np.std(cons_id_accs)))
    print('Selective Classification Accuracy:  \t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(sel_class_accs), 100 * np.std(sel_class_accs)))
    print('Selective Classification Coverage:  \t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(sel_class_covs), 100 * np.std(sel_class_covs)))
    print('Selective Classification AUC:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(sel_class_aucs), 100 * np.std(sel_class_aucs)))


def show_performance_comparison(pos_base, neg_base, pos_ours, neg_ours, baseline_name='Baseline',
                                method_name='Ours', recall_level=recall_level_default):
    '''
    :param pos_base: 1's class, class to detect, outliers, or wrongly predicted
    example scores from the baseline
    :param neg_base: 0's class scores generated by the baseline
    '''
    auroc_base, aupr_base, fpr_base = get_measures(pos_base[:], neg_base[:], recall_level)
    auroc_ours, aupr_ours, fpr_ours = get_measures(pos_ours[:], neg_ours[:], recall_level)

    print('\t\t\t' + baseline_name + '\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
        int(100 * recall_level), 100 * fpr_base, 100 * fpr_ours))
    print('AUROC:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * auroc_base, 100 * auroc_ours))
    print('AUPR:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * aupr_base, 100 * aupr_ours))
    # print('FDR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
    #     int(100 * recall_level), 100 * fdr_base, 100 * fdr_ours))
