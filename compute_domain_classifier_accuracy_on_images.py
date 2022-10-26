import os
import sys
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, adjusted_mutual_info_score
import torch
import torch.nn as nn
from tqdm import tqdm
from embedding_domain_and_class_dataset import EmbeddingDomainAndClassDataset
from train_domain_classifier import load_model_and_domain_names

EMBEDDING_SIZE = 768
VAL_BATCH_SIZE = 512
SAMPLING_NUM_IMAGES_PER_DOMAIN = [100, 1000, 10000]

def do_sampling_analysis(all_log_probs, all_gts, domain_names, experiment_dir):
    sampling_queues = []
    for domain in tqdm(range(len(domain_names))):
        pairs = sorted([(log_prob, gt) for log_prob, gt in zip(all_log_probs[:,domain], all_gts)], key=lambda p: p[0], reverse=True)
        sampling_queues.append([p[1] for p in pairs])

    sampling_makeup_mat_as_cond_probs = {}
    for N in tqdm(SAMPLING_NUM_IMAGES_PER_DOMAIN):
        y_true = []
        y_pred = []
        for domain in tqdm(range(len(domain_names))):
            y_true.append((domain * np.ones(N)).astype(all_gts.dtype))
            y_pred.append(sampling_queues[domain][:N])

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        sampling_makeup_mat_as_cond_probs[N] = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(sampling_makeup_mat_as_cond_probs[N], display_labels=domain_names)
        disp.plot(include_values=False, xticks_rotation='vertical')
        disp.figure_.savefig(os.path.join(experiment_dir, 'domain_classifier_sampling_makeup_mat_on_images_%09d.png'%(N)), dpi=1000)

    return sampling_makeup_mat_as_cond_probs

def just_make_the_plots(experiment_dir):
    with open(os.path.join(experiment_dir, 'domain_classifier_results_on_images.pkl'), 'rb') as f:
        results = pickle.load(f)

    with open(os.path.join(experiment_dir, 'train_domain_filter.pkl'), 'rb') as f:
        domain_names = pickle.load(f)

    disp = ConfusionMatrixDisplay(results['confusion_mat_as_cond_probs'], display_labels=domain_names)
    disp.plot(include_values=False, xticks_rotation='vertical')
    disp.figure_.savefig(os.path.join(experiment_dir, 'domain_classifier_confusion_mat_on_images.png'), dpi=1000)
    for N in tqdm(SAMPLING_NUM_IMAGES_PER_DOMAIN):
        disp = ConfusionMatrixDisplay(results['sampling_makeup_mat_as_cond_probs'] [N], display_labels=domain_names)
        disp.plot(include_values=False, xticks_rotation='vertical')
        disp.figure_.savefig(os.path.join(experiment_dir, 'domain_classifier_sampling_makeup_mat_on_images_%09d.png'%(N)), dpi=1000)


def compute_domain_classifier_accuracy_on_images(experiment_dir, image_embedding_dict_filename_prefix, image_base_dir):
    experiment_dir = os.path.expanduser(experiment_dir)
    image_embedding_dict_filename_prefix = os.path.expanduser(image_embedding_dict_filename_prefix)
    image_base_dir = os.path.expanduser(image_base_dir)

    if os.path.exists(os.path.join(experiment_dir, 'domain_classifier_results_on_images.pkl')):
        just_make_the_plots(experiment_dir)
        return

    model, domain_names = load_model_and_domain_names(experiment_dir, EMBEDDING_SIZE, device='cuda')
    model.eval()
    test_dataset = EmbeddingDomainAndClassDataset(image_embedding_dict_filename_prefix, 'image', domain_filter=domain_names, base_dir=image_base_dir, image_shots_per_domainclass=None)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    all_preds = []
    all_gts = []
    all_log_probs = []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            embeddings, domains = batch['embedding'].to('cuda'), batch['domain'].cpu().numpy()
            logits = model(embeddings)
            log_probs = (logits - torch.logsumexp(logits, dim=1, keepdim=True)).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(domains)
            all_log_probs.append(log_probs)

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    all_log_probs = np.concatenate(all_log_probs)
    accuracy_as_percentage = 100.0 * np.sum(all_preds == all_gts) / len(all_gts)
    print('experiment_dir = %s'%(experiment_dir))
    print('accuracy = %.1f%%'%(accuracy_as_percentage))
    confusion_mat_as_cond_probs = confusion_matrix(all_gts, all_preds, normalize='true')
    adjusted_mutual_info = adjusted_mutual_info_score(all_gts, all_preds)
    print('adjusted_mutual_info = %.3f'%(adjusted_mutual_info))
    results = {'accuracy_as_percentage' : accuracy_as_percentage, 'confusion_mat_as_cond_probs' : confusion_mat_as_cond_probs, 'adjusted_mutual_info' : adjusted_mutual_info}
    results['sampling_makeup_mat_as_cond_probs'] = do_sampling_analysis(all_log_probs, all_gts, domain_names, experiment_dir)
    with open(os.path.join(experiment_dir, 'domain_classifier_results_on_images.pkl'), 'wb') as f:
        pickle.dump(results, f)

    disp = ConfusionMatrixDisplay(confusion_mat_as_cond_probs, display_labels=domain_names)
    disp.plot(include_values=False, xticks_rotation='vertical')
    disp.figure_.savefig(os.path.join(experiment_dir, 'domain_classifier_confusion_mat_on_images.png'), dpi=1000)

def usage():
    print('Usage: python compute_domain_classifier_accuracy_on_images.py <experiment_dir> <image_embedding_dict_filename_prefix> <image_base_dir>')

if __name__ == '__main__':
    compute_domain_classifier_accuracy_on_images(*(sys.argv[1:]))
