
import argparse
import os
from data_utils.MRICONV2DataLoader import MRICONV2DatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from riconv2_utils import compute_LRA

classes = ['casting', 'turning','milling']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in testing [default: 4]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point number [default: 2048]')
    parser.add_argument('--log_dir', type=str, default='2024-12-12_00-21', help='experiment root')
    parser.add_argument('--visual', type=bool, default=False, help='visualize result [default: False]')
    parser.add_argument('--test_group', type=int, default=2, help='group for testing, option: 1-2 [default: 2]')
    parser.add_argument('--num_votes', type=int, default=1, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def save_confusion_matrix(cm, classes, save_dir, file_id):
    """
    Save confusion matrix as an image and a text file.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for ' + file_id)
    plt.savefig(os.path.join(save_dir, file_id + '_confusion_matrix.png'))
    plt.close()

    np.savetxt(os.path.join(save_dir, file_id + '_confusion_matrix.txt'), cm, fmt='%d')

def save_classification_report(class_report, save_dir, file_id):
    """
    Save classification report as a text file.
    """
    with open(os.path.join(save_dir, file_id + '_classification_report.txt'), 'w') as f:
        f.write(class_report) 


            
def save_metrics_per_class(metrics, metric_name, class_labels, save_dir, file_id):
    """
    Save calculated metrics per class to a text file.
    """
    if len(metrics) != len(class_labels):
        raise ValueError(f"Mismatch in number of metrics and class labels for {metric_name}")

    with open(os.path.join(save_dir, file_id + f'_{metric_name}.txt'), 'w') as f:
        for i, cls in enumerate(class_labels):
            f.write(f"Class {cls}: {metrics[i]}\n")
def plot_and_save_metrics(metrics, metric_name, class_labels, save_dir, file_id):
    plt.bar(class_labels, metrics)
    plt.xlabel('Classes')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} for each class')
    plt.savefig(os.path.join(save_dir, file_id + f'_{metric_name}.png'))
    plt.close()
    
def save_confusion_matrix_plot(cm, class_name, save_dir, file_id):
    """
    Generate and save a plot of the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for ' + file_id)
    plt.savefig(os.path.join(save_dir, file_id + '_confusion_matrix.png'))
    plt.close()

def main(args):
    
    def log_string(str):
        logger.info(str)
        print(str)
     

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 3
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    
    root = '../data/s3dis/mriconv2_3d/'
    TEST_DATASET_WHOLE_SCENE = MRICONV2DatasetWholeScene(root, split='test', test_group=args.test_group, block_points=2048)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    MODEL = importlib.import_module('riconv2_sem_seg')
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')


        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        all_true_labels = []
        all_pred_labels = []
        # Initialize variables for TP, TN, FP, FN for each class
        tp = {cls: 0 for cls in classes}
        tn = {cls: 0 for cls in classes}
        fp = {cls: 0 for cls in classes}
        fn = {cls: 0 for cls in classes}
        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
    
                num_blocks = scene_data.shape[0]
 
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data[:,:,:6])
                    norm = compute_LRA(torch_data[:, :, :3], True)
                    torch_data[:,:,3:6] = norm
                    torch_data = torch_data.float().cuda()
                    # torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])
                    
                    pred_label = np.argmax(vote_label_pool, 1)
                    all_true_labels.extend(whole_scene_label.tolist())
                    all_pred_labels.extend(pred_label.tolist())
                  
                    
            # Create variables to keep track of IoU for each class
            casting_iou = 0.0
            turning_iou = 0.0

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]
        
        
        
                true_label = whole_scene_label  # Assuming whole_scene_label contains true labels for this part

            # Compute confusion matrix for "Casting" vs. "Turning"vs. "Milling" for this part
            cm = confusion_matrix(true_label, pred_label, labels=[class2label['casting'], class2label['turning'], class2label['milling']])
            print(f"Confusion Matrix for Part {scene_id[batch_idx]} (Casting vs. Turningvs. Milling):\n", cm)

            # Save the confusion matrix data for this part
            np.savetxt(os.path.join(results_dir, f'part_{scene_id[batch_idx]}_confusion_matrix.txt'), cm, fmt='%d')

            # Save the confusion matrix plot for this part
            save_confusion_matrix_plot(cm, ['Casting', 'Turning','Milling'], results_dir, f'part_{scene_id[batch_idx]}_confusion_matrix')
    
            # Calculate IoU for each class
            casting_iou = total_correct_class_tmp[class2label['casting']] / float(total_iou_deno_class_tmp[class2label['casting']] + 1e-6)
            turning_iou = total_correct_class_tmp[class2label['turning']] / float(total_iou_deno_class_tmp[class2label['turning']] + 1e-6)
            milling_iou = total_correct_class_tmp[class2label['milling']] / float(total_iou_deno_class_tmp[class2label['milling']] + 1e-6)
           
            # Print IoU values for each class
            log_string('IoU for Casting: %.4f' % casting_iou)
            log_string('IoU for Turning: %.4f' % turning_iou)
            log_string('IoU for Milling: %.4f' % milling_iou)
            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')
            
        
            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()
            for i in range(whole_scene_label.shape[0]):
                color = g_label2color[pred_label[i]]
                color_gt = g_label2color[whole_scene_label[i]]
                if args.visual:
                    fout.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                        color[2]))
                    fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                            color_gt[1], color_gt[2]))
            if args.visual:
                fout.close()
                fout_gt.close()
        
        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
     
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))
        
        print("Done!")
       
        # Compute the standard confusion matrix
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[class2label['casting'], class2label['turning'],class2label['milling']])
        print(f"Standard Confusion Matrix for Casting vs. Turning vs. Milling:\n", cm)
        save_confusion_matrix(cm, ['Casting', 'Turning','Milling'], results_dir, 'casting_vs_turning_vs_milling_standard')

       # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Normalized Confusion Matrix for Casting vs. Turning vs. Milling:\n", cm_normalized)
        save_confusion_matrix(cm_normalized, ['Casting', 'Turning','Milling'], results_dir, 'casting_vs_turning_vs_milling_normalized')

   

        # Calculate precision, recall, F1 score, and MCC for each class
        precision_per_class = precision_score(all_true_labels, all_pred_labels, average=None)
        recall_per_class = recall_score(all_true_labels, all_pred_labels, average=None)
        f1_per_class = f1_score(all_true_labels, all_pred_labels, average=None)
        

        for metric_name, metric_values in zip(['Precision', 'Recall', 'F1 Score'], 
                                          [precision_per_class, recall_per_class, f1_per_class]):
            save_metrics_per_class(metric_values, metric_name, classes, results_dir, 'classification')
        
       
        
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
