import numpy as np
import sys


def compute_accuracy_and_iou(result_blob, label_blob):
    max_label = 2
    class_ious = np.zeros((max_label, 1))
    overall_iou = 0
    overall_accuracy = 0

    tp = np.zeros((max_label))
    fp = np.zeros((max_label))
    fn = np.zeros((max_label))

    img_tp = 0
    img_pixels = 0

    [num_images, channels, img_height, img_width] = label_blob.shape

    for t in range(num_images):
        gt_labels = label_blob[t, 0, :, :]
        result_prob = result_blob[t, :, :, :]
        result_labels = result_prob.argmax(axis=0)

        # IoU computation
        if np.max(result_labels) > max_label - 1:
            print('Result has invalid labels: ' + str(np.max(result_labels)))
        else:
            for class_id in range(0, max_label):
                class_gt = np.equal(gt_labels, class_id)
                class_result = np.equal(result_labels, class_id)
                tp[class_id] = tp[class_id] + np.count_nonzero(class_gt & class_result)
                fp[class_id] = fp[class_id] + np.count_nonzero(class_result & ~class_gt)
                fn[class_id] = fn[class_id] + np.count_nonzero(~class_result & class_gt)

        # Accuracy computation
        img_result = result_labels
        img_gt = gt_labels
        img_tp = img_tp + np.count_nonzero(np.equal(img_gt, img_result))
        img_pixels = img_pixels + np.size(img_gt)

    for class_id in range(0, max_label):
        class_ious[class_id] = tp[class_id] / (tp[class_id] + fp[class_id] + fn[class_id])

    overall_iou = np.mean(class_ious)
    overall_accuracy = img_tp / (img_pixels * 1.0)

    return [overall_accuracy, overall_iou]


if __name__ == "__main__":
    if (sys.argv < 3):
        print('Usage: python ComputeAccuracyAndIoU.py <result_blob>\
              <label_blob>')
    else:
        compute_accuracy_and_iou(sys.argv[1], sys.argv[2])
