import numpy as np

"""
reproject bounding box (xmin, ymin, xmax, ymax) to the transformed grid formed
by the spatial transformer

Return the new bbox in the new grid
In [33]: gt_boxes
Out[33]:
array([[ 109.09091187,   81.8181839 ,  127.27272797,  104.54545593,    1.        ],
   [ 268.18182373,  268.18182373,  318.18182373,  322.7272644 ,    1.        ],
   [  95.45454407,  495.45455933,  131.81817627,  536.36364746,    1.        ]], dtype=float32)

"""

identity = np.array([[1., 0., 0.],
                     [0., 1., 0.]], dtype=np.float32)

def project_bbox(gt_boxes, theta):
    """
    """
    print("*** theta = {0}".format(theta))
    theta = np.reshape(theta, (-1, 3)) # in case it is unflattened
    if (np.sum(theta == identity) == 6):
        return gt_boxes

    num_gtb = gt_boxes.shape[0]
    cls_lbl = np.reshape(gt_boxes[:, -1], [num_gtb, 1])

    boxes = np.transpose(gt_boxes)[0:-1]
    boxes = np.hstack((boxes[0:2, :], boxes[2:4, :])) # shape = (2, num_gtb * 2)
    assert(boxes.shape[1] == num_gtb * 2)
    new_boxes = np.vstack((boxes,  np.ones((1, num_gtb * 2), dtype=np.float32)))
    new_coord = np.dot(theta, new_boxes)
    new_cc = np.transpose(np.vstack((new_coord[:, 0:num_gtb], new_coord[:, num_gtb:num_gtb * 2])))
    return np.hstack((new_cc, cls_lbl))

def project_bbox_inv(pred_boxes, theta):
    """
    pred_boxes.shape == [num_boxes, num_boxes * 4]

    x = (T5 * (x1 - T3) - T2 * (y1 - T6)) / (T1 * T5 - T2 * T4)
    y = (T4 * (x1 - T3) - T1 * (y1 - T6)) / (T2 * T4 - T5 * T1)

    Imagine T1 = 1, T5 = 1, everything else is 0
    """
    theta = np.reshape(theta, (-1, 3)) # in case it is unflattened
    if (np.sum(theta == identity) == 6):
        return

    T1 = theta[0][0]
    T2 = theta[0][1]
    T3 = theta[0][2]
    T4 = theta[1][0]
    T5 = theta[1][1]
    T6 = theta[1][2]
    denominator = T1 * T5 - T2 * T4
    num_row = pred_boxes.shape[0]
    num_col = pred_boxes.shape[1]
    for r in range(num_row):
        for i, xy in enumerate(np.split(pred_boxes[r, :], num_col / 2)):
            x1_T3 = xy[0] - T3
            y1_T6 = xy[1] - T6
            x = (T5 * x1_T3 - T2 * y1_T6) / denominator
            y = (T4 * x1_T3 - T1 * y1_T6) / (-1 * denominator)
            pred_boxes[r, i * 2] = x
            pred_boxes[r, i * 2 + 1] = y
