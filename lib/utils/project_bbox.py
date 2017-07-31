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

def project_bbox(gt_boxes, theta):
    """
    """
    num_gtb = gt_boxes.shape[0]
    theta = np.reshape(theta, (-1, 3)) # in case it is unflattened
    cls_lbl = np.reshape(gt_boxes[:, -1], [num_gtb, 1])

    boxes = np.transpose(gt_boxes)[0:-1]
    boxes = np.hstack((boxes[0:2, :], boxes[2:4, :])) # shape = (2, num_gtb * 2)
    assert(boxes.shape[1] == num_gtb * 2)
    new_boxes = np.vstack((boxes,  np.ones((1, num_gtb * 2))))
    new_coord = np.dot(theta, new_boxes)
    new_cc = np.transpose(np.vstack((new_coord[:, 0:num_gtb], new_coord[:, num_gtb:num_gtb * 2])))
    return np.hstack((new_cc, cls_lbl))
