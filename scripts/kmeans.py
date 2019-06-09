import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def write_clusters(clusters, output_file):
    clusters = sorted(clusters.tolist(), key=lambda x: x[0]*x[1])
    f = open(output_file, 'w')
    for i in xrange(len(clusters)):
        if i == 0:
            x_y = "%d,%d" % (clusters[i][0], clusters[i][1])
        else:
            x_y = ", %d,%d" % (clusters[i][0], clusters[i][1])
        f.write(x_y)
    f.close()

def parse_annotations(input_file):
    dataset = []
    with open(input_file, 'r') as f:
        max_width = 0
        max_height = 0
        for line in f:
            infos = line.strip().split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                if (width > 0 and height > 0):
                    dataset.append([width, height])
                    print('found box with non-zero area')
                else:
                    print('skip box with 0 area')
                print(infos[i])
                if width > max_width:
                    max_width = width
                if height > max_height:
                    max_height = height
        result = np.array(dataset)
        print("max_width: {}".format(max_width))
        print("max_height: {}".format(max_height))
        return result

if __name__ == "__main__":
    cluster_number = 9
    input_file = "/home/yuan/Learning/cs230/backup/annotations/dataset/open_image_train_v2.txt"
    output_file = "/home/yuan/Learning/cs230/backup/yolo_anchors.txt"
    # Row format: image_file_path box1 box2 ... boxN;
    # Box format: x_min,y_min,x_max,y_max,class_id (no space).
    bboxes = parse_annotations(input_file)
    clusters = kmeans(bboxes, cluster_number)
    write_clusters(clusters, output_file)
    print("K anchors:\n {}".format(clusters))
    print("Accuracy: {:.2f}%".format(avg_iou(bboxes, clusters) * 100))
