import os

conf_count_dict = {}


def bbox_count_per_confidence_level():
    for filename in os.listdir('./mAP/predicted/'):
        if filename.endswith('.txt'):
            with open(filename) as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split()
                    conf = int(10 * float(items[1]))
                    if conf not in conf_count_dict.keys():
                        conf_count_dict[conf] = 1
                    else:
                        conf_count_dict[conf] += 1
    count = 0
    for key in sorted(conf_count_dict, reverse=True):
        count += conf_count_dict[key]
        print("Number of bounding boxes greater or equal than confidence {} is {}.".format(float(key) / 10,
              count))


if __name__ == '__main__':
    bbox_count_per_confidence_level()
