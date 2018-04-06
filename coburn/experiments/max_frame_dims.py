"""
This gets the max dimensions of frames over all videos
"""
from coburn.data import loader




def main():
    dataset = loader.load(samples='all')
    print("samples loaded")
    max_row = 0
    max_cols = 0
    for i in range(len(dataset)):

        tmp_row = dataset[i].shape[1]
        print(tmp_row)
        tmp_cols = dataset[i].shape[2]
        print(tmp_cols)
        if tmp_row > max_row:
            max_row = tmp_row
            print("updating rows to: {}".format(max_row))
        if tmp_cols > max_cols:
            max_cols = tmp_cols
            print("updating cols to: {}".format(max_cols))
    print("Final max rows: {}".format(max_row))
    print("Final max cols: {}".format(max_cols))
