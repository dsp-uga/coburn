"""
This gets the max dimensions of frames over all videos
"""
from coburn.data import loader




def main():
    # these samples will be automatically downloaded if they are not found locally
    # dataset = loader.load(samples=['4bad52d5ef5f68e87523ba40aa870494a63c318da7ec7609e486e62f7f7a25e8',
    #                                'a7e37600a431fa6d6023514df87cfc8bb5ec028fb6346a10c2ececc563cc5423',
    #                                '70a6300a00dbac92be9238252ee2a75c86faf4729f3ef267688ab859eed1cc60'])
    dataset = loader.load(samples='all')
    print("samples loaded")
    # transforms = Compose([cuda_transform])
    max_row = 0
    max_cols = 0
    for i in range(len(dataset)):
         #thunder image series
        # print(dataset[i].type())
        #
        # print(dataset[i].shape)
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
