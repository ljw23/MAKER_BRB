from maker_brb.brb.brb_dataset import BrbDataset

def test_dataset():
    data_file = 'data/demodata.txt'
    dataset = BrbDataset(data_file)

    print(len(dataset))

if __name__ == '__main__':
    test_dataset()