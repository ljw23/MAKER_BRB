from maker_brb.brb.train import Trainer


def test_train():
    trainer = Trainer()
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    test_train()
