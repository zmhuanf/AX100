from 测试.MJ_test.config import config
from 测试.MJ_test.model import Brain


def main():
    mortal = Brain(version=config['version'], **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    all_models = (mortal, dqn, aux_net)
    if enable_compile:
        for m in all_models:
            m.compile()


if __name__ == '__main__':
    main()
