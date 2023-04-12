import argparse
import os
import random
random.seed(2022)

def split_openmic_train(root_dir, valid=0.15):
    """
    Split the openmic training data to separate training and valid

    Parameters
    ----------
    root_dir : str
        directory with openmic_train.json
    valid : float
        percent of validation data
    """
    input_json = os.path.join(root_dir, 'openmic_train.json')
    with open(input_json, 'r') as f:
        data = f.readlines()

    train_json = open(os.path.join(root_dir, 'openmic_train.json'), 'w')
    valid_json = open(os.path.join(root_dir, 'openmic_valid.json'), 'w')
    for line in data:
        if random.uniform(0., 1.) < valid:
            valid_json.write(line)
        else:
            train_json.write(line)
    
    train_json.close()
    valid_json.close()
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--root_dir', type=str, required=True,
                        help="Directory containing openmic_train.json")
    parser.add_argument('-p', '--valid', type=float, default=0.15,
                        help="Percent of validation data")

    args = parser.parse_args()

    split_openmic_train(
        root_dir=args.root_dir,
        valid=args.valid,
    )

