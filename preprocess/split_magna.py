import os
import random
random.seed(2022)

def split_openmic_train(root_dir, valid=0.15, test=0.15):
    """
    Split the openmic training data to separate training and valid

    Parameters
    ----------
    root_dir : str
        directory with magna.json
    valid : float
        percent of validation data
    test : float
        percent of test data
    """
    input_json = os.path.join(root_dir, 'magna.json')
    with open(input_json, 'r') as f:
        data = f.readlines()

    train_json = open(os.path.join(root_dir, 'magna_train.json'), 'w')
    valid_json = open(os.path.join(root_dir, 'magna_valid.json'), 'w')
    test_json = open(os.path.join(root_dir, 'magna_test.json'), 'w')
    for line in data:
        t = random.uniform(0., 1.)
        if t > valid + test:
            train_json.write(line)
        elif t < test:
            test_json.write(line)
        else:
            valid_json.write(line)
    
    train_json.close()
    valid_json.close()
    test_json.close()
    
    return


if __name__ == '__main__':
    import fire

    fire.Fire(split_openmic_train)
