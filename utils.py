def process_train_args(arguments):
    num_train_epochs = 50
    batch_size = 50
    learning_rate = 1e-4
    restore_epoch = None
    dataset_dir = './dataset/'

    for args in arguments:

        if args.startswith("num_train_epochs"):
            num_train_epochs = int(args.split("=")[1])

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("restore_epoch"):
            restore_epoch = int(args.split("=")[1])

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

    print("The following parameters will be applied for training:")
    print("Training epochs: " + str(num_train_epochs))
    print("Batch size: " + str(batch_size))
    print("Learning rate: " + str(learning_rate))
    print("Restore epoch: " + str(restore_epoch))
    print("Path to the dataset: " + dataset_dir)

    return num_train_epochs, batch_size, learning_rate, restore_epoch, dataset_dir

def process_test_args(arguments):
    batch_size = 50
    restore_epoch = None
    dataset_dir = './dataset/'

    for args in arguments:

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("restore_epoch"):
            restore_epoch = int(args.split("=")[1])

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

    return batch_size, restore_epoch, dataset_dir

def process_visual_args(arguments):
    restore_epoch = None
    dataset_dir = './dataset/'

    for args in arguments:

        if args.startswith("restore_epoch"):
            restore_epoch = int(args.split("=")[1])

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

    return restore_epoch, dataset_dir

def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std
