import copy
from pathlib import Path
import random
from statistics import mean
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from notebooks.get_dataset import *
import torch
from torch.utils.data import Dataset
from easyfsl.modules import resnet12
from easyfsl.methods import PrototypicalNetworks
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from easyfsl.utils import evaluate

random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GestureDataset(Dataset):
    def __init__(self, gesture_data, labels):
        self.data = gesture_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gesture_sample = torch.tensor(self.data[idx],dtype=torch.float)
        label = torch.tensor(self.labels[idx].astype(np.int64))
        
        return (gesture_sample, label)
    def get_labels(self):
        return self.labels


def training_epoch(model_: nn.Module, data_loader: DataLoader, optimizer: Optimizer):
    all_loss = []
    model_.train()
    with tqdm(data_loader, total=len(data_loader), desc="Training") as tqdm_train:
        for images, labels in tqdm_train:
            optimizer.zero_grad()

            loss = LOSS_FUNCTION(model_(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)


train_path = '/home/qg224/Gesture_Dataset/gestures/data/train/'
annotation_path = '/home/qg224/Gesture_Dataset/gestures/annotations/'
trainpart1_path = 'train part1-'
trainpart2_path = 'train part2-'
trainpart3_path = 'train part3-'
p_ids = [2,5,6,7,8,9,10,11,23,24]
single_frame_gestures = {'Thumb Up': trainpart1_path, 'Stop':trainpart1_path, 
                         'Heart':trainpart2_path}
multiframe_gestures = {'Gun':trainpart1_path, "Two hands delete":trainpart3_path}

single_frame_gestures_val = {'paper':trainpart2_path, 'Heart': trainpart1_path}
multiframe_gestures_val = {'Gun':trainpart1_path, 'Nozzle rotation':trainpart1_path,
                           "Two hands scale":trainpart3_path}

test_path = '/home/qg224/Gesture_Dataset/gestures/data/test/'
testpart1_path = 'test 1-'
testpart2_path = 'test 2-'
testpart3_path = 'test 3-'
single_frame_gestures_test = {'Rock':testpart3_path, 'Circle':testpart2_path}
multiframe_gestures_test = {'Scissor':trainpart2_path, "Teleport":trainpart3_path,
                            "Two hands flick":trainpart3_path}


if __name__ == "__main__":
    n_train_classes = 3
    n_way = 5
    n_train = 70
    n_val = 10
    n_test = 10
    n_shot = 3
    n_query = 7
    n_validation_tasks = 200
    n_epochs = 100
    batch_size = 16    
    n_workers = 12
    scheduler_milestones = [150, 180]
    scheduler_gamma = 0.1
    learning_rate = 1e-01
    tb_logs_dir = Path("/home/qg224/fsl_gesture/log/")
    DEVICE = "cuda"
    store_dir = '/home/qg224/fsl_gesture/model/classical_5way3shot_static_model.pt'
    
    print("-------loading dataset 53--------")
    train_1, train_label_1 = get_data_singlestate(train_path, single_frame_gestures, p_ids,n_train)
    train_2, train_label_2 = get_data_singlestate(train_path, multiframe_gestures, p_ids,n_train, attri='startTime')
    train = np.concatenate((train_1, train_2), axis=0)
    train_label = np.concatenate((train_label_1, train_label_2))
    train_dataset = GestureDataset(train, train_label)
    val_1, val_label_1 = get_data_singlestate(train_path, single_frame_gestures_val, p_ids,n_val)
    val_2, val_label_2 = get_data_singlestate(train_path, multiframe_gestures_val, p_ids, n_val, attri='startTime')
    val = np.concatenate((val_1, val_2), axis=0)
    val_label = np.concatenate((val_label_1, val_label_2))
    val_dataset = GestureDataset(val, val_label)
    test_1, test_label_1 = get_data_singlestate(test_path, single_frame_gestures_test, p_ids,n_test)
    test_2, test_label_2 = get_data_singlestate(test_path, multiframe_gestures_test, p_ids,n_test, attri='startTime')
    test = np.concatenate((test_1, test_2), axis=0)
    test_label = np.concatenate((test_label_1, test_label_2))
    test_dataset = GestureDataset(test, test_label)

    model = resnet12(
        use_fc=True,
        num_classes=n_train_classes,
    ).to(DEVICE)
    
    few_shot_classifier = PrototypicalNetworks(model).to(DEVICE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_sampler = TaskSampler(
        val_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )
        
    n_test_tasks = 100
    test_sampler = TaskSampler(
        test_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    LOSS_FUNCTION = nn.CrossEntropyLoss()

    train_optimizer = SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))

    best_state = model.state_dict()
    best_validation_accuracy = 0.0
    validation_frequency = 10
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = training_epoch(model, train_loader, train_optimizer)

        if epoch % validation_frequency == validation_frequency - 1:
            model.set_use_fc(False)
            validation_accuracy = evaluate(
                few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
            )
            model.set_use_fc(True)
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_state = copy.deepcopy(few_shot_classifier.state_dict())
            tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)
        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        train_scheduler.step()
    print("-------complete training--------")
    model.load_state_dict(best_state, strict=False)
    torch.save(model, store_dir)
    
    ''' Test stage '''
    print("-------evaluation--------")
    model.set_use_fc(False)
    accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE)
    print(f"Average accuracy : {(100 * accuracy):.2f} %")