import copy
from pathlib import Path
import random
from statistics import mean
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from notebooks.get_dataset import get_dataset
import torch
from torch.utils.data import Dataset
from easyfsl.modules import resnet12
from easyfsl.methods import PrototypicalNetworks, FewShotClassifier
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

train_path = '/home/qg224/Gesture_Dataset/gestures/data/train/'
annotation_path = '/home/qg224/Gesture_Dataset/gestures/annotations/'
trainpart1_path = 'train part1-'
trainpart2_path = 'train part2-'
trainpart3_path = 'train part3-'
p_ids = [2,5,6,7,8,9,10,11,23,24]
single_frame_gestures = {'Thumb Up': trainpart1_path, 'Stop':trainpart1_path, 
                         'Heart':trainpart2_path}

multiframe_gestures = {'Scissor':trainpart2_path,'Gun':trainpart1_path, 
                       'Swipe':trainpart1_path, "Two hands scale":trainpart3_path}

single_frame_gestures_val = {'paper':trainpart2_path, 'Thumb Up': trainpart1_path}
multiframe_gestures_val = {'Scissor':trainpart2_path, "Drive car":trainpart3_path,
                            "Two hands delete":trainpart3_path, "Two hands scale":trainpart3_path}

test_path = '/home/qg224/Gesture_Dataset/gestures/data/test/'
testpart1_path = 'test 1-'
testpart2_path = 'test 2-'
testpart3_path = 'test 3-'
single_frame_gestures_test = {'Rock':testpart3_path, 'Circle':testpart2_path}
multiframe_gestures_test = {'Grab things':testpart2_path, 'Nozzle rotation':testpart1_path,
                            "Teleport":testpart3_path, "Two hands flick":testpart3_path,
                           'Null': testpart1_path}


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

def training_epoch(
    model: FewShotClassifier, data_loader: DataLoader, optimizer: Optimizer
):
    all_loss = []
    model.train()
    with tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Training"
    ) as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            optimizer.zero_grad()
            model.process_support_set(
                support_images.to(DEVICE), support_labels.to(DEVICE)
            )
            classification_scores = model(query_images.to(DEVICE))

            loss = LOSS_FUNCTION(classification_scores, query_labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)

if __name__ == "__main__":
    n_way = 5
    n_train = 70
    n_val = 10
    n_test = 10
    n_shot = 5
    n_query = 5
    n_tasks_per_epoch = 100
    n_validation_tasks = 100
    n_test_tasks = 100

    n_epochs = 100
    batch_size = 32    
    n_workers = 12
    scheduler_milestones = [120, 160]
    scheduler_gamma = 0.1
    learning_rate = 1e-2
    tb_logs_dir = Path("/home/qg224/fsl_gesture/log/")
    DEVICE = "cuda"
    store_dir = '/home/qg224/fsl_gesture/model/meta_5way5shot_model.pt'
       
    print("-------loading dataset--------")
    train, train_label = get_dataset(train_path, single_frame_gestures,multiframe_gestures, p_ids,n_train)
    train_dataset = GestureDataset(train, train_label)
    val, val_label = get_dataset(train_path, single_frame_gestures_val,multiframe_gestures_val, p_ids,n_val)
    val_dataset = GestureDataset(val, val_label)
    test, test_label = get_dataset(test_path, single_frame_gestures_test,multiframe_gestures_test, p_ids,n_test)
    test_dataset = GestureDataset(test, test_label)
    
    train_sampler = TaskSampler(
        train_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch
    )
    val_sampler = TaskSampler(
        val_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )
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

    convolutional_network = resnet12()
    few_shot_classifier = PrototypicalNetworks(convolutional_network).to(DEVICE)
    
    LOSS_FUNCTION = nn.CrossEntropyLoss()

    train_optimizer = SGD(
        few_shot_classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    train_scheduler = MultiStepLR(
        train_optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )

    tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))
    best_state = few_shot_classifier.state_dict()
    best_validation_accuracy = 0.0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        average_loss = training_epoch(few_shot_classifier, train_loader, train_optimizer)
        validation_accuracy = evaluate(
            few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(few_shot_classifier.state_dict())

        tb_writer.add_scalar("Train/loss", average_loss, epoch)
        tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

        train_scheduler.step()
    
    print("-------complete training--------")
    few_shot_classifier.load_state_dict(best_state, strict=False)
    torch.save(few_shot_classifier, store_dir)
    
    ''' Test stage '''
    print("-------evaluation--------")
    accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE)
    print(f"Average accuracy : {(100 * accuracy):.2f} %")