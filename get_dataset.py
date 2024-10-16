import pandas as pd
import os
import numpy as np
from numpy.linalg import inv
import torch
from notebooks.data_utils import *
import torch.nn.functional as F

label_map = {"Thumb Up":0,"Heart":1, "Stop":2,"paper":3,"Rock":4,
             "Circle":5,"Scissor":6,"Gun":7, "Swipe":8,
              "Two hands scale":9, "Grab things":10,
              "Nozzle rotation":11, "Drive car":12, "Teleport":13,  
              "Two hands delete":14, "Two hands flick":15,"Null":16}
#train_path = '/home/qg224/Gesture_Dataset/gestures/data/train/'
#test_path = '/home/qg224/Gesture_Dataset/gestures/data/test/'
annotation_path = '/home/qg224/Gesture_Dataset/gestures/annotations/'
annotation_path = 'D:/iib_project/data/Gesture_Dataset/gestures/annotations/'

def get_gesture_ids_participant(gesture_name,trainpart, participant_id, path):
    result = []
    for root, subdirs, files in os.walk(path):
        for filename in files:
            if trainpart in filename and gesture_name in filename and 'participant-'+str(participant_id)+'.csv' in filename:
                id = filename.split('-')[3]
                result.append(id)
    return result

def read_gesture_data(path, trainpart, gesture_name, p_id, name_attri):
    anno = pd.read_csv(os.path.join(annotation_path + gesture_name + '.csv'))
    id_list = get_gesture_ids_participant(gesture_name,trainpart, p_id, path)
    df = []
    ppath = trainpart + gesture_name +'-id-'
    for id in id_list:
        df_sub = pd.read_csv(os.path.join(path+ppath+ id +'-participant-'+str(p_id)+'.csv'))
        df.append(df_sub)
    annot_gesture = []
    for i in range(len(df)):
        df_sub = df[i]
        id = id_list[i]
        Time = anno[anno.id == int(id)].iloc[0][name_attri]
        gesture_sub = df_sub.iloc[[Time-1]]
        annot_gesture.append(gesture_sub)
    return annot_gesture


def read_gesture_data_continuous(path, trainpart, gesture_name, p_id):
    anno = pd.read_csv(os.path.join(annotation_path + gesture_name + '.csv'))
    id_list = get_gesture_ids_participant(gesture_name,trainpart, p_id,path)
    df = []
    ppath = trainpart + gesture_name +'-id-'
    for id in id_list:
        df_sub = pd.read_csv(os.path.join(path+ppath+ id +'-participant-'+str(p_id)+'.csv'))
        df.append(df_sub)
    annot_gesture = []
    for i in range(len(df)):
        df_sub = df[i]
        id = id_list[i]
        startTime = anno[anno.id == int(id)].iloc[0]['startTime']
        endTime = anno[anno.id == int(id)].iloc[0]['endTime']
        annot = df_sub[startTime:endTime+1]
        #pos = [col for col in df_sub.columns if ('l' in col or 'r' in col) and 'pos' in col]
        annot_gesture.append(annot)#annot_gesture.append(annot[pos])
    return annot_gesture

def get_absolute_position_bimanual(raw_data):
    pos = [col for col in raw_data[0].columns if (('l' in col)or('r' in col)) and 'pos' in col]
    abs_pos = []

    for j in range(len(raw_data)):
        i=0
        ps=[]
        t = raw_data[j]
        while i < len(pos):
            pps=[]
            pps.append(t[pos[i]].iloc[0])
            pps.append(t[pos[i+1]].iloc[0])
            pps.append(t[pos[i+2]].iloc[0])
            ps.append(pps)
            i+=3
        abs_pos.append(ps)
    return np.array(abs_pos)

def get_R(i,raw_data):
    wrist_pos = [raw_data[i].l0_quat_x.iloc[0], raw_data[i].l0_quat_y.iloc[0], raw_data[i].l0_quat_z.iloc[0], raw_data[i].l0_quat_w.iloc[0]]
    R = np.zeros((3,3))
    x = wrist_pos[0]
    y = wrist_pos[1]
    z = wrist_pos[2]
    w = wrist_pos[3]
    R[0][0] = 1-2*y**2-2*z**2
    R[0][1] = 2*x*y-2*z*w
    R[0][2] = 2*x*z+2*y*w
    R[1][0] = 2*x*y+2*z*w
    R[1][1] = 1-2*x**2-2*z**2
    R[1][2] = 2*y*z-2*x*w
    R[2][0] = 2*x*z-2*y*w
    R[2][1] = 2*y*z+2*x*w
    R[2][2] = 1-2*x**2-2*y**2
    R = np.array(R)
    return R

def get_R_cam(i,raw_data):
    cam_pos = [raw_data[i].cam_quat_x.iloc[0], raw_data[i].cam_quat_y.iloc[0], raw_data[i].cam_quat_z.iloc[0], raw_data[i].cam_quat_w.iloc[0]]
    R = np.zeros((3,3))
    x = cam_pos[0]
    y = cam_pos[1]
    z = cam_pos[2]
    w = cam_pos[3]
    R[0][0] = 1-2*y**2-2*z**2
    R[0][1] = 2*x*y-2*z*w
    R[0][2] = 2*x*z+2*y*w
    R[1][0] = 2*x*y+2*z*w
    R[1][1] = 1-2*x**2-2*z**2
    R[1][2] = 2*y*z-2*x*w
    R[2][0] = 2*x*z-2*y*w
    R[2][1] = 2*y*z+2*x*w
    R[2][2] = 1-2*x**2-2*y**2
    R = np.array(R)
    return R


def get_wrist_abspos(i,raw_data):
    return [raw_data[i].l0_pos_x.iloc[0], raw_data[i].l0_pos_y.iloc[0], raw_data[i].l0_pos_z.iloc[0]]

def get_cam_abspos(i,raw_data):
    return [raw_data[i].cam_pos_x.iloc[0], raw_data[i].cam_pos_y.iloc[0], raw_data[i].cam_pos_z.iloc[0]]


def frame_transformation(raw_data, abs_pos, norm_wrist=True,norm_cam=False):
    transformed_pos = np.zeros_like(abs_pos)
    if norm_wrist:
        for j in range(len(raw_data)):
            for i in range(len(abs_pos[0])):
                transformed_pos[j][i] = np.matmul(inv(get_R_wrist(j,raw_data)),(abs_pos[j][i]-get_wrist_abspos(j,raw_data)))
    if norm_cam:
        for j in range(len(raw_data)):
            for i in range(len(abs_pos[0])):
                transformed_pos[j][i] = np.matmul(inv(get_R_cam(j,raw_data)),(abs_pos[j][i]-get_cam_abspos(j,raw_data)))
    return transformed_pos

def multiframe_transformation(data, norm_wrist=True, norm_cam=False):
    if not data:
        return
    transformed_data = []
    for i in range(len(data)):
        raw_data = [data[i].iloc[[j]] for j in range(len(data[i]))]
        abs_pos = get_absolute_position_bimanual(raw_data)
        transformed = frame_transformation(raw_data,abs_pos,norm_wrist, norm_cam)
        transformed_data.append(transformed.reshape((len(transformed),150)))
    return transformed_data

def get_data_singlestate(path, gesture_dict, p_ids, n_shot, attri='time',length=300, norm_cam=False):  # length=300
    dataset = []
    label = []
    for name in gesture_dict.keys():
        count = 0
        for pid in p_ids:
            if count==n_shot:
                break
            trainpart = gesture_dict[name]
            data = read_gesture_data(path, trainpart, name,pid,attri)
            abs_pos = get_absolute_position_bimanual(data)
            if norm_cam:
                transformed_data = frame_transformation(data, abs_pos,False, True)
            else:
                transformed_data = frame_transformation(data, abs_pos)
            for d in transformed_data:
                tensor = torch.tensor(d.reshape(1,150))
                pad = F.pad(input=tensor, pad=(0, 0, 0, length - 1), mode='constant', value=0)
                dataset.append(pad)
                count+=1
                if count==n_shot:
                    break
        label.extend(label_map[name] for i in range(n_shot))
    dataset = np.asarray(dataset)
    label = np.asarray(label)
    return dataset.reshape(len(dataset),300,50,3).transpose(0, 3, 1, 2), label


def get_data_multiframes(path, gesture_dict,p_ids,n_shot, length=300, norm_cam=False):
    dataset = []
    label = []
    for name in gesture_dict.keys():
        count = 0
        for pid in p_ids:
            if count==n_shot:
                break
            trainpart = gesture_dict[name]
            data = read_gesture_data_continuous(path,trainpart, name,pid)
            if norm_cam:
                transformed_data = multiframe_transformation(data,False, True)
            else:
                transformed_data = multiframe_transformation(data)
            dataset_temp = []
            for i in range(len(data)):
                tensor = torch.tensor(np.asarray(transformed_data[i]))
                dataset_temp.append(F.pad(input=tensor, pad=(0, 0, 0, length - tensor.shape[0]), mode='constant', value=0))
                count+=1
                if count==n_shot:
                    break
            dataset.extend(d for d in dataset_temp)
        label.extend(label_map[name] for i in range(n_shot))
    dataset = np.asarray(dataset)
    label = np.asarray(label)
    return dataset.reshape(len(dataset),300,50,3).transpose(0, 3, 1, 2), label


def get_max_len(path, gesture_dict,p_ids):
    max_len = 0
    for pid in p_ids:
        for name in gesture_dict.keys():
            df = read_gesture_data_continuous(path,gesture_dict[name],name,pid)
            for i in range(len(df)):
                max_len = max(max_len,len(df[i]))
    return max_len

def get_dataset(path, single_frame_dict, multi_frame_dict, p_ids, n_shot, norm_cam=False):
    if norm_cam:
        s_set, s_label = get_data_singlestate(path, single_frame_dict, p_ids, n_shot,norm_cam=True)
        m_set, m_label = get_data_multiframes(path, multi_frame_dict, p_ids, n_shot,norm_cam=True)
    else:
        s_set, s_label = get_data_singlestate(path, single_frame_dict, p_ids, n_shot)
        m_set, m_label = get_data_multiframes(path, multi_frame_dict, p_ids, n_shot)
    d_set = np.concatenate((s_set, m_set), axis=0)
    d_label = np.concatenate((s_label, m_label))
    return d_set, d_label
