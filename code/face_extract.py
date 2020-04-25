from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
from collections import defaultdict
import os
import pickle
import sys
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2 as cv

os.makedirs('/content/drive/My Drive/kaggle/batch03/', exist_ok=True)
os.chdir('/content/drive/My Drive/kaggle/batch03/')

DATA_FOLDER = '/content/drive/My Drive/kaggle'
TRAIN_SAMPLE_FOLDER = 'batch03/dfdc_train_part_47'
TEST_FOLDER = 'test_videos'

train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))
json_file = [file for file in train_list if  file.endswith('json')][0]

def sampleFacesFromVid(link, offset=0, interval=30, mtcnn=MTCNN(margin=20, keep_all=True, post_process=False, device='cuda:0'), showimg=False):
  try:
    print("Extracting Face from: ", link)
    # Load a video
    v_cap = cv.VideoCapture(link)
    v_len = int(v_cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Loop through video, taking a handful of frames to form a batch
    frames = []
    for i in tqdm(range(v_len)):
        
        # Load frame
        success = v_cap.grab()
        if i % interval == offset:
            success, frame = v_cap.retrieve()
        else:
            continue
        if not success:
            continue
            
        # Add to batch
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    # Detect faces in batch
    faces = mtcnn(frames)
    
    if showimg:
      fig, axes = plt.subplots(len(faces), 2, figsize=(6, 15))
      for i, frame_faces in enumerate(faces):
        for j, face in enumerate(frame_faces):
          axes[i, j].imshow(face.permute(1, 2, 0).int().numpy())
          axes[i, j].axis('off')
      fig.show()

    return faces
  except:
    print("Failed for file: ", link)

def saveFacesToPics(faces, dirPrefix=""):
  try:
    d = defaultdict(list)
    for i, frame_faces in enumerate(faces):
      for j, face in enumerate(frame_faces):
        faceObj = face.permute(1, 2, 0).int().numpy().astype(np.float32)
        d[j].append(faceObj)

    count = max([len(v) for k, v, in d.items()])
    for k, v in d.items():
      if len(v) == count:
        if dirPrefix:
          os.makedirs(dirPrefix + "face_" + str(k), exist_ok=True)  # succeeds even if directory exists.
        # Save files
        num = 0
        for img in v:
          cv.imwrite(os.path.join(dirPrefix + "face_" + str(k), str(num).zfill(2) + '.png'), cv.cvtColor(img, cv.COLOR_RGB2BGR))
          num += 1
  except:
    print("Failed for file: ", dirPrefix)

def saveFacesToArray(faces):
  try:
    d = defaultdict(list)
    for i, frame_faces in enumerate(faces):
      for j, face in enumerate(frame_faces):
        faceObj = face.permute(1, 2, 0).int().numpy().astype(np.float32)
        d[j].append(faceObj)

    count = max([len(v) for k, v, in d.items()])

    data = []
    for k, v in d.items():
      if len(v) == count:
        data.append([cv.cvtColor(img, cv.COLOR_RGB2BGR) for img in v])

    return data

  except:
    print("Failed!")
    return []

def saveBatch(data, labels, name="output"):
  d = defaultdict(list)

  d['y'] = labels

  for frameList in data:
    for i in range(0, 10):
      d['x' + str(i)].append(frameList[i])

  d = dict(d)
  np.save(name, np.array(dict(d)))

def get_meta_from_json(path):
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df


mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cuda:0')

train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))
meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)
meta_train_df.head()

passNum, passIdx = 0, 0
batchNum, batchIdx, batchCount = 256, 0, 0
data, target = [], []

for file in train_list:
    if file == "metadata.json":
        continue

    if passNum > passIdx:
        passIdx += 1
        continue
  
    print(batchIdx)

    label = meta_train_df.loc[file, 'label']
    faces = sampleFacesFromVid(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, file), mtcnn=mtcnn)
    arr = saveFacesToArray(faces)
    data.extend(arr)
    target.extend([label for _ in arr])
    batchIdx += 1

    if label == 'REAL':
        # Multiple sample for real
        print("Multiple sample for REAL: ", file)
        for ofst in [8, 16, 24]:
            faces = sampleFacesFromVid(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, file), offset=ofst, interval=30, mtcnn=mtcnn)
            arr = saveFacesToArray(faces)
            data.extend(arr)
            target.extend([label for _ in arr])
            batchIdx += 1

    if batchIdx >= batchNum:
        saveBatch(data, target, name="batch_" + str(batchCount))
        data.clear()
        target.clear()
        batchIdx = 0
        batchCount += 1

saveBatch(data, target, name="batch_" + str(batchCount))