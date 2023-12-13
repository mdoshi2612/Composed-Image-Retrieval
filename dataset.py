from builtins import RuntimeWarning
from builtins import NotImplementedError
import os, sys
import numpy as np
import PIL
from PIL import Image
import warnings
import random
import pickle

import torch
import json
import torch.utils.data
import torchvision
from torchvision import transforms as tfms
import torch, logging
import torch.nn as nn

import matplotlib.pyplot as plt
logging.disable(logging.WARNING)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from IPython.display import display
import shutil
import os

## For video display
from IPython.display import HTML
from base64 import b64encode

PIL.Image.MAX_IMAGE_PIXELS = 1000000000
PIL.Image.warnings.simplefilter('error', PIL.Image.DecompressionBombWarning)

# For Aux Annotation in CIRR
WORD_REPLACE = {
  '[c] None existed': 'noneexisted',
  '[cr0] Nothing worth mentioning': 'nothingworth',
  '[cr1] Covered in query': 'coveredinquery',
}

class BaseDataset(torch.utils.data.Dataset):
  """Base class for a dataset.
  This portion is based on the TIRG implementation,
  see https://github.com/google/tirg.
  """

  def __init__(self):
    super(BaseDataset, self).__init__()
    self.imgs = []
    self.test_queries = []

    # self.logger = logger
    # self.logger.write(''); self.logger.write('Start init BaseDataset class...')
  
  def get_loader(self, batch_size, shuffle=False, drop_last=False):
    # self.logger.write('\nNum_worker: %i, pin_memory: %s' % (num_workers, str(pin_memory)))
    return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=lambda i: i)

  def get_all_texts(self):
    raise NotImplementedError
  
  def get_test_queries(self):
    return self.test_queries if self.split != 'train' else self.train_queries

  def generate_random_query_target(self):
    raise NotImplementedError
  
  def get_img(self, idx, raw_img=False):
    raise NotImplementedError

  def __getitem__(self, idx):
    return self.generate_random_query_target()

class CIRR(BaseDataset):
  """ The CIRR dataset.
  This is partially based on TIRG implementation and Fashion-IQ implementation
  see https://github.com/google/tirg, https://github.com/XiaoxiaoGuo/fashion-iq.
  """
  def __init__(self, path, processor, image_encoder, tokenizer, text_encoder, split='train', val_loader=None, device = "cuda:7"):
    super(CIRR, self).__init__()
    '''stable dataset version, DO NOT CHANGE unless you are sure
    corresponding dataset version can be found in our repository
    https://github.com/Cuberick-Orion/CIRR/tree/cirr_dataset
    '''

    self.device = device
   
    self.processor = processor
    self.image_encoder = image_encoder

    self.tokenizer = tokenizer
    self.text_encoder = text_encoder
    
    self.version = 'rc2' 
      
    assert split in ['train', 'val', 'test1']
    self.split = split

    assert val_loader in [None, 'img+txt', 'img'] # if None, then proceed as original, if otherwise, return will be different
    self.val_loader = val_loader

    self.img_path = path
    data = { # hold all data read from json files
        'image_splits': {},
        'captions': {},
        'captions_ext': {} 
    }
     # load the corresponding json files
    for subfolder_name in data:
      for json_name in os.listdir(path + '/' + subfolder_name):
        if (split == 'train' and 'train' in json_name) \
        or (split == 'val' and 'val' in json_name) \
        or (split == 'test1' and 'test' in json_name):
          json_load = json.load(open(path + '/' + subfolder_name + '/' + json_name))
          data[subfolder_name][json_name] = json_load
    
    imgs = []
    asin2id = {}; id2asin=[]
    for json_name in data['image_splits']:
      for asin_,img_path_ in data['image_splits'][json_name].items():
        asin2id[asin_] = len(imgs)
        id2asin.append(asin_)
        imgs += [{
          'asin': asin_,
          'img_feat_res152_path': os.path.join(self.img_path, 'img_feat_res152', img_path_.replace('.png','.pkl')),
          'captions': [asin2id[asin_]],
          'img_raw_path': os.path.join(self.img_path, 'img_raw', img_path_), #! Uncomment this line if raw img is downloaded
        }]

    # process queries from loaded data
    queries = []
    for json_name in data['captions']:
      for query in data['captions'][json_name]:
        if self.split != 'test1':
          query['source_id'] = asin2id[query['reference']]
          query['target_id'] = asin2id[query['target_hard']]
          query['captions'] = [query['caption']]
          query['target_soft_id'] = {asin2id[kkk]:vvv for kkk,vvv in query['target_soft'].items()}
          queries += [query]
        else:
          query['source_id'] = asin2id[query['reference']]
          query['captions'] = [query['caption']]
          queries += [query]

    # add Aux Annoation from cap.ext
    queries_temp = {qqq['pairid']:qqq for qqq in queries} 
    for kkk,qqq in queries_temp.items():
      queries_temp[kkk]['caption_extend'] = None
    for json_name in data['captions_ext']:
      for query in data['captions_ext'][json_name]:
        query_cap_ext_ = {}
        for kkk_,vvv_ in query['caption_extend'].items():
          if vvv_ in WORD_REPLACE.keys():
            query_cap_ext_[kkk_] = WORD_REPLACE[vvv_]
          else:
            query_cap_ext_[kkk_] = vvv_
        queries_temp[query['pairid']]['caption_extend'] = query_cap_ext_
    queries = [qqq for kkk,qqq in queries_temp.items()]
   
    self.data = data
    self.imgs = imgs
    self.asin2id = asin2id; self.id2asin = id2asin
    self.queries = queries

    # prepare a copy of test_queries from queries
    if split in ['train', 'val']:
      self.test_queries = [{
        'source_img_id': query['source_id'],
        'target_img_id': query['target_id'],
        'source_caption': query['source_id'],
        'target_caption': query['target_id'],
        'target_caption_soft': query['target_soft_id'],
        'set_member_idx': [self.asin2id[ii] for ii in query['img_set']['members'] if ii != query['reference']],
        'mod': {'str': query['captions'][0], **query['caption_extend']},
        'caption_ext': query['caption_extend'],
        'pairid': query['pairid']
      } for _, query in enumerate(queries)]
    elif split == 'test1': 
      self.test_queries = [{
        'source_img_id': query['source_id'],
        'source_caption': query['source_id'],
        'set_member_idx': [self.asin2id[ii] for ii in query['img_set']['members'] if ii != query['reference']],
        'mod': {'str': query['captions'][0], **query['caption_extend']},
        'caption_ext': query['caption_extend'],
        'pairid': query['pairid']
      } for _, query in enumerate(queries)]

  def __len__(self):
    if self.split == 'train' and not self.val_loader: # in training
      return len(self.imgs)
    else: # in validation/test
      if self.val_loader == 'img+txt':
        return len(self.test_queries)
      elif self.val_loader == 'img':
        return len(self.imgs)

  def __getitem__(self, idx):
      generated_ = self.generate_random_query_target()
      return self.image_enc(generated_['source_image_data']), self.image_enc(generated_['target_image_data']), self.text_enc(generated_['mod']['str'])
  
  def get_triplet(self, idx):
      generated_ = self.generate_random_query_target()
      return {'source_image':generated_['source_image_data'][0],
              'source_image_embeddings':self.image_enc(generated_['source_image_data']),
              'target_image':generated_['target_image_data'][0],
              'target_image_embeddings':self.image_enc(generated_['target_image_data']),
              'caption':generated_['mod']['str'],
              'caption_embeddings':self.text_enc(generated_['mod']['str'])}

  def image_enc(self, images):
      input = self.processor(images = images, return_tensors = "pt").to(self.device)
      return self.image_encoder.get_image_features(**input)

  def text_enc(self, prompts, maxlen=None):
      '''
      A function to take a texual prompt and convert it into embeddings
      '''
      if maxlen is None: maxlen = self.tokenizer.model_max_length
      inp = self.tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
      return self.text_encoder(inp.input_ids.to(self.device))[0]

  def get_imgs_in_set(self, set_member_idx):
    if not set_member_idx is None:
        img_feats = []
        img_feat_ = np.stack([self.get_img(d) for d in set_member_idx])
        img_feat_ = torch.from_numpy(img_feat_).float()
        img_feats.append(img_feat_)
        return img_feats
    else:
        return None
  
  def generate_random_query_target(self):
    query_idx = random.choice(range(len(self.queries)))
    query = self.queries[query_idx]
    
    mod_str = query['captions'][0]
    mod_str_ext = query['caption_extend'] # The Aux Annotation
    
    other_set_member_asin = [ii for ii in query['img_set']['members'] if ii != query['reference']]
    other_set_member_idx = [self.asin2id[ii] for ii in other_set_member_asin]

    target_soft_within_set = {kkk:vvv for kkk,vvv in query['target_soft'].items() if kkk in query['img_set']['members'] and kkk != query['reference']} # only consider things in set, filter out target == reference image which is just mistakes
    
    random_stageI_idx = None

    return {
      'source_image_id': query['source_id'],
      'target_image_id': query['target_id'],
      'source_image_data': [self.get_img(query['source_id'])],
      'target_image_data': [self.get_img(query['target_id'])],
      'mod': {'str': mod_str, **mod_str_ext},
      'target_caption': query['target_id'],
    }
  
  def get_img(self, idx):
      img_path = self.imgs[idx]['img_raw_path']
      with open(img_path, 'rb') as f:
        img = PIL.Image.open(f)
        img = img.convert('RGB')
        return img



