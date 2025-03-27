import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def load_model_dict(model, model_path):
    model_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in model_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # 'module.' 제거
        new_state_dict[name] = v

    # 수정된 가중치로 모델 로드
    model.load_state_dict(new_state_dict)
    model.eval()
    return model  # 모델 반환 추가


class CTCLabelConverter(object):
    """ 문자열 레이블과 인덱스 간 변환 클래스 """

    def __init__(self, character):
        # character (str): 문자 집합 문자열 (가능한 모든 문자).
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # 참고: 0은 빈 시퀀스용('blank' 라고도 함)
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # blank 레이블 추가
        
    def encode(self, text, batch_max_length=25):
        """문자를 인덱스로 변환합니다.
        input:
            text: text_line의 리스트  [batch_size]
            batch_max_length: 최대 시퀀스 길이
        output:
            text: 인덱스 텐서 [batch_size x (max_length)] 
            length: 텐서 [batch_size]
        """
        length = [len(s) for s in text]
        
        # 빈 텍스트는 blank로 변환
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)
        
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ 인덱스를 문자열로 변환합니다.
        input:
            text_index: 인덱스 텐서 [batch_size x text_len]
            length: 텐서 [batch_size]
        output:
            text: 텍스트 리스트 [batch_size]
        """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # CTC 디코딩을 위한 중복 제거 
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class AttnLabelConverter(object):
    """ Attention 디코딩을 위한 문자열 레이블-인덱스 변환 """

    def __init__(self, character):
        # character (str): 문자 집합 문자열
        list_token = ['[GO]', '[s]']  # 시작 및 종료 토큰
        list_character = list(character)
        self.character = list_token + list_character
        self.dict = {}
        for i, char in enumerate(self.character):
            # GO, s 토큰이 추가되었으므로 인덱스는 타이밍에 맞춤
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ 문자열을 인덱스로 변환합니다.
        input:
            text: 텍스트 리스트 [batch_size]
            batch_max_length: 최대 이미지 너비
        output:
            text : [batch_size x (max_length+2)] GO와 종료 토큰을 포함
            length : [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # 이 batch에 대한 최대 길이
        batch_max_length += 1
        
        # GO를 추가하지 마세요. 디코더가 이를 추가합니다.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][0:len(text)] = torch.LongTensor(text)
        
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ 인덱스를 문자열로 변환합니다.
        input:
            text_index: 인덱스 텐서 [batch_size x text_len]
            length: 텐서 [batch_size]
        output:
            텍스트 리스트 [batch_size]
        """
        texts = []
        for index, l in enumerate(length):
            # 인덱스에서 문자로 변환
            chars = []
            for i in range(l):
                char_idx = text_index[index, i]
                char = self.character[char_idx]
                # [s] 토큰이 나오면 중지
                if char == '[s]':
                    break
                # [GO] 토큰이 아닌 경우에만 추가
                if char != '[GO]':
                    chars.append(char)
            
            text = ''.join(chars)
            texts.append(text)
        return texts 


# def process_image(image, opt):
#     """
#     이미지를 전처리하여 모델 입력 텐서로 변환합니다.
#     """
#     # 그레이스케일 변환
#     if opt['rgb']:
#         img = image
#     else:
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # 크기 조정
#     h, w = img.shape[:2]
#     ratio = opt['imgH'] / h
#     target_w = int(w * ratio)
    
#     # 너비가 제한을 초과하는 경우 처리
#     if target_w > opt['imgW']:
#         target_w = opt['imgW']
    
#     img = cv2.resize(img, (target_w, opt['imgH']))
    
#     # 패딩 추가
#     if target_w < opt['imgW']:
#         pad_img = np.zeros((opt['imgH'], opt['imgW']), dtype=np.uint8)
#         pad_img[:, :target_w] = img
#         img = pad_img
    
#     # 정규화 및 텐서 변환
#     img = img.astype(np.float32) / 127.5 - 1.0
#     img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
#     return img 