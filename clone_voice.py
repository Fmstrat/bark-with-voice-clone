#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio

import torchaudio
import torch

device = 'cuda' # or 'cpu'
model = load_codec_model(use_gpu=True if device == 'cuda' else False)


# In[ ]:


# From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
from hubert.hubert_manager import HuBERTManager
hubert_manager = HuBERTManager()
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()


# In[ ]:


# From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer 
# Load HuBERT for semantic tokens
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer

# Load the HuBERT model
hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(device)

# Load the CustomTokenizer model
tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').to(device)  # Automatically uses the right layers


# In[ ]:


# Load and pre-process the audio waveform
audio_filepath = 'audio.wav' # the audio you want to clone (under 13 seconds)
wav, sr = torchaudio.load(audio_filepath)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.to(device)


# In[ ]:


semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
semantic_tokens = tokenizer.get_token(semantic_vectors)


# In[ ]:


# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav.unsqueeze(0))
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]


# In[ ]:


# move codes to cpu
codes = codes.cpu().numpy()
# move semantic tokens to cpu
semantic_tokens = semantic_tokens.cpu().numpy()


# In[ ]:


import numpy as np
voice_name = sys.argv[1] # whatever you want the name of the voice to be
output_path = 'voices/' + voice_name + '.npz'
np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)


# In[ ]:


# That's it! Now you can head over to the generate.ipynb and use your voice_name for the 'history_prompt'

