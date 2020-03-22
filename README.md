# CREPE Pitch Tracker (PyTorch) #

- Original Tensorflow Implementation : [https://github.com/marl/crepe](https://github.com/marl/crepe)

---
CREPE is a monophonic pitch tracker based on a deep convolutional neural network operating directly on the time-domain waveform input. CREPE is originally implemented with tensorflow, which is very inconvenient framework to use.


## Usage

```python
import torch

# valid model capacity can be 'full', 'large', 'medium', 'small', 'tiny'
model = torch.hub.load('tuan3w/crepe-pytorch', 'load_crepe', 'full')
model.eval()
wav = torch.rand(1, 2000)
predict = model.predict(x, 16000)

# predict from file
model.process_file('path/to/audio/file', output=None)
```





