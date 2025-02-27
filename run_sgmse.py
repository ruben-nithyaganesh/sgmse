from sgmse.data_module import SpecsDataModule
from sgmse.model import ScoreModel
import torchaudio
import matplotlib.pyplot as plt
import torch

torch.serialization.add_safe_globals([SpecsDataModule])
wav, sr = torchaudio.load("../data/p257_051.wav")

params = torch.load("../checkpoints/sgmse.ckpt", map_location='cpu', weights_only=True)
#print(params)

sgmse_ckpt = "../checkpoints/sgmse.ckpt"
vb_ckpt = "../checkpoints/train_vb_29nqe0uh_epoch=115.ckpt"

net = ScoreModel.load_from_checkpoint(vb_ckpt)

stft = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=1024, hop_length=512)
spec = stft(wav)

print(wav)

wav_enhanced = torch.tensor(net.enhance(wav)).unsqueeze(0)

print(wav_enhanced)

plt.imshow(stft(wav).log2()[0].numpy(), cmap="viridis", aspect="auto", origin="lower")
plt.title("Original")
plt.show()

plt.imshow(stft(wav_enhanced).log2()[0].numpy(), cmap="viridis", aspect="auto", origin="lower")
plt.title("Enchanced")
plt.show()
