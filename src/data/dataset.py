# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .vqa_rad import encode_question, is_closed, normalize_answer

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def default_transform(image_size=128, train=True, imagenet=False):
    """imagenet=True cho backbone pretrained (ResNet-18); nguoc lai norm 0.5."""
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if imagenet else ([0.5] * 3, [0.5] * 3)
    aug = [transforms.Resize((image_size, image_size)),
           transforms.Grayscale(num_output_channels=3)]
    if train:
        aug.append(transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)))
    aug += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(aug)


class VQARADClsDataset(Dataset):
    """Item: (image[3,S,S], tokens[MAX_LEN], label, closed).

    label = -1 khi answer khong nam trong train vocab (chi o test set):
    khong bao gio match duoc -> tinh sai trong EM, dong protocol voi generation.
    """

    def __init__(self, records, answer2id, q_vocab, transform, max_len=32):
        self.records = records
        self.answer2id = answer2id
        self.q_vocab = q_vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = self.transform(r["image"].convert("RGB"))
        tokens = torch.tensor(
            encode_question(r["question"], self.q_vocab, self.max_len),
            dtype=torch.long)
        ans = normalize_answer(r["answer"])
        label = self.answer2id.get(ans, -1)
        return img, tokens, label, is_closed(ans)
