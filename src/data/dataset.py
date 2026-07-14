# src/data/dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .vqa_rad import encode_question, encode_sequence, is_closed, normalize_answer

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# Thong ke chuan hoa cua CLIP (openai/clip-vit-base-patch32) cho PubMedCLIP
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def default_transform(image_size=128, train=True, imagenet=False, clip=False):
    """Chuan hoa theo backbone: clip -> CLIP stats, imagenet -> ImageNet, con lai -> 0.5."""
    if clip:
        mean, std = CLIP_MEAN, CLIP_STD
    elif imagenet:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = [0.5] * 3, [0.5] * 3
    aug = [transforms.Resize((image_size, image_size)),
           transforms.Grayscale(num_output_channels=3)]
    if train:
        aug.append(transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)))
    aug += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(aug)


def build_text_tokenizer(text_encoder):
    """HF tokenizer rieng cho text encoder pretrained. None -> tokenize bang q_vocab.

    PubMedBERT phai dung dung tokenizer/vocab cua no; neu khong, id q_vocab la vo nghia.
    """
    if text_encoder == "pubmedbert":
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(
                "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        except Exception as e:
            print(f"⚠️ Khong tai duoc PubMedBERT tokenizer ({e}); fallback q_vocab.")
    return None


class VQARADClsDataset(Dataset):
    """Item: (image[3,S,S], tokens[MAX_LEN], label, closed, dec_in[16], dec_tgt[16]).

    label = -1 khi answer khong nam trong train vocab (chi o test set):
    khong bao gio match duoc -> tinh sai trong EM, dong protocol voi generation.
    """

    def __init__(self, records, answer2id, q_vocab, transform, max_len=32, max_ans_len=16,
                 text_tokenizer=None):
        self.records = records
        self.answer2id = answer2id
        self.q_vocab = q_vocab
        self.transform = transform
        self.max_len = max_len
        self.max_ans_len = max_ans_len
        # None -> tokenize cau hoi bang q_vocab; khac None (HF tokenizer) -> dung cho encoder pretrained
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = self.transform(r["image"].convert("RGB"))
        if self.text_tokenizer is not None:
            enc = self.text_tokenizer(
                str(r["question"]), max_length=self.max_len,
                truncation=True, padding="max_length", return_tensors="pt")
            tokens = enc["input_ids"].squeeze(0).long()   # [max_len]; PAD=0 khop pad_mask
        else:
            tokens = torch.tensor(
                encode_question(r["question"], self.q_vocab, self.max_len),
                dtype=torch.long)
        ans = normalize_answer(r["answer"])
        label = self.answer2id.get(ans, -1)
        
        # Generative sequences
        dec_in = torch.tensor(
            encode_sequence(ans, self.q_vocab, self.max_ans_len, is_target=False),
            dtype=torch.long)
        dec_tgt = torch.tensor(
            encode_sequence(ans, self.q_vocab, self.max_ans_len, is_target=True),
            dtype=torch.long)
            
        return img, tokens, label, is_closed(ans), dec_in, dec_tgt
