import os
import torch
import json
import random
import re
from PIL import Image
from .vqa_dataset import VQADataset
import utils.data.DST as DST 
from utils.utils import print_rank_0, is_rank_0, get_rank
from .utils import save_debug_image, save_debug_text
import numpy as np
import copy


class HaoyishengDataset(VQADataset):
    def __init__(self, data_path, data_debug_path, per_sample_image, tokenizer, vis_processor, **kwargs):
        vis_root = f"{data_path}/haoyisheng/images"

        ann_path_raw = f"{data_path}/haoyisheng/annotations/anno.json"
        
        toy_anno = [
            [
                {
                    "role": "user",
                    "type": "img",
                    "content": "1",
                },
                {
                    "role": "user",
                    "type": "str",
                    "content": "这个是什么",
                },
                {
                    "role": "assistant",
                    "type": "str",
                    "content": "这个什么都不是",
                },
            ],
            [
                {
                    "role": "user",
                    "type": "img",
                    "content": "2",
                },
                {"role": "assistant", "type": "str", "content": "发生了什么？"},
                {
                    "role": "user",
                    "type": "img",
                    "content": "3",
                },
            ],
            [
                {"role": "user", "type": "str", "content": "Why this happened"},
                {
                    "role": "user",
                    "type": "img",
                    "content": "4",
                },
                {
                    "role": "user",
                    "type": "img",
                    "content": "5",
                },
                {"role": "assistant", "type": "str", "content": "I don't know"},
            ],
            [
                {
                    "role": "user",
                    "type": "img",
                    "content": "6",
                },
                {"role": "assistant", "type": "str", "content": "这个是什么"},
            ],
        ]

        for _ in range(5):
            toy_anno.extend(copy.deepcopy(toy_anno))

        ann_path = f"{data_path}/haoyisheng/haoyisheng.json"
        
        if not os.path.isfile(ann_path):
            print_rank_0(f"HaoyishengDataset: starting an one-time preprocessing:")
            annotations = []
            # raw_annotation = json.load(open(ann_path_raw, "r"))
            raw_annotation = toy_anno
            for raw_ann in raw_annotation:
                meet_criteria = False
                for d_idx in range(len(raw_ann)):
                    if raw_ann[d_idx]["role"] == "user" and raw_ann[d_idx]["type"] == "img":
                        img_id = raw_ann[d_idx]["content"]
                        if os.path.isfile(f"{vis_root}/{img_id}.jpg"):
                            raw_ann[d_idx]["image_path"] = vis_root
                            meet_criteria = True
                if meet_criteria:
                    annotations.append(raw_ann)
            with open(ann_path, 'w') as f:
                json.dump(annotations, f)

        super().__init__(data_path, data_debug_path, per_sample_image, tokenizer, vis_processor,
                         vis_root, [ann_path], **kwargs)
        self.image_tag_dict = [{0: "image a", 1: "image b", 2: "image c", 3: "image d", 4: "image e", 5: "image f", 6: "image g", 7: "image h"},
                               {0: "image A", 1: "image B", 2: "image C", 3: "image D", 4: "image E", 5: "image F", 6: "image G", 7: "image H"},
                               {0: "the first image", 1: "the second image", 2: "the third image", 3: "the fourth image",
                                4: "the fifth image", 5: "the sixth image", 6: "the seventh image", 7: "the eighth image"}]

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_image(self, ann, data_debug_path=None, data_debug_counter=0):
        output_images = []
        img_counter = 0
        for dialogue in ann:
            if dialogue["type"] == "img":
                image_path = os.path.join(dialogue["image_path"], str(dialogue["content"]) + ".jpg")
                save_debug_image(image_path, data_debug_path, data_debug_counter,
                                    get_rank(), img_idx=img_counter)
                img_counter += 1
                image = Image.open(image_path).convert("RGB")

                image = self.vis_processor(image)
                try:
                    image = image['pixel_values'][0]
                except:
                    image = image
                output_images.append(image)
        
        return output_images
    
    def tokenize(self, text, ignore_instruction=True):
        res = self.tokenizer(
            text,
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )
        if res["input_ids"][-1] != self.tokenizer.eos_token_id and self.add_eos:
            res["input_ids"].append(self.tokenizer.eos_token_id)
            res["attention_mask"].append(1)

        # ignore instruction_token
        if ignore_instruction:
            labels = [DST.DEFAULT_LABEL_PADDING_NUM] * len(res["input_ids"])
        else:
            labels = copy.deepcopy(res["input_ids"])

        res.update(labels=labels)
        return res

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=False, num_images=1):
        regex = re.compile(r'((?<=[\.\?!]\s)(\w+)|(^\w+))')
        def capitalize_sentence(match):
            return(match.group().capitalize())
        conv_list = {"input_ids": [], "attention_mask": [], "labels": []}
        tot_num_image = 0
        for dialogue in ann:
            num_image = 0
            if dialogue["role"] == "user" and dialogue["type"] == "str":
                question = DST.DEFAULT_HUMAN_QUESTION_PRETOKEN + '\n' + regex.sub(capitalize_sentence, dialogue["content"]) + '\n\n'
                res = self.tokenize(question, ignore_instruction=self.ignore_instruction)
                conv_list["input_ids"].extend(res["input_ids"])
                conv_list["attention_mask"].extend(res["attention_mask"])
                conv_list["labels"].extend(res["labels"])
            if dialogue["role"] == "user" and dialogue["type"] == "img":
                image =  DST.DEFAULT_HUMAN_IMAGE_PRETOKEN + '\n' + DST.DEFAULT_IMAGE_TOKEN + '\n\n'
                res = self.tokenize(image, ignore_instruction=self.ignore_instruction)
                conv_list["input_ids"].extend(res["input_ids"])
                conv_list["attention_mask"].extend(res["attention_mask"])
                conv_list["labels"].extend(res["labels"])
                num_image += 1
            if dialogue["role"] == "assistant" and dialogue["type"] == "str":
                answer = DST.DEFAULT_ASSISTANT_TOKEN + '\n' + regex.sub(capitalize_sentence, dialogue["content"]) + '\n\n'
                res = self.tokenize(answer, ignore_instruction=False)
                conv_list["input_ids"].extend(res["input_ids"])
                conv_list["attention_mask"].extend(res["attention_mask"])
                conv_list["labels"].extend(res["labels"])

            tot_num_image += num_image

        return [conv_list]

    def __getitem__(self, index):
        ann = self.annotation[index][0] # self.annotation[index] is a list because of "self.annotation = DST.random_grouping(self.annotation, self.per_sample_image)" in VQADataset init
        images_list = self.process_image(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter)
        res_list = self.process_text(ann,
                                    data_debug_path=self.data_debug_path,
                                    data_debug_counter=self.data_debug_counter,
                                    first_message=True,
                                    num_images=len(images_list))

        self.data_debug_counter += 1

        input_ids = []
        attention_mask = []
        labels = []
        for res in res_list:
            input_ids.extend(res["input_ids"])
            attention_mask.extend(res["attention_mask"])
            labels.extend(res["labels"])
        
        res = dict(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        res.update(image=images_list)
        res.update(image_num=len(images_list))

        return res
