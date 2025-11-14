"""
Text-based Graph Sampling for CLIP-ReID
Based on "Graph Sampling Based Deep Metric Learning for Generalizable Person Re-Identification"
by Shengcai Liao and Ling Shao (arXiv:2104.01546)
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from random import shuffle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler

from utils.iotools import get_img_name_for_captions
from utils.metrics import cosine_similarity


class TextGraphSampler(Sampler):
    """
    Graph sampler that uses text embeddings from captions to build similarity graph.

    Args:
        data_source: Dataset with (img_path, pid, camid, trackid) format
        model: CLIP-ReID model for extracting text features
        captions_json_path: Path to captions JSON file
        batch_size: Mini-batch size (P * K)
        num_instance: Number of instances per class (K)
        device: Device for computation
        verbose: Whether to print progress
    """

    def __init__(
        self, data_source, model, captions_map, batch_size, num_instance, verbose=False
    ):
        super(TextGraphSampler, self).__init__(data_source)

        self.data_source = data_source
        self.model = model
        self.batch_size = batch_size
        self.num_instance = num_instance
        self.verbose = verbose
        self.device = next(model.parameters()).device  # Get device from model

        self.captions_map = captions_map
        if self.verbose:
            print(f"TextGraphSampler: Loaded {len(self.captions_map)} captions")

        self.index_dic = defaultdict(list)
        self.pid_to_paths = defaultdict(list)

        for index, (img_path, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
            self.pid_to_paths[pid].append(img_path)

        self.pids = list(self.index_dic.keys())
        self.num_pids = len(self.pids)
        self.num_classes_per_batch = self.batch_size // self.num_instance

        for pid in self.pids:
            shuffle(self.index_dic[pid])

        self.sam_index = None
        self.sam_pointer = [0] * self.num_pids
        self.similarity_graph = None

        if self.verbose:
            print(
                f"TextGraphSampler initialized: {self.num_pids} classes, "
                f"batch_size={batch_size}, num_instance={num_instance}"
            )

    def _get_caption_for_image(self, img_path):
        img_filename = img_path.split("/")[-1]

        caption_key = get_img_name_for_captions(self.captions_map, img_filename)
        if caption_key:
            img_captions = self.captions_map[caption_key]
            return np.random.choice(img_captions)

        return None

    def _extract_text_features_for_classes(self):
        start_time = time.time()
        class_text_features = []
        class_labels = []

        with torch.no_grad():
            for pid in self.pids:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                img_paths = self.pid_to_paths[pid]
                selected_path = np.random.choice(img_paths)

                caption = self._get_caption_for_image(selected_path)
                if caption:
                    captions_batch = [caption]
                else:
                    captions_batch = None

                label_tensor = torch.tensor([pid]).to(self.device)

                text_features = self.model(
                    label=label_tensor, get_text=True, captions=captions_batch
                )

                # No need to normalize - cosine_similarity function handles normalization internally
                # Нормализуем features для матричного умножения
                text_features = torch.nn.functional.normalize(text_features.cpu(), p=2, dim=1)
                class_text_features.append(text_features)
                class_labels.append(pid)

        if len(class_text_features) > 0:
            class_text_features = torch.cat(class_text_features, dim=0)
            print(
                f"TextGraphSampler: len(class_text_features): {len(class_text_features)}"
            )
        else:
            class_text_features = torch.zeros(len(self.pids), 512)
            print(f"TextGraphSampler: len(class_text_features): 0")

        if self.verbose:
            print(f"TextGraphSampler: Features shape: {class_text_features.shape}")

        return class_text_features, class_labels

    def _build_similarity_graph(self, features):
        # Lower values = more similar
        # dist_mat = cosine_similarity(features, features)
        # similarities = torch.from_numpy(dist_mat)

        features_tensor = (
            torch.stack(features) if isinstance(features, list) else features
        )
        # Матричное умножение: higher values = more similar
        similarity_matrix = torch.matmul(features_tensor, features_tensor.t())
        # Конвертируем similarity в distance: lower values = more similar
        similarities = 1.0 - similarity_matrix

        similarities.fill_diagonal_(float("inf"))

        topk = self.num_classes_per_batch - 1  # P-1 neighbors

        if topk >= similarities.shape[1]:
            topk = similarities.shape[1] - 1

        if topk <= 0:
            topk_indices = torch.arange(similarities.shape[0]).unsqueeze(1)
        else:
            _, topk_indices = torch.topk(similarities, topk, dim=1, largest=False)

        if self.verbose:
            print(f"TextGraphSampler: Graph built with top-{topk} neighbors per class")
            for i in range(min(3, similarities.shape[0])):
                top_distances = similarities[i][topk_indices[i]]
                print(
                    f"  Class {i}: distances to top neighbors: {top_distances.numpy()[:3]}"
                )

        return topk_indices.cpu().numpy()

    def make_index(self):
        class_features, _ = self._extract_text_features_for_classes()
        similarity_graph = self._build_similarity_graph(class_features)
        
        self.similarity_graph = similarity_graph

        sam_index = []
        for i, pid in enumerate(self.pids):
            if similarity_graph.shape[1] > 0:
                connected_indices = similarity_graph[i].tolist()
                connected_indices.append(i)
            else:
                connected_indices = [i]

            batch_indices = []
            for class_idx in connected_indices:
                if class_idx >= len(self.pids):
                    continue

                target_pid = self.pids[class_idx]
                available_indices = self.index_dic[target_pid]

                sampled_indices = []
                remain = self.num_instance
                while remain > 0:
                    start_ptr = self.sam_pointer[class_idx]
                    end_ptr = min(start_ptr + remain, len(available_indices))

                    batch_indices_for_class = available_indices[start_ptr:end_ptr]
                    sampled_indices.extend(batch_indices_for_class)
                    remain -= len(batch_indices_for_class)

                    self.sam_pointer[class_idx] = end_ptr
                    if end_ptr >= len(available_indices):
                        shuffle(available_indices)
                        self.sam_pointer[class_idx] = 0

                batch_indices.extend(sampled_indices)

            sam_index.extend(batch_indices)

        sam_index = np.array(sam_index)

        total_samples = len(sam_index)
        num_complete_batches = total_samples // self.batch_size

        if num_complete_batches > 0:
            complete_samples = num_complete_batches * self.batch_size
            sam_index = sam_index[:complete_samples]
            sam_index = sam_index.reshape((-1, self.batch_size))
            np.random.shuffle(sam_index)
            sam_index = sam_index.flatten().tolist()
        else:
            sam_index = sam_index.tolist()

        self.sam_index = sam_index

        if self.verbose:
            print(f"TextGraphSampler: Generated {len(self.sam_index)} sampling indices")

    def __len__(self):
        if self.sam_index is None:
            return len(self.data_source)
        return len(self.sam_index)

    def __iter__(self):
        if self.sam_index is None:
            print(
                "⚠️  Warning: Graph sampling index not built, calling make_index()..."
            )
            self.make_index()
        return iter(self.sam_index)
