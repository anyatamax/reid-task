import json
from pathlib import Path

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from configs.constants import MODELS

from .clip import clip
from .clip.model import build_model
from .clip.simple_tokenizer import SimpleTokenizer


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)

    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        batch_indices = torch.arange(x.shape[0], device=torch.device("cpu"))
        eot_indices = tokenized_prompts.argmax(dim=-1).to(torch.device("cpu"))
        x = x[batch_indices, eot_indices] @ self.text_projection
        return x


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.model.name
        self.cos_layer = cfg.model.cos_layer
        self.neck = cfg.model.neck
        self.neck_feat = cfg.testing.neck_feat
        if self.model_name == "ViT-B-16":
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == "RN50":
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.model.sie_coe

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(
            self.in_planes_proj, self.num_classes, bias=False
        )
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int(
            (cfg.preprocessing.size_train[0] - 16) // cfg.model.stride_size[0] + 1
        )
        self.w_resolution = int(
            (cfg.preprocessing.size_train[1] - 16) // cfg.model.stride_size[1] + 1
        )
        self.vision_stride_size = cfg.model.stride_size[0]
        clip_model = load_clip_to_cpu(
            self.model_name,
            self.h_resolution,
            self.w_resolution,
            self.vision_stride_size,
        )
        # clip_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.image_encoder = clip_model.visual

        if cfg.model.sie_camera and cfg.model.sie_view:
            self.cv_embed = nn.Parameter(
                torch.zeros(camera_num * view_num, self.in_planes)
            )
            trunc_normal_(self.cv_embed, std=0.02)
            print("camera number is : {}".format(camera_num))
        elif cfg.model.sie_camera:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
            print("camera number is : {}".format(camera_num))
        elif cfg.model.sie_view:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
            print("camera number is : {}".format(view_num))

        dataset_name = cfg.dataset.names
        self.prompt_learner = PromptLearner(
            num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding
        )
        
        self.prompt_learner_stage0 = PromptLearnerStage0(
            dtype=clip_model.dtype,
            token_embedding=clip_model.token_embedding
        )
        self.text_encoder = TextEncoder(clip_model)

    def forward(
        self,
        x=None,
        label=None,
        get_image=False,
        get_text=False,
        is_stage0=False,
        cam_label=None,
        view_label=None,
        captions=None,
    ):
        if is_stage0:
            (
                image_features_last,
                image_features,
                image_features_proj,
            ) = self.image_encoder(x)
            
            if self.model_name == "RN50":
                image_features_final = image_features_proj[0]
            elif self.model_name == "ViT-B-16":
                image_features_final = image_features_proj[:, 0]
            else:
                image_features_final = image_features_proj[0]
            
            if captions is not None:
                prompts, tokenized_prompts = self.prompt_learner_stage0(captions)
                text_features = self.text_encoder(prompts, tokenized_prompts)
            else:
                prompts = self.prompt_learner(label, captions)
                text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            
            return {
                'image_features': image_features_final,
                'text_features': text_features
            }
        
        if get_text is True:
            prompts = self.prompt_learner(label, captions)

            if captions is not None:
                # print("Captions in get_text != None")
                text_features_list = []
                for i, prompt in enumerate(prompts):
                    single_prompt = prompt.unsqueeze(0)

                    caption = (
                        captions[i]
                        if i < len(captions) and captions[i] and captions[i].strip()
                        else None
                    )
                    if caption:
                        tokenized_caption = clip.tokenize(
                            self.prompt_learner.tokenizer, caption
                        ).to(single_prompt.device)
                        text_feature = self.text_encoder(
                            single_prompt, tokenized_caption
                        )
                        # print("Text feature from caption: ", text_feature)
                    else:
                        text_feature = self.text_encoder(
                            single_prompt, self.prompt_learner.tokenized_prompts
                        )
                    text_features_list.append(text_feature)

                text_features = torch.cat(text_features_list, dim=0)
            else:
                # print("Captions in get_text = None")
                text_features = self.text_encoder(
                    prompts, self.prompt_learner.tokenized_prompts
                )

            return text_features

        if get_image is True:
            (
                image_features_last,
                image_features,
                image_features_proj,
            ) = self.image_encoder(x)
            if self.model_name == "RN50":
                return image_features_proj[0]
            elif self.model_name == "ViT-B-16":
                return image_features_proj[:, 0]

        if self.model_name == "RN50":
            (
                image_features_last,
                image_features,
                image_features_proj,
            ) = self.image_encoder(x)

            img_feature_last = nn.functional.avg_pool2d(
                image_features_last, image_features_last.shape[2:4]
            ).view(x.shape[0], -1)

            height, width = image_features.shape[2:4]
            if torch.is_tensor(height):
                height = height.item()
                width = width.item()
            img_feature = nn.functional.avg_pool2d(
                image_features, [height, width]
            ).view(x.shape[0], -1)

            img_feature_proj = image_features_proj[0]

        elif self.model_name == "ViT-B-16":
            if cam_label is not None and view_label is not None:
                cv_embed = (
                    self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
                )
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            (
                image_features_last,
                image_features,
                image_features_proj,
            ) = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return (
                [cls_score, cls_score_proj],
                [img_feature_last, img_feature, img_feature_proj],
                img_feature_proj,
            )

        else:
            if self.neck_feat == "after":
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print("Loading pretrained model for finetuning from {}".format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(
        state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size
    )

    return model


class PromptLearnerStage0(nn.Module):
    def __init__(self, dtype, token_embedding):
        super().__init__()
        self.tokenizer = SimpleTokenizer()
        self.token_embedding = token_embedding
        self.dtype = dtype
    
    def forward(self, captions):
        """
        Process captions directly for CLIP training.
        Args:
            captions: List[str] - Raw text captions
        Returns:
            prompts: Tensor - Token embeddings
            tokenized_prompts: Tensor - Token IDs for EOT extraction
        """
        tokenized_prompts = clip.tokenize(self.tokenizer, captions).to(
                    self.token_embedding.weight.device
                )
        prompts = self.token_embedding(tokenized_prompts).type(self.dtype)
        
        return prompts, tokenized_prompts


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()

        if dataset_name == "VehicleID" or dataset_name == "veri":
            self.default_ctx_init = "A photo of a X X X X vehicle."
        else:
            self.default_ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        n_ctx = 4

        self.tokenizer = SimpleTokenizer()
        self.token_embedding = token_embedding
        self.dtype = dtype

        default_tokenized_prompts = clip.tokenize(self.tokenizer, self.default_ctx_init)
        with torch.no_grad():
            default_embedding = token_embedding(default_tokenized_prompts).type(dtype)
        self.tokenized_prompts = default_tokenized_prompts  # Fallback tokenized prompts

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", default_embedding[:, : n_ctx + 1, :])
        self.register_buffer(
            "token_suffix", default_embedding[:, n_ctx + 1 + n_cls_ctx :, :]
        )
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, captions=None):
        if captions is not None:
            return self._forward_with_captions(label, captions)
        else:
            return self._forward_standard(label)

    def _forward_standard(self, label):
        """Standard prompt generation using learnable context."""
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

    def _forward_with_captions(self, label, captions):
        """Generate prompts using individual captions for each image."""
        batch_size = label.shape[0]
        prompts_list = []

        for i in range(batch_size):
            caption = (
                captions[i]
                if i < len(captions) and captions[i] and captions[i].strip()
                else None
            )

            if caption:
                tokenized_caption = clip.tokenize(self.tokenizer, caption).to(
                    self.token_embedding.weight.device
                )
                with torch.no_grad():
                    caption_embedding = self.token_embedding(tokenized_caption).type(
                        self.dtype
                    )
                prompts_list.append(caption_embedding.squeeze(0))
            else:
                cls_ctx = self.cls_ctx[label[i]].unsqueeze(0)
                prefix = self.token_prefix
                suffix = self.token_suffix

                standard_prompt = torch.cat([prefix, cls_ctx, suffix], dim=1)
                prompts_list.append(standard_prompt.squeeze(0))

        prompts = torch.stack(prompts_list, dim=0)
        return prompts
