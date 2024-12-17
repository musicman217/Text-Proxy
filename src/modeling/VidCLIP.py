import torch
import torch.nn as nn
from functools import partial
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from src.modeling.CLIP_ViP import CLIPModel, clip_loss
from src.modeling.CLIP import CLIPModel as CLIP
from src.module.text_proxy import TextProxy

class VidCLIP(nn.Module):
    def __init__(self, args):
        super(VidCLIP, self).__init__()
        clipconfig = CLIPConfig.from_pretrained(args.clip_config) # openai/clip-vit-base-patch32
        setattr(clipconfig, "vision_additional_config", args.clip_vision_additional_config)
        self.vision_additional_config = args.clip_vision_additional_config # ViP
        if args.clip_weights: # openai/clip-vit-base-patch32
            if self.vision_additional_config.type == "ViP":
                self.clipmodel = CLIPModel.from_pretrained(args.clip_weights, config=clipconfig)
            else:
                self.clipmodel = CLIP.from_pretrained(args.clip_weights, config=clipconfig)
        else:
            if self.vision_additional_config.type == "ViP":
                self.clipmodel = CLIPModel(clipconfig)
            else:
                self.clipmodel = CLIP(clipconfig)
        
        # init logit scale from 
        logit_scale_value = self.vision_additional_config.logit_scale_init_value # 4.60
        self.clipmodel.logit_scale.data.fill_(logit_scale_value)

        # text proxy
        self.text_proxy = TextProxy(args)



    def overload_logit_scale(self, overload_logit_scale):
        self.clipmodel.logit_scale.data.fill_(overload_logit_scale)

    def forward(self, is_train, step, video, text_input_ids, text_input_mask,\
                image=None, caption_ids=None, caption_masks=None):
        """
        video [B, n_clips*num_frms, C, H, W]
        text_input_ids [B, L]
        text_input_mask [B, L]
        image [B, img_num, C, H, W]
        caption_ids [B, img_num, L]
        caption_masks [B, img_num, L]
        """
        B, N, C, H, W = video.shape

        if self.vision_additional_config.type == "ViP":
            inputs = {"input_ids": text_input_ids,
                    "attention_mask": text_input_mask,
                    "pixel_values": video,
                    "return_loss": False,
                    }
            outputs = self.clipmodel(**inputs)
            results = {}

            results["text_features"] = outputs["text_embeds"]
            results["vis_features"] = outputs["image_embeds"] # (b,dim)

            M = self.vision_additional_config.add_cls_num + 1
            results['vis_patch_features'] = outputs['vision_model_output'][0][:, :M, :]


            # for text proxy learning
            vis_patch_feat = results['vis_patch_features']  # (b,M,dim)
            vis_feat = results["vis_features"] # (b,dim)
            text_feat = results["text_features"] # (a,dim)

            if is_train:
                text_proxy = self.text_proxy(text_feat, vis_patch_feat, vis_patch_feat, is_train=is_train, step=step) # (a,b,dim)
                results['text_proxy'] = text_proxy




        else:
            video = video.reshape(-1, C, H, W)
            inputs = {"input_ids": text_input_ids,
                    "attention_mask": text_input_mask,
                    "pixel_values": video}
            outputs = self.clipmodel(**inputs)
            vis_features = outputs["vision_model_output"][1]

            vis_features = self.clipmodel.visual_projection(vis_features)
            vis_features = vis_features / vis_features.norm(dim=-1, keepdim=True)
            vis_features = vis_features.reshape(B, N, -1).mean(1)
            vis_features = vis_features / vis_features.norm(dim=-1, keepdim=True)
            
            results = {}
            results["text_features"] = outputs["text_embeds"]
            results["vis_features"] = vis_features
        if image is not None:
            B, img_num, C, H, W = image.shape
            L = caption_ids.shape[-1]
            inputs = {"input_ids": caption_ids.reshape(-1, L),
                    "attention_mask": caption_masks.reshape(-1, L),
                    "pixel_values": image.reshape(-1, 1, C, H, W),
                    "return_loss": False}
            outputs = self.clipmodel(**inputs)
            results["img_features"] = outputs["image_embeds"]
            results["cap_features"] = outputs["text_embeds"]
        
        return results
    
    def forward_video(self, video):
        inputs = {"pixel_values": video,
                "if_norm": True}
        video_features = self.clipmodel.get_image_features(**inputs)
        return video_features
    
    def forward_text(self, text_input_ids, text_input_mask):
        inputs = {"input_ids": text_input_ids,
                "attention_mask": text_input_mask,
                "if_norm": True}
        text_features = self.clipmodel.get_text_features(**inputs)
        return text_features

    def freeze_text_encoder(self, freeze_text_proj):
        freeze_list = [self.clipmodel.text_model]
        if freeze_text_proj:
            freeze_list.append(self.clipmodel.text_projection)
        for m in freeze_list:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def sim_proxy(self, text_proxy, vid_embeds, text_embeds=None, is_train=True):
        """

        :param text_embeds: (a,b,dim)
        :param vid_embeds: (b,dim)
        :return:
        """
        text_proxy = text_proxy / text_proxy.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) if text_embeds is not None else None
        vid_embeds = vid_embeds / vid_embeds.norm(dim=-1, keepdim=True)

        proxy_logits = torch.matmul(text_proxy.unsqueeze(2),vid_embeds.unsqueeze(2)).squeeze()
        # (a,b,1,dim)x(b,dim,1)->(a,b,1,1)->(a,b)
        # proxy_logits = proxy_logits.max(2)[0]
        # ->(a,b)


        if is_train:
            # hard negative
            # 对第i个text的第i个proxy与所有video进行相似度计算
            pos_indices = torch.arange(text_proxy.size(0))  # [0,1,...,a-1] for training as they have same batch size
            pos_proxy = text_proxy[pos_indices, pos_indices]  # ->(a,512)表示第i个text的第i个proxy
            ## 和所有video进行相似度计算
            pos_logits = torch.matmul(pos_proxy.unsqueeze(1), vid_embeds.transpose(0,1)).squeeze()
            # (a,1,dim)x(dim,b)->(a,1,b)->(a,b)

            # proxy regularization
            # 先使得正样本proxy距离video的距离比text距离video的距离更远
            proxy_dist = torch.sqrt(torch.sum((text_proxy - vid_embeds)**2, dim=-1)) # (a,b,dim)-(b,dim)->(a,b)
            text_dist = torch.sqrt(torch.sum((text_embeds - vid_embeds)**2, dim=-1)).unsqueeze(1) # (a,dim)-(b,dim)->(a=b,1)
            contrast_logits = proxy_dist / text_dist




            return proxy_logits, pos_logits, contrast_logits

        return proxy_logits
