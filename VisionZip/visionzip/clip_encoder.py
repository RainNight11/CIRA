import torch
import torch.nn as nn


class CLIPVisionTower_VisionZip(nn.Module):
    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            features = []
            for image in images:
                output = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    output_attentions=True,
                )
                features.append(self.feature_select(output).to(image.dtype))
            return features

        images = images.to(device=self.device, dtype=self.dtype)
        output = self.vision_tower(
            images, output_hidden_states=True, output_attentions=True
        )
        attention = output.attentions[-2]
        hidden = output.hidden_states[-2]
        metric = self.vision_tower.vision_model.encoder.layers[-2].metric

        dominant = int(self.vision_tower._info["dominant"])
        contextual = int(self.vision_tower._info["contextual"])
        cls_attention = attention[:, :, 0, 1:].sum(dim=1)
        dominant_indices = cls_attention.topk(dominant, dim=1).indices + 1
        cls_index = torch.zeros(
            (hidden.shape[0], 1),
            dtype=dominant_indices.dtype,
            device=hidden.device,
        )
        keep_indices = torch.cat((cls_index, dominant_indices), dim=1)

        keep_count = dominant + 1
        mask = torch.ones_like(hidden[:, :, 0], dtype=torch.bool)
        mask.scatter_(1, keep_indices, False)
        dominant_tokens = hidden.masked_select(~mask.unsqueeze(-1)).view(
            hidden.shape[0], keep_count, hidden.shape[2]
        )
        filtered_hidden = hidden.masked_select(mask.unsqueeze(-1)).view(
            hidden.shape[0], -1, hidden.shape[2]
        )
        filtered_metric = metric[mask].view(hidden.shape[0], -1, metric.shape[2])

        if contextual:
            normalized = filtered_metric / filtered_metric.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            step = max(1, normalized.shape[1] // contextual)
            targets = torch.arange(
                0, normalized.shape[1], step=step, device=normalized.device
            )[:contextual]
            target_features = normalized[:, targets]
            all_indices = torch.arange(normalized.shape[1], device=normalized.device)
            merge_mask = ~torch.isin(all_indices, targets)
            to_merge = normalized[:, merge_mask]
            similarity = torch.bmm(to_merge, target_features.transpose(1, 2))
            assignments = torch.zeros(
                to_merge.shape[0],
                to_merge.shape[1],
                contextual,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            assignments.scatter_(2, similarity.argmax(dim=2, keepdim=True), 1)
            counts = assignments.sum(dim=1).clamp_min(1).unsqueeze(-1)
            merged = (
                torch.bmm(assignments.transpose(1, 2), filtered_hidden[:, merge_mask])
                / counts
            )
            contextual_tokens = filtered_hidden[:, targets] + merged
        else:
            contextual_tokens = filtered_hidden[:, :0]

        patches = torch.cat((dominant_tokens[:, 1:], contextual_tokens), dim=1)
        result = torch.cat((dominant_tokens[:, :1], patches), dim=1)
        return result.to(images.dtype), keep_indices
