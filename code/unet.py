"""
UNet model for diffusion processes.
"""

import torch
import torch.nn as nn

from nn import (
    TimestepEmbedSequential, timestep_embedding, conv_nd, linear, 
    ResBlock, AttentionBlock, Downsample, Upsample, normalization, zero_module
)


class UNetModel(nn.Module):
    """The full UNet model with attention and timestep embedding."""
    
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True,
                 dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=1,
                 num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=False,
                 resblock_updown=False, use_new_attention_order=False):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout, out_channels=int(mult * model_channels),
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult),
                        dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        ) if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y=None):
        assert (y is not None) == (self.num_classes is not None)

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class DiffusionInpaintingModel(nn.Module):
    """UNet model specifically designed for inpainting tasks."""
    
    def __init__(self, base_model, in_channels=9):
        super().__init__()
        self.base_model = base_model

        # Modify the first layer to accept inpainting inputs
        old_conv = self.base_model.input_blocks[0][0]
        self.base_model.input_blocks[0] = TimestepEmbedSequential(
            conv_nd(2, in_channels, old_conv.out_channels, 
                   old_conv.kernel_size, old_conv.stride, old_conv.padding)
        )

        # Initialize the new layer weights
        with torch.no_grad():
            # Copy RGB weights to the first 3 channels
            self.base_model.input_blocks[0][0].weight[:, :3] = old_conv.weight
            # Initialize the additional channels with zeros
            self.base_model.input_blocks[0][0].weight[:, 3:] = 0

    def forward(self, x, t, masked_image, mask):
        # Concatenate inputs: [noisy_image, masked_image, mask_3channel]
        inpaint_input = torch.cat([x, masked_image, mask.repeat(1, 3, 1, 1)], dim=1)
        return self.base_model(inpaint_input, t)
