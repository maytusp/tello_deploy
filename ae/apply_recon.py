import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class apply_ae:
    '''
    # To add new backbone
    # 1. Add ae_{backbone}.py file
    # 2. make sure forward() is reconstruction, i.e. forward(x) = dec(enc(x))
    # 3. define block config
    '''
    def __init__(self, device, adapt_encoder=True, apply_ablation=False, ablation_block=None, state_dict_path=None, decode_block="block3", backbone="cliprn50", use_instance=True):
        if backbone == "seresnext50":
            if decode_block == "block4":
                from .nav_vae_4th import AE
                self.dec_block_config_str = "1x1,1u1,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
                self.dec_channel_config_str = "256:64,128:64,64:64,32:128,16:128,8:256,4:512,1:1024"
            elif decode_block == "block3":
                from .nav_vae_3rd import AE
                self.dec_block_config_str = "1x1,1u1,1t4,4x2,4u1,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
                self.dec_channel_config_str = "256:64,128:64,64:64,32:128,16:128,8:256,4:512,1:512"
            elif decode_block == "block2":
                from .nav_vae_2nd import AE
                self.dec_block_config_str = "1x1,1u1,1t4,4x2,4u1,4t8,8x2,8u1,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
                self.dec_channel_config_str = "256:64,128:64,64:64,32:128,16:128,8:256,4:512,1:256"
            elif decode_block == "block1":
                from .nav_vae_1st import AE
                self.dec_block_config_str = "1x1,1u1,1t4,4x2,4u1,4t8,8x2,8u1,8t16,16x6,16u2,16t32,32x2,32u1,32t64,64x2,64u2,64t128,128x1"
                self.dec_channel_config_str = "256:64,128:64,64:64,32:128,16:128,8:256,4:512,1:128"
        elif backbone == "resnet50":
            if decode_block == "block3":
                from .ae_imagenet_3rd import AE
                self.dec_block_config_str = "1x1,1u1,1t4,4x2,4u1,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
                self.dec_channel_config_str = "256:64,128:64,64:64,32:128,16:128,8:256,4:512,1:1024"
            else:
                assert decoder_block == "block3", "support only block3 for ResNet50"
        elif backbone == "cliprn50":
            if decode_block == "block3":
                from .ae_clip import AE
                # from ae_clip import AE
                self.dec_block_config_str = "1x1,1u1,1t4,4x2,4u1,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
                self.dec_channel_config_str = "256:64,128:64,64:64,32:128,16:128,8:256,4:512,1:1024"
            else:
                assert decoder_block == "block3", "support only block3 for CLIP-ResNet50"
        elif backbone == "vqvae": # This one is seresnext50 without norm layers
            if decode_block == "block3":
                from .vqvae_3rd import AE
                self.dec_block_config_str = "1x1,1u1,1t4,4x2,4u1,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1"
                self.dec_channel_config_str = "256:64,128:64,64:64,32:128,16:128,8:256,4:512,1:512"
            else:
                assert decoder_block == "block3", "support only block3 for VQVAE seresnext50"
        self.model = AE(input_res=256, dec_block_str=self.dec_block_config_str, dec_channel_str=self.dec_channel_config_str)
        if state_dict_path is None:
            print("AAE path is not specified")
            assert 1 == 0
            # state_dict_path = "/mnt/iusers01/fatpou01/compsci01/n70579mp/habitat-lab/data/autoencoder/ae_scratch_epoch226.pt"
        # print("state_dict_path=", state_dict_path)
        
        #  # This part: change lighting .ckpt file to .pt file so we can load with habitat conda environment
        # ckpt_path = "/mnt/iusers01/fatpou01/compsci01/n70579mp/robotta_ae/logs/ae_scratch/checkpoints/vae-gibson_scratch-epoch=09.ckpt"
        # ckpt = torch.load(ckpt_path)
        # temp_state_dict = ckpt['state_dict']
        # print("Loaded Keys", temp_state_dict.keys())
        # self.model.load_state_dict(temp_state_dict)
        # torch.save(self.model.state_dict(), state_dict_path)
        # assert 1 == 0
        # #######

        self.state_dict = torch.load(state_dict_path) 
        # print("LOADED_STATE_DICT", self.state_dict.keys())
        self.model.load_state_dict(self.state_dict)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.adapt_encoder = adapt_encoder
        self.apply_ablation = apply_ablation
        self.ablation_block = ablation_block
        # DUA
        self.mom_pre = 0.1
        self.decay_factor = 0.94
        self.min_mom = 0.005
        self.use_instance = use_instance
        # count for printing the first step (debugging purpose)
        self.count = 0
        self.backbone = backbone
    def recon(self, observation):
        self.model.eval()
        # Use DUA to adapt encoder
        if self.adapt_encoder and "vqvae" not in self.backbone:
            if not(self.apply_ablation):
                self.model, self.mom_pre, self.decay_factor, self.min_mom = self._adapt(self.model, 
                                                                                            self.mom_pre, 
                                                                                            self.decay_factor, 
                                                                                            self.min_mom)
            else: # Ablation Study
                self.model, self.mom_pre, self.decay_factor, self.min_mom = self._adapt_ablation(self.model, 
                                                                                            self.mom_pre, 
                                                                                            self.decay_factor, 
                                                                                            self.min_mom)
        observation = observation / 255.0
        x = torch.from_numpy(observation).permute(2, 0, 1).float().unsqueeze(dim=0).to(self.device)
        if not(self.use_instance):
            _ = self.model(x) # To update running statistics
            self.model.eval() # To use running statistics in normalization
        with torch.no_grad():
            decoder_out = self.model(x)
        decoder_out = decoder_out.squeeze().permute(1, 2, 0)
        decoder_out = decoder_out.cpu().detach().numpy()
        decoder_out = (decoder_out*255).clip(0, 255).astype(np.uint8)
        return decoder_out

    def _adapt(self, 
                model,
                mom_pre,
                decay_factor,
                min_mom):
        encoder = model.enc
        mom_new = (mom_pre * decay_factor)
        min_momentum_constant = min_mom
        for m in encoder.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                m.momentum = mom_new + min_momentum_constant
        mom_pre = mom_new
        return model, mom_pre, decay_factor, min_mom

    def _adapt_ablation(self, 
                model,
                mom_pre,
                decay_factor,
                min_mom):
        
        encoder = model.enc
        mom_new = (mom_pre * decay_factor)
        min_momentum_constant = min_mom
        encoder_block = getattr(encoder.backbone, f"layer{self.ablation_block}")
        if self.count == 0:
            print(f"Ablation: allow only block {self.ablation_block} to update norm", encoder_block)
            self.count += 1
        for m in encoder_block.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                m.momentum = mom_new + min_momentum_constant
        mom_pre = mom_new
        return model, mom_pre, decay_factor, min_mom

if __name__ == "__main__":
    device = torch.device("cuda:{}".format(0))
    print(f"device {device}")
    model = apply_ae(device, state_dict_path="/home/maytusp/Projects/drone/pointnav/checkpoints/aae_ckpt_deploy/ae_clip_drone_it.pt")
    print("DONE")
    img = np.zeros((224,224,3))
    output = model.recon(img)
    print("output", output.shape)
