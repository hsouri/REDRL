import json
import os.path as osp
import time
import sys
import pdb
sys.path.append('.')


class Configs():
    def __init__(self):
        # ----------------------------------------
        # Data Configuration
        # ----------------------------------------
        self.Dataset = 'MNIST'
        # Size of input images to the pipeline
        self.Input_Size = [64, 64]
        # Values to be used for image normalization
        self.Input_Pixel_MEAN = [0.0, 0.0, 0.0]
        self.Input_Pixel_STD = [1.0, 1.0, 1.0]
        # Dataloader configuration
        self.Num_Workers = 16
        # Sub Sampling Dataset
        self.sub_sample_ratio = 1.0


        # ----------------------------------------
        # Model Configuration
        # ----------------------------------------
        
        # Generator architecture Type
        self.Generator_Type =  'NRP' #'SegNet_VAE'  # 'Unet'
        self.Generator_Pretrained = False
        self.Generator_Pretrained_Path = ''

        # Discriminator architecture Type
        self.Discriminator_Type = 'PatchGan'
        self.Gan_Mode = 'Vanilla'

        # Unet Configs
        self.Input_nc = 3
        self.Output_nc = 3
        self.Ngf = 64
        self.Ndf = 64
        self.N_layers = 3
        self.Use_dropout = False
        self.Norm_layer = 'batch'
        self.Init_type = 'normal'
        self.Init_gain = 0.02

        # Image Classifier Architecture
        self.Image_Classifier = 'resnet34'
        self.Num_Classes = 200

        # Attack Type Classifier Architecture 
        self.Attack_Classifier = 'resnet18'
        self.Attack_Classifier_Cat = False
        self.Attack_Classifier_Pretrained = False
        self.Num_Attack_Types = 5
        self.Label_Smoothing = False
        self.Label_Smoothing_epsilon = 0.2
        self.Neck = 'no' # no or bnneck
        # GPU/CPU selection
        self.Device = 'cuda'
        self.GPU_IDs = (0,1)
        
        # ----------------------------------------
        # Solver Configuration
        # ----------------------------------------
        
        # Training configuration
        self.Baseline = False
        self.Baseline_Source = 'IMG' # 'RES'
        self.Optimizer = 'Adam'
        self.Num_Epochs = 30
        self.Base_Learning_Rate = 1e-4
        self.Bias_Learning_Rate_Factor = 1
        self.Momentum = 0.9
        self.Nestrov = True
        self.Weight_Decay = 5e-4
        self.Weight_Decay_Bias = 0
        self.Gamma = 0.1
        self.Steps = (15, 25)
        self.Warmup_Factor = 1.0 / 3
        self.Warmup_Iters = 10
        self.Warmup_Method = 'linear'
        self.Checkpoint_Period = 50 
        self.Log_Period = 500 
        self.Eval_Period = 50
        self.Train_Batch_Size = 128
        self.Val_Batch_Size = 32
        self.Test_Batch_Size = 128
        self.Reconstruction_Loss = True
        self.L2 = True
        self.Reconstruction_Loss_Lambda = 10
        self.KLD_Loss_Lambda = 1
        self.Perceptual_Loss = True
        self.Perceptual_Loss_Lambda = 1
        self.Adversarial_Loss = True
        self.Adversarial_Loss_Lambda = 1
        self.Image_Classification_Loss = True
        self.Image_Classification_Loss_Lambda = 1
        self.Attack_Det_Loss = False
        self.Attack_Det_Loss_Lambda = 1
        self.Attack_Type_Triplet_Loss = False
        self.Attack_Type_Triplet_Margin = 0.3
        self.DataLoader_Num_Instaces = 4
        # Test Configuration
        self.TEST_ATTACK = 'SSP'
        self.TEST_EPS = 16
        self.TEST_NORM = 'linf'
        # MISC Configuration
        self.Output_Path = ''

        self.TEST_ADV_DATA = ''
        self.TEST_CKPT_PATH = ''
        self.Test_Epoch = 16
        self.do_save = False
        self.Save_Path = ''

    def apply_cmdline_cfgs(self, user_defined_cfg=None, logger=None):
        if user_defined_cfg:
            for key, value in zip(user_defined_cfg.__dict__['opts'][::2], user_defined_cfg.__dict__['opts'][1::2]):
                if key in self.__dict__:
                    try:
                        setattr(self, key, eval(value))
                    except:
                        setattr(self, key, value)
                else:
                    AttributeError('Configs class does not have attribute: {}'.format(key))
            print('User Configs are applied!')
        
        if not self.Baseline:
            self.Output_Path = './checkpoints/{}-{}-{}-{}{}{}{}{}{}'.format(time.strftime("%m-%d-%Y-%H"),
                                                                                    self.Dataset,
                                                                                    self.Generator_Type,
                                                                                    self.Attack_Classifier,
                                                                                    '_Recon_{}'.format(self.Reconstruction_Loss_Lambda) if self.Reconstruction_Loss else '',
                                                                                    '_ADV_{}'.format(self.Adversarial_Loss_Lambda) if self.Adversarial_Loss else '',
                                                                                    '_IMCLS_{}'.format(self.Image_Classification_Loss_Lambda) if self.Image_Classification_Loss else '',
                                                                                    '_PERCEP_{}'.format(self.Perceptual_Loss_Lambda) if self.Perceptual_Loss else '',
                                                                                    '_ATTDET_{}'.format(self.Attack_Det_Loss_Lambda) if self.Attack_Det_Loss else '')
        else:
            self.Output_Path = './checkpoints/{}-{}-{}-{}-Baseline_{}'.format(time.strftime("%m-%d-%Y-%H"), 
                                                                                self.Dataset,
                                                                                self.Generator_Type, 
                                                                                self.Attack_Classifier,
                                                                                self.Baseline_Source)
    
    def save_cfg2json(self):
        json.dump(self.__dict__, open(osp.join(self.Output_Path, 'configs.json'), 'w'), indent=4)
        