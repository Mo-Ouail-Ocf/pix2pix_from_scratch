import torch
import torch.nn as nn
from config import Config
# Discriminator
class DiscCNNBlock(nn.Module):

    def __init__(self,in_channels,out_channels,stride=2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,stride,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.conv(x)
    
# use of Patch Gan discriminator
class Discriminator(nn.Module):
    def __init__(self,in_channels=3,feature_maps=[64,128,256,512]):
        super().__init__()


        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels*2,feature_maps[0],kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        self.blocks = nn.Sequential()
        in_channels = feature_maps[0]
        for feature_map in feature_maps[1:]:
            self.blocks.append(
                DiscCNNBlock(in_channels,feature_map,stride=1 if feature_maps[-1]==feature_map else 2)
            )
            in_channels=feature_map

        self.blocks.append(
            nn.Conv2d(
                in_channels,1,4,1,1,padding_mode='reflect'
            )    
        )
        

    def forward(self,x,y):
        # x : before image , y : label image (fake or real)
        input_img = torch.concat([x,y],dim=1)
        out = self.initial_block(input_img)
        out = self.blocks(out)
        return out

# Generator
# Encoder : use of Leaky ReLU 
# Decoder : use of ReLU

class GenConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,
                 down:bool,activation:str,drop:bool=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,2,1,bias=False) 
            if down 
            else nn.ConvTranspose2d(in_channels,out_channels,4,2,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if activation=="relu"
            else nn.LeakyReLU(0.2)
        )

        if drop:
            self.conv.append(
                nn.Dropout2d(0.5)
            )
    def forward(self,x):
        return self.conv(x)

class Generator(nn.Module):

    def __init__(self,in_channels=3,features=64):
        super().__init__()
        # No  use of batch norm for initial downsampling layer
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels,features,4,2,1,padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        ) #(256->128)

        self.encoder = self._make_encoder(features) #(256->128->64->32->16->8->4->2)
                                                    #(3->64->128->256->512->512->512->512)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8,4,2,1,padding_mode='reflect'),
            nn.ReLU() 
        ) #(256->128->2->1)

        # The decoder 
        self.up1 = GenConvBlock(features*8,features*8,False,'relu',True) # (2,512)
        self.up2 = GenConvBlock(features*8*2,features*8,False,'relu',True) # (4,512)
        self.up3 = GenConvBlock(features*8*2,features*8,False,'relu',True) # (8,512)
        self.up4 = GenConvBlock(features*8*2,features*8,False,'relu',True) # (16,512)
        self.up5 = GenConvBlock(features*8*2,features*4,False,'relu',True) # (32,256)
        self.up6 = GenConvBlock(features*4*2,features*2,False,'relu',True) # (64,128)
        self.up7 = GenConvBlock(features*2*2,features,False,'relu',True) # (128,64)

        self.decoder = nn.ModuleList([self.up2,self.up3,self.up4,self.up5,self.up6,self.up7])
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2,in_channels,4,2,1),
            nn.Tanh()
        ) # (256,3)
    
    def forward(self,x):
        # Encoder
        activations =[]
        x = self.initial_down(x)
        activations.append(x)
        for layer in self.encoder:
            x = layer(x)
            activations.append(x)
        # there will be 
        # Bottleneck
        x = self.bottleneck(x)
        x = self.up1(x)
        # Decoder
        level =0
        for encoder_output in activations[::-1]:
            concat = torch.cat([x,encoder_output],dim=1)
            if level < len(self.decoder):
                x = self.decoder[level](concat)
                level+=1
            else: # pass to the final upsampling output layer
                x=concat
        x = self.final_up(x)
        return x 
        


    
    def _make_encoder(self,features)->nn.ModuleList:
        # 6 gen conv blocks
        encoder = nn.ModuleList()
        in_channels = features
        for _ in range(6):
            #there will be blocks in the encoder
            if in_channels!=features*8:
                out_channels =in_channels*2 

            encoder.append(
                GenConvBlock(
                    in_channels= in_channels,
                    activation='leaky',
                    down=True,
                    out_channels=out_channels,
                    drop=False
                )
            )
            in_channels = out_channels

        return encoder 

def get_gen_disc(device='cuda'):
    gen = Generator().to(device)
    disc = Discriminator().to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    gen_optim = torch.optim.Adam(
        gen.parameters(),
        lr=Config.LEARNING_RATE,
        betas=Config.BETAS
    )

    disc_optim = torch.optim.Adam(
        disc.parameters(),
        lr=Config.LEARNING_RATE,
        betas=Config.BETAS
    )

    return gen,disc,bce_loss,l1_loss,gen_optim,disc_optim


if __name__=="__main__":
    disc =Discriminator().to('cuda') 
    gen =Generator().to('cuda')
    x , y = torch.zeros((1,3,256,256),device='cuda'),torch.zeros((1,3,256,256),device='cuda')
    out = gen(x)
    print(out.shape) 