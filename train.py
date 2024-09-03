
from ignite.handlers import Checkpoint, DiskSaver
import torch.amp
from model import get_gen_disc
from data import train_dl
import torch
from config import Config
from ignite.engine import Engine ,Events
from utils import attach_ignite

gen,disc,bce_loss,l1_loss,gen_optim,disc_optim =get_gen_disc()



def train_step(engine,batch):
    input_images , real_output_images = batch

    # Train the disc :
    gen.eval()
    disc.zero_grad()

    fake_imgs = gen(input_images)
    
    D_real = disc(input_images,real_output_images)
    D_fake = disc(input_images,fake_imgs.detach())

    d_loss_real = bce_loss(
        D_real , torch.ones_like(D_real)
    )

    d_loss_fake = bce_loss(
        D_fake , torch.zeros_like(D_fake,dtype=torch.float)
    )

    d_loss = (d_loss_fake+d_loss_real)/2
    d_loss.backward()
    disc_optim.step()

    # train the gen
    disc.eval()
    gen.zero_grad()

    D_fake = disc(input_images,fake_imgs)

    adv_loss_g = bce_loss(
        D_fake , torch.ones_like(D_fake,dtype=torch.float) 
    )
    l1_loss_g = l1_loss(
        fake_imgs,real_output_images
    )

    g_loss = adv_loss_g + Config.L1_LAMBDA*l1_loss_g

    g_loss.backward()

    gen_optim.step()

    return {
        'generator_loss':g_loss.item(),
        'discriminator_loss':d_loss.item(),
    }


trainer = Engine(train_step)


checkpoint_handler = Checkpoint(
        to_save={'gen': gen,
                  'disc': disc,
                  'engine':trainer,
                  'gen_optim':gen_optim,
                  'disc_optim':disc_optim},
        save_handler=DiskSaver('pix2pix_models', create_dir=True,require_empty=False),  
        n_saved=2,  # Number of checkpoints to keep
        filename_prefix='model',  # Prefix for the checkpoint filenames
        global_step_transform=lambda engine, event: engine.state.epoch  
    )

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)




attach_ignite(trainer,gen)

if __name__=="__main__":
    trainer.run(train_dl,max_epochs=100)



