from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

from ignite.engine import Engine,Events

from torchvision.utils import make_grid 

from ignite.contrib.handlers import tqdm_logger

from data import valid_dl

import torch
def attach_ignite(
        trainer:Engine,
        gen
):
    
    tb_logger = TensorboardLogger(log_dir ='./pix2pix_log')

    tqdm_train = tqdm_logger.ProgressBar().attach(trainer,output_transform=lambda x:x)

    
    tb_logger.attach_output_handler(
        engine=trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag='train',
        output_transform=lambda x: {
            'g_loss':x['generator_loss'],
            'd_loss':x['discriminator_loss']
        }
    )

    def log_generated_images(engine, logger, gen, epoch):
        gen.eval()
        with torch.no_grad():
            batch = next(iter(valid_dl))
            input_images, real_output_images = batch
            
            # Generate fake images
            fake_imgs = gen(input_images)

            # Prepare the images to be logged
            input_grid = make_grid(input_images, normalize=True, value_range=(-1, 1))
            fake_grid = make_grid(fake_imgs, normalize=True, value_range=(-1, 1))
            real_grid = make_grid(real_output_images, normalize=True, value_range=(-1, 1))

            # Log the images
            logger.writer.add_image('input_images', input_grid, epoch)
            logger.writer.add_image('fake_images', fake_grid, epoch)
            logger.writer.add_image('real_images', real_grid, epoch)
        
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_images(engine):
        epoch = engine.state.epoch
        log_generated_images(engine, tb_logger, gen, epoch)






    
    
