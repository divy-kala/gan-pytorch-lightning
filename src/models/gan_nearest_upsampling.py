import torch
from torch import Tensor
from pytorch_lightning import LightningModule
import torch.nn as nn
import torchvision

def down(in_channels, out_channels,  size=3, apply_bn=True):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,  size, stride=1, padding='same', bias=False),
        nn.Conv2d(out_channels, out_channels,  size, stride=2, padding=1, bias=False),
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    if apply_bn:
        model.add_module('batch_norm', nn.BatchNorm2d(out_channels))
    model.add_module('leaky_relu', nn.LeakyReLU())
    
    return model


def upsample(in_channels, out_channels, size=3, apply_drop=False):
    model = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.Conv2d(in_channels, out_channels,  size, stride=1, padding='same', bias=False),
        nn.Conv2d(out_channels, out_channels,  size, stride=1, padding='same', bias=False),
        nn.BatchNorm2d(out_channels),
    )
    
    if apply_drop:
        model.add_module('dropout', nn.Dropout(0.5))
    
    model.add_module('relu', nn.ReLU())
    
    return model



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define the encoder (down_stack)
        self.down_stack = nn.ModuleList([
            down(1, 32, apply_bn=False),
            down(32, 64),
            down(64, 128),
            down(128, 256),
            down(256, 512),
            down(512, 512),
        ])

        # Define the decoder (up_stack)
        self.up_stack = nn.ModuleList([
            upsample(512, 512, apply_drop=True),
            upsample(1024, 256, apply_drop=True),
            upsample(512, 128, apply_drop=True),
            upsample(256, 64),
            upsample(128, 32),
        ])

        # Final Convolutional Layer
        self.final_upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.final_act = nn.Sigmoid()


    def forward(self, x):
        skips = []
        for down_ in self.down_stack:
            x = down_(x)
            skips.append(x)

        skips = list(reversed(skips[:-1]))

        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat((x, skip), dim=1)

        x = self.final_conv(self.final_upsample(x))
        x = self.final_act(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Down-sampling layers
        self.down_stack = nn.ModuleList([
            down(4, 32, apply_bn=False),
            down(32, 64),
            down(64, 128),
        ])

        # Final Convolutional Layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding='same'),
        )

    def forward(self, inputs, targets):
        # Concatenate inputs and targets
        x = torch.cat((inputs, targets), dim=1)  # Concatenate along the channel dimension

        # Down-sampling layers
        for down_layer in self.down_stack:
            x = down_layer(x)

        # Final Convolutional Layers
        x = self.final_conv(x)

        return x


class GANNearest(LightningModule):
    def __init__(self):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()

        self.loss_fn = nn.BCEWithLogitsLoss()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):

        g_opt, d_opt = self.optimizers()
        images, labels = batch
        
        # Optimize Generator 
        self.toggle_optimizer(g_opt)
        generated_images = self.G(labels)
        
        # # Log sampled images
        # sample_imgs = self.generated_imgs[:6]
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image("generated_images", grid, 0)

        disc_generated_output = self.D(labels, generated_images)
        gen_loss = self.loss_fn(torch.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = torch.mean(torch.abs(images - generated_images))
        total_gen_loss = gen_loss + l1_loss * 10

        self.manual_backward(total_gen_loss)
        g_opt.step()
        g_opt.zero_grad()
        
        self.untoggle_optimizer(g_opt)
        
        # Optimize Discriminator 
        self.toggle_optimizer(d_opt)
        disc_real_output = self.D(labels,images)
        generated_images = self.G(labels).detach()
        disc_generated_output = self.D(labels,generated_images)
        
        real_loss = self.loss_fn (torch.ones_like(disc_real_output), disc_real_output)
        fake_loss = self.loss_fn (torch.zeros_like(disc_generated_output), disc_generated_output)
        disc_loss = real_loss + fake_loss

        
        self.manual_backward(disc_loss)
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)

        
        self.log_dict({"total_g_loss": total_gen_loss, "gen_l1": l1_loss, "gen_adversarial": gen_loss, "d_loss": disc_loss}, prog_bar=True)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-3)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=1e-4)
        return g_opt, d_opt
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        gen_images = self.G(labels)

        epoch = self.current_epoch

        grid_gen_images = torchvision.utils.make_grid(gen_images)
        grid_images = torchvision.utils.make_grid(images)   
        grid_labels = torchvision.utils.make_grid(labels*36) 

        self.logger.experiment.add_image("generated_images", grid_gen_images, self.current_epoch)
        self.logger.experiment.add_image("real_images", grid_images, self.current_epoch)
        self.logger.experiment.add_image("true_labels", grid_labels, self.current_epoch)


if __name__ == '__main__':
    
    # Test Generator
    G = Generator()
    x = torch.rand(16,1,64,64)
    print(G(x).shape)

    # Test Discriminator
    discriminator = Discriminator()
    input_data = torch.randn(16, 3, 64, 64) 
    output_data = discriminator(input_data, input_data)   # 16,6,64,64 input
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
    
    # Test GAN
    model = GANNearest()
    

    
