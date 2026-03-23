# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

To build and train a Denoising Autoencoder using PyTorch to remove noise from images. The goal is to reconstruct a clean image from a corrupted (noisy) input image. The MNIST dataset is used as a benchmark to demonstrate this image denoising capability. The dataset used in this code is the MNIST dataset. It's a large database of handwritten digits commonly used for training various image processing systems. Each image in the MNIST dataset is a grayscale image of a handwritten digit (0-9) with a size of 28x28 pixels.


## DESIGN STEPS
### STEP 1: 

Import Libraries and Device Setup: Import necessary PyTorch and utility libraries, then configure the device for computation (CPU/GPU).

### STEP 2: 


Prepare Dataset and Add Noise: Load the MNIST dataset, apply transformations, create DataLoaders, and define a function to add noise to images.


### STEP 3: 

Define Denoising Autoencoder Architecture: Design the Denoising Autoencoder model with convolutional encoder and transpose convolutional decoder layers.



### STEP 4: 

Initialize Model, Loss, and Optimizer: Instantiate the Denoising Autoencoder, set up the Mean Squared Error (MSE) loss, and choose the Adam optimizer.



### STEP 5: 

Implement Training Loop: Create a training function that iterates through epochs, adds noise, performs forward/backward passes, and updates model weights.



### STEP 6: 

Visualize Denoising Performance: Develop a visualization function to display original, noisy, and denoised images from the test set for evaluation.





## PROGRAM

### Name:PRIYADHARSHINI E

### Register Number:212223230159

```
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
     def __init__(self):
      super(DenoisingAutoencoder,self).__init__()
      self.encoder=nn.Sequential(
          nn.Conv2d(1,16,kernel_size=3,stride=2,padding=1),
          nn.ReLU(),
          nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
          nn.ReLU()
      )
      self.decoder=nn.Sequential(
          nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,output_padding=1,padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,output_padding=1,padding=1),
          nn.ReLU()

      )



    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x





# Initialize model, loss function and optimizer
model =DenoisingAutoencoder().to(device)
criterion =nn.MSELoss()
optimizer =optim.Adam(model.parameters(),lr=1e-3)# Initialize model, loss function and optimizer


# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name:PRIYADHARSHINI E")
    print("Reg No:212223230159")
    for epoch in range(epochs):
        running_loss = 0.0
        for images,_ in loader:
          images=images.to(device)
          noisy_images=add_noise(images).to(device)

          #forward pass
          outputs=model(noisy_images)
          loss=criterion(outputs,images)

          #backward pass
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          running_loss+=loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], loss:[{running_loss/len(loader)}]")


# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:PRIYADHARSHINI E")
    print("Register Number:212223230159")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


```

### OUTPUT

### Model Summary
<img width="791" height="527" alt="image" src="https://github.com/user-attachments/assets/36bd25c9-6049-488a-b847-2289fbf427fe" />
### Training loss
<img width="562" height="211" alt="image" src="https://github.com/user-attachments/assets/9685f0aa-99a3-44c3-8f55-27dd0deb58c4" />

## Original vs Noisy Vs Reconstructed Image
<img width="759" height="287" alt="image" src="https://github.com/user-attachments/assets/eb427fa1-c418-4769-a125-3e6e04d10b87" />

## RESULT
Thus the model has been trained for image denoising application.
