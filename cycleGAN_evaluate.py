import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from cycleGAN_training import Generator

# Define the device and initialize your generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_AB = Generator().to(device)

# Load the saved weights (change the filename as needed)
G_AB.load_state_dict(torch.load("checkpoints/G_AB_epoch_100.pth", map_location=device))
G_AB.eval()  # Set the model to evaluation mode

# Define the same transform used during training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load a test image (e.g., a typed image that you want to convert to handwritten style)
test_image = Image.open("path_to_your_test_image.png").convert("RGB")
test_tensor = transform(test_image).unsqueeze(0).to(device)

# Run inference without tracking gradients
with torch.no_grad():
    output_tensor = G_AB(test_tensor)

# To visualize, convert the output tensor back to a PIL image
output_tensor = output_tensor.squeeze(0).cpu().detach()
# Denormalize the image from [-1,1] to [0,1]
output_image = (output_tensor * 0.5 + 0.5).clamp(0,1)
output_image = transforms.ToPILImage()(output_image)

# Display the result
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(test_image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Output (Handwritten Style)")
plt.imshow(output_image)
plt.axis("off")
plt.show()