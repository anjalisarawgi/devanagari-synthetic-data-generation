import kagglehub

# Download latest version
path = kagglehub.dataset_download("suvooo/hindi-character-recognition")

print("Path to dataset files:", path)