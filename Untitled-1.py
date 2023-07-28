import os
from PIL import Image
from facenet_pytorch import MTCNN

# Load the MTCNN model
mtcnn = MTCNN(keep_all=True)

# Set the path to the folder containing the images
folder_path = 'F:\Driver Drowsiness Dataset (DDD)\Drowsy'
folder_path_d = 'F:\Driver Drowsiness Dataset (DDD)\Drowsy1'

# Set the desired size of the output images
output_size = (48, 48)

# Loop through all the images in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the input image
        img = Image.open(os.path.join(folder_path, filename))

        # Use MTCNN to detect faces in the image
        boxes, _ = mtcnn.detect(img)

        # Create a list to store the cropped faces
        cropped_faces = []

        # Loop through the detected boxes and crop the faces
        for box in boxes:
            # Convert box coordinates to integers
            box = [int(coord) for coord in box]

            # Crop the face from the image
            face = img.crop(box)

            # Resize the face to the desired size
            face = face.resize(output_size)

            # Add the cropped face to the list
            cropped_faces.append(face)

        # Save the cropped faces
        for i, face in enumerate(cropped_faces):
            face.save(os.path.join(folder_path_d, f'{filename}_{i}.jpg'))