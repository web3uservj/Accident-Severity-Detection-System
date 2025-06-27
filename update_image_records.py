from pymongo import MongoClient
import gridfs
import os
from bson.objectid import ObjectId
import glob

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Update with your MongoDB connection string
db = client['accident_detection']  # Your database name
fs = gridfs.GridFS(db)

# Get all detections without image_id
detections = db.detections.find({'image_id': {'$exists': False}})

# Path to your old images directory
OLD_IMAGES_DIR = 'uploads'  # Update with your actual path

print(f"Checking for detections without image_id...")
count = 0

for detection in detections:
    detection_id = str(detection['_id'])

    # Look for image files with detection ID in the name
    possible_files = glob.glob(f"{OLD_IMAGES_DIR}/*{detection_id}*.*")

    if possible_files:
        image_path = possible_files[0]
        print(f"Found image for detection {detection_id}: {image_path}")

        # Read the image file
        with open(image_path, 'rb') as f:
            image_data = f.read()

            # Determine content type based on file extension
            ext = os.path.splitext(image_path)[1].lower()
            content_type = 'image/jpeg'  # Default
            if ext == '.png':
                content_type = 'image/png'
            elif ext == '.gif':
                content_type = 'image/gif'

            # Store in GridFS
            image_id = fs.put(
                image_data,
                filename=os.path.basename(image_path),
                content_type=content_type
            )

            # Update the detection record
            db.detections.update_one(
                {'_id': detection['_id']},
                {'$set': {'image_id': image_id}}
            )

            print(f"Updated detection {detection_id} with image_id {image_id}")
            count += 1
    else:
        print(f"No image found for detection {detection_id}")

print(f"Updated {count} detection records with image_id")

# Check for any detections that still don't have an image_id
remaining = db.detections.count_documents({'image_id': {'$exists': False}})
print(f"{remaining} detections still without image_id")

if remaining > 0:
    print("You may need to manually update these records or check for images in a different location.")