import math

import cv2
import time
import numpy as np
import torch
from torchvision.transforms import transforms

#from models.cnn_tracking5_2 import CNNTracking5
from models.cnn_tracking5_3 import CNNTracking5


def crop_center(image, target_width, target_height):
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the center of the image
    center_x, center_y = original_width // 2, original_height // 2

    # Calculate the cropping box
    crop_x1 = max(center_x - target_width // 2, 0)
    crop_y1 = max(center_y - target_height // 2, 0)
    crop_x2 = min(center_x + target_width // 2, original_width)
    crop_y2 = min(center_y + target_height // 2, original_height)

    # Crop the image
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    cropped_image = cv2.resize(cropped_image, (target_width, target_height))
    return cropped_image


def run_cnn_v5(frame, model, device):
    cropped_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow("Test", cropped_image)

    input = transforms.ToTensor()(cropped_image).to(device)
    input = torch.unsqueeze(input, 0)

    with torch.no_grad():
        out_mask, out_coords = model(input)

    out_mask = out_mask.to('cpu')
    out_coords = out_coords.to('cpu')
    sigmoid_output = torch.sigmoid(out_mask).numpy()
    sigmoid_coords = torch.sigmoid(out_coords).numpy()

    scaled_output = (sigmoid_output * 255).clip(0, 255).astype(np.uint8)
    scaled_coords = (sigmoid_coords * 255).clip(0, 255).astype(np.uint8)  # 255

    segmentation_map = scaled_output.squeeze()
    combined_coords = scaled_coords.squeeze()

    robot_coords = combined_coords[0]
    plant_coords = combined_coords[1]

    return segmentation_map, robot_coords, plant_coords


# Initialize VideoCapture with your video source (0 for webcam)
#cap = cv2.VideoCapture('test_videos/r_test_tracking1.avi')
#cap = cv2.VideoCapture('test_videos/r_test_tracking2.gif')
#cap = cv2.VideoCapture('test_videos/r_test_tracking3.gif')
#cap = cv2.VideoCapture('test_videos/test tracking 5.mkv')
cap = cv2.VideoCapture('test_videos/test tracking 6.mkv')

# Check if a GPU is available and use it, otherwise use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#device = 'cpu'

board_image = cv2.imread('accessories/tabla.png')
board_image = cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE)
board_image = cv2.resize(board_image, (200, 300), interpolation=cv2.INTER_AREA)

model = CNNTracking5().to(device)
checkpoint = torch.load('model_weights/tracking_v5_3 32ep.pth', map_location=torch.device(device))
#checkpoint = torch.load('model_weights/tracking_v5_241 38ep.pth', map_location=torch.device(device))


model.load_state_dict(checkpoint)

# Number of frames to skip before starting
skip_count = 10

frame_count = 0
desired_fps = 9999
frame_skip_count = math.ceil(cap.get(cv2.CAP_PROP_FPS) / desired_fps)
print(f'Frames to skip per detection: {frame_skip_count}')

occlusion_masks = []
location_masks = []
location_coords = []
masks_generated = False

total_plant_areas = 6

empty_counters = [0 for _ in range(total_plant_areas)]
occupancy_values = [0 for _ in range(total_plant_areas)]

for i in range(total_plant_areas):
    occlusion_masks.append(cv2.imread(f'occlusion_masks/occlusion_mask{i}.png', cv2.IMREAD_GRAYSCALE))

while True:
    ret, frame = cap.read()
    frame_backup = frame
    input_res = (200, 150)
    scale_factor = input_res[1] / frame.shape[0]
    new_res = (int(scale_factor * frame.shape[1]), input_res[1])
    offset = (new_res[0] - input_res[0]) // 2
    frame = cv2.resize(frame, new_res)  # interpolation=cv2.INTER_AREA
    frame = crop_center(frame, input_res[0] - 4, input_res[1] - 4)
    frame = cv2.resize(frame, input_res)

    if frame_count < skip_count:
        frame_count += 1
        continue
    elif frame_count == skip_count:
        start_time = time.time()

    if not ret:
        print("Error reading frame.")
        break

    segmentation_map, occupancy_map, plant_map = run_cnn_v5(frame, model, device)

    location_display = np.zeros((plant_map.shape[0], plant_map.shape[1], 3), dtype=np.uint8)

    if not masks_generated:
        for _ in range(total_plant_areas):
            location_masks.append(np.zeros((plant_map.shape[0], plant_map.shape[1]), dtype=np.uint8))
            occlusion_masks.append(np.zeros((plant_map.shape[0], plant_map.shape[1]), dtype=np.uint8))

        plant_area_radius = math.ceil(125 / 3000 * plant_map.shape[0])
        location_coords.append((int(700 / 2000 * plant_map.shape[1]), int(1000 / 3000 * plant_map.shape[0])))
        location_coords.append((int(1300 / 2000 * plant_map.shape[1]), int(1000 / 3000 * plant_map.shape[0])))

        location_coords.append((int(500 / 2000 * plant_map.shape[1]), int(1500 / 3000 * plant_map.shape[0])))
        location_coords.append((int(1500 / 2000 * plant_map.shape[1]), int(1500 / 3000 * plant_map.shape[0])))

        location_coords.append((int(700 / 2000 * plant_map.shape[1]), int(2000 / 3000 * plant_map.shape[0])))
        location_coords.append((int(1300 / 2000 * plant_map.shape[1]), int(2000 / 3000 * plant_map.shape[0])))

        for l in range(total_plant_areas):
            cv2.circle(location_masks[l], location_coords[l], plant_area_radius, (255, 255, 255), -1)

        masks_generated = True

    areas = []
    area_threshold = 0.75
    occlusion_area_threshold = 15.0
    max_empty_counter = 3
    plant_map_int32 = plant_map.astype(np.int32)
    occupancy_map_int32 = occupancy_map.astype(np.int32)

    for l in range(total_plant_areas):
        cv2.circle(location_masks[l], location_coords[l], plant_area_radius, (255, 255, 255), -1)
        location_mask_int32 = location_masks[l].astype(np.int32)
        test = cv2.multiply(plant_map_int32, location_mask_int32)
        print(np.sum(test))
        area = np.sum(cv2.multiply(plant_map_int32, location_mask_int32)) / (255 * 255)
        areas.append(area)

        if area > area_threshold:
            cv2.circle(location_display, location_coords[l], plant_area_radius, (0, 255, 0), -1)
            empty_counters[l] = 0
            occupancy_values[l] = 1
        else:
            occlusion_mask_int32 = occlusion_masks[l].astype(np.int32)
            occlusion_area = np.sum(cv2.multiply(occupancy_map_int32, occlusion_mask_int32)) / (255 * 255)
            if occlusion_area > occlusion_area_threshold and empty_counters[l] < max_empty_counter:
                cv2.circle(location_display, location_coords[l], plant_area_radius, (0, 0, 255), -1)
                occupancy_values[l] = -1
            else:
                occupancy_values[l] = 0
                if empty_counters[l] < max_empty_counter:
                    empty_counters[l] += 1
            #print(occlusion_area)

    print(occupancy_values)

    threshold_segmentation = 100
    threshold_occupancy = 200
    # Apply thresholding to create a binary image
    _, binary_segmentation = cv2.threshold(segmentation_map, threshold_segmentation, 255, cv2.THRESH_BINARY)
    #_, binary_occupancy = cv2.threshold(occupancy_map, threshold_occupancy, 255, cv2.THRESH_BINARY)

    occupancy_map = cv2.resize(occupancy_map, (board_image.shape[1], board_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    plant_map = cv2.resize(plant_map, (board_image.shape[1], board_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    location_display = cv2.resize(location_display, (board_image.shape[1], board_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a 3-channel grayscale image by replicating the single-channel grayscale
    grayscale_3channel_occupancy = cv2.merge([occupancy_map] * 3)
    rgb_plant_map = np.zeros((plant_map.shape[0], plant_map.shape[1], 3), dtype=np.uint8)
    rgb_plant_map[:, :, 1] = plant_map

    # Blend the white image and the BGR image based on grayscale intensities
    board_occupancy_image = cv2.addWeighted(board_image, 1.0, grayscale_3channel_occupancy, 0.8, 0.0)
    board_occupancy_image = cv2.addWeighted(board_occupancy_image, 1.0, rgb_plant_map, 0.8, 0.0)

    elapsed_time = time.time() - start_time
    fps = (frame_count + 1 - skip_count) / elapsed_time

    cv2.putText(frame_backup, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Original Frame', frame_backup)
    cv2.imshow('Frame', frame)
    cv2.imshow('Location Display', location_display)
    cv2.imshow('Segmentation Map', segmentation_map)
    cv2.imshow('Binary Segmentation Map', binary_segmentation)
    cv2.imshow('Occupancy Map', occupancy_map)
    cv2.imshow('Plant Map', plant_map)
    cv2.imshow('Board Occupancy', board_occupancy_image)

    # Skip frames based on frame_skip_count
    for _ in range(frame_skip_count - 1):
        _ = cap.grab()  # Skip frames without decoding

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
