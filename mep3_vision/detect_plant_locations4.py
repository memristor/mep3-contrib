import math

import cv2
import time
import numpy as np
import torch
from torchvision.transforms import transforms

from models.cnn_tracking7_31 import CNNTracking7


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


def draw_text(image, text, coords):
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Font scale (multiplied by the font size)
    font_color = (255, 255, 255)  # White color in BGR format
    font_thickness = 1  # Thickness of the text

    # Get the size of the text box to position it in the center
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size[0]

    # Calculate the position to center the text
    x = int(coords[0] * image.shape[1] - text_width // 2)
    y = int(coords[1] * image.shape[0] + text_height // 2)

    # Draw the text on the image
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), 3)
    cv2.putText(image, text, (x, y), font, font_scale, font_color, font_thickness)


def draw_pillars(image, probabilities, coords):
    # Define pillar settings
    pillar_color = (255, 255, 255)  # White color in BGR format
    pillar_thickness = 1  # Thickness of the pillar
    max_pillar_height = 10  # Maximum height of the pillar (adjust as needed)
    pillar_width = 2  # Width of each pillar

    # Calculate the total width of the pillars
    total_width = len(probabilities) * (pillar_width + 2)  # Include spacing between pillars

    # Calculate the starting x-coordinate to center the pillars around coords
    start_x = int(coords[0] * image.shape[1]) - total_width // 2

    # Iterate over the probabilities and draw pillars
    for i, prob in enumerate(probabilities):
        # Calculate pillar height based on probability
        pillar_height = int(prob * max_pillar_height)

        # Calculate pillar position
        x1 = start_x + i * (pillar_width + 2)  # Include spacing between pillars
        y1 = int(coords[1] * image.shape[0])
        x2 = x1 + pillar_width
        y2 = y1 - pillar_height  # Draw upwards

        # Draw the pillar
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.rectangle(image, (x1, y1), (x2, y2), pillar_color, -1)


def get_area_occupancy(new_area_counts, running_area_counts, occupancy_map_int32):
    threshold_probability = 0.5
    alpha = 0.7
    threshold_count = 3

    area_counts = []
    new_running_area_counts = []
    for l in range(total_plant_areas):
        occlusion_mask_int32 = occlusion_masks[l].astype(np.int32)
        occlusion_area = 0
        #occlusion_area = np.sum(cv2.multiply(occupancy_map_int32, occlusion_mask_int32)) / (255 * 255)
        if occlusion_area > occlusion_area_threshold:
            new_running_area_counts.append(running_area_counts[l])
        else:
            new_running_area_counts.append(running_area_counts[l] * alpha + new_area_counts[l] * (1 - alpha))

    area_counts = [p1 * alpha + p2 * (1 - alpha) for p1, p2 in zip(running_area_counts, new_area_counts)]

    cumulative_probabilities = [0.0 for _ in range(6)]
    plant_area_occupancy = []
    for i in range(6):
        area_count = area_counts[i]
        for j in range(threshold_count):
            cumulative_probabilities[i] += area_count[j]

    print(cumulative_probabilities)

    plant_area_occupancy = []
    for i in range(6):
        if 1.0 - cumulative_probabilities[i] > threshold_probability:
            plant_area_occupancy.append(1)
        else:
            plant_area_occupancy.append(0)

    return plant_area_occupancy, area_counts, new_running_area_counts


def run_cnn_v6(frame, model, device):
    cropped_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #cv2.imshow("Test", cropped_image)

    #cropped_image = cropped_image.astype(np.float32)

    input = transforms.ToTensor()(cropped_image).to(device)
    input = torch.unsqueeze(input, 0)

    with torch.no_grad():
        out_mask, out_coords, out_counts, probe1 = model(input)

    out_mask = out_mask.to('cpu')
    out_coords = out_coords.to('cpu')
    out_counts = out_counts.to('cpu').squeeze(0)
    probe1 = probe1.to('cpu').squeeze(0)
    sigmoid_output = torch.sigmoid(out_mask).numpy()
    sigmoid_coords = torch.sigmoid(out_coords).numpy()
    sigmoid_probe1_full = torch.sigmoid(probe1)

    area_counts = []
    for i in range(6):
        area_count = out_counts[i*7:(i+1)*7]
        area_count = torch.softmax(area_count, dim=0).numpy()
        area_counts.append(area_count)

    scaled_output = (sigmoid_output * 255).clip(0, 255).astype(np.uint8)
    scaled_coords = (sigmoid_coords * 255).clip(0, 255).astype(np.uint8)  # 255

    for channel in range(10):
        sigmoid_probe1 = sigmoid_probe1_full[channel:channel + 1]
        sigmoid_probe1 = sigmoid_probe1.permute(1, 2, 0).numpy()
        scaled_probe1 = (sigmoid_probe1 * 255).clip(0, 255).astype(np.uint8)
        scaled_probe1 = cv2.resize(scaled_probe1, (scaled_probe1.shape[1] * 4, scaled_probe1.shape[0] * 4), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(f"Probe{channel}", scaled_probe1)

    segmentation_map = scaled_output.squeeze()
    combined_coords = scaled_coords.squeeze()

    robot_coords = combined_coords[0]
    plant_coords = combined_coords[1]

    return segmentation_map, robot_coords, plant_coords, area_counts


# Initialize VideoCapture with your video source (0 for webcam)
#cap = cv2.VideoCapture('r_test_tracking1.avi')
#cap = cv2.VideoCapture('r_test_tracking2.gif')
#cap = cv2.VideoCapture('r_test_tracking3.gif')
#cap = cv2.VideoCapture('test tracking 5.mkv')
#cap = cv2.VideoCapture('test tracking 6.mkv')
#cap = cv2.VideoCapture('test tracking 7.mkv')
cap = cv2.VideoCapture('test_videos/test tracking 8.mkv')
#cap = cv2.VideoCapture('test_video_real1.avi')

# Check if a GPU is available and use it, otherwise use the CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
#device = 'cpu'

board_image = cv2.imread('accessories/tabla.png')
board_image = cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE)
board_image = cv2.resize(board_image, (200, 300), interpolation=cv2.INTER_AREA)

model = CNNTracking7().to(device)
model.eval()
#checkpoint = torch.load('tracking_v5_3 31ep.pth', map_location=torch.device(device))
#checkpoint = torch.load('tracking_v5_241 38ep.pth', map_location=torch.device(device))
checkpoint = torch.load('model_weights/tracking_v7_31 26ep.pth', map_location=torch.device(device))

model.load_state_dict(checkpoint)

# Number of frames to skip before starting background update
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

for i in range(total_plant_areas):
    occlusion_masks.append(cv2.imread(f'occlusion_masks/occlusion_mask{i}.png', cv2.IMREAD_GRAYSCALE))

running_area_counts = [np.ones(7) / 7 for _ in range(6)]

unadjusted_coords = []
unadjusted_coords.append((700 / 2000, 1000 / 3000))
unadjusted_coords.append((1300 / 2000, 1000 / 3000))

unadjusted_coords.append((500 / 2000, 1500 / 3000))
unadjusted_coords.append((1500 / 2000, 1500 / 3000))

unadjusted_coords.append((700 / 2000, 2000 / 3000))
unadjusted_coords.append((1300 / 2000, 2000 / 3000))

occlusion_area_threshold = 15.0

unadjusted_plant_area_radius = 125 / 3000

photo_mode = False

while True:
    if not photo_mode:
        ret, frame = cap.read()
    else:
        frame = cv2.imread('test_images/img4.jpg')
    frame_backup = frame
    input_res = (150, 200)
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

    #if not ret:
    #    print("Error reading frame.")
    #    break

    segmentation_map, occupancy_map, plant_map, new_area_counts = run_cnn_v6(frame, model, device)

    location_display = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1], 3), dtype=np.uint8)

    if not masks_generated:
        for _ in range(total_plant_areas):
            occlusion_masks.append(np.zeros((occupancy_map.shape[0], occupancy_map.shape[1]), dtype=np.uint8))

        plant_area_radius = math.ceil(unadjusted_plant_area_radius * occupancy_map.shape[0])
        location_coords.append((int(unadjusted_coords[0][0] * occupancy_map.shape[1]), int(unadjusted_coords[0][1] * occupancy_map.shape[0])))
        location_coords.append((int(unadjusted_coords[1][0] * occupancy_map.shape[1]), int(unadjusted_coords[1][1] * occupancy_map.shape[0])))

        location_coords.append((int(unadjusted_coords[2][0] * occupancy_map.shape[1]), int(unadjusted_coords[2][1] * occupancy_map.shape[0])))
        location_coords.append((int(unadjusted_coords[3][0] * occupancy_map.shape[1]), int(unadjusted_coords[3][1] * occupancy_map.shape[0])))

        location_coords.append((int(unadjusted_coords[4][0] * occupancy_map.shape[1]), int(unadjusted_coords[4][1] * occupancy_map.shape[0])))
        location_coords.append((int(unadjusted_coords[5][0] * occupancy_map.shape[1]), int(unadjusted_coords[5][1] * occupancy_map.shape[0])))

        masks_generated = True

    areas = []
    max_empty_counter = 3
    occupancy_map_int32 = occupancy_map.astype(np.int32)

    plant_area_occupancy, area_counts, running_area_counts = get_area_occupancy(new_area_counts, running_area_counts, occupancy_map_int32)

    threshold_segmentation = 100
    threshold_occupancy = 200
    # Apply thresholding to create a binary image
    _, binary_segmentation = cv2.threshold(segmentation_map, threshold_segmentation, 255, cv2.THRESH_BINARY)
    #_, binary_occupancy = cv2.threshold(occupancy_map, threshold_occupancy, 255, cv2.THRESH_BINARY)

    occupancy_map = cv2.resize(occupancy_map, (board_image.shape[1], board_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    plant_map = cv2.resize(plant_map, (board_image.shape[1], board_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    location_display = cv2.resize(location_display, (board_image.shape[1], board_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    circle_radius = int(unadjusted_plant_area_radius * board_image.shape[0])

    for l in range(total_plant_areas):
        circle_x, circle_y = int(unadjusted_coords[l][0] * board_image.shape[1]), int(unadjusted_coords[l][1] * board_image.shape[0])
        if plant_area_occupancy[l] == 1:
            cv2.circle(location_display, (circle_x, circle_y), circle_radius, (0, 255, 0), -1)
            empty_counters[l] = 0
        else:
            occlusion_mask_int32 = occlusion_masks[l].astype(np.int32)
            occlusion_area = 0
            #occlusion_area = np.sum(cv2.multiply(occupancy_map_int32, occlusion_mask_int32)) / (255 * 255)
            if occlusion_area > occlusion_area_threshold and empty_counters[l] < max_empty_counter:
                cv2.circle(location_display, (circle_x, circle_y), circle_radius, (0, 0, 255), -1)
            elif empty_counters[l] < max_empty_counter:
                empty_counters[l] += 1
            #print(occlusion_area)

    for l in range(total_plant_areas):
        draw_pillars(location_display, area_counts[l], unadjusted_coords[l])
        #max_val = np.argmax(area_counts[l])
        #draw_text(location_display, str(max_val), unadjusted_coords[l])

    # Create a 3-channel grayscale image by replicating the single-channel grayscale
    grayscale_3channel_occupancy = cv2.merge([occupancy_map] * 3)

    # Blend the white image and the BGR image based on grayscale intensities
    board_occupancy_image = cv2.addWeighted(board_image, 1.0, grayscale_3channel_occupancy, 0.8, 0.0)

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
    if not photo_mode:
        for _ in range(frame_skip_count - 1):
            _ = cap.grab()  # Skip frames without decoding

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
