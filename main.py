import cv2
import sys
import numpy as np


# vid_fn = r'videos\video1.mp4'
vid_fn = r'videos\VideoHit.mp4'

cam = cv2.VideoCapture(vid_fn)

NEW_SIZE_FACTOR = 0.5
OBJ_DIM = (15, 15)  # heigh, width
CROSSHAIR_DIM = (15, 15)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

NUM_PARTICLES = 200
PARTICLE_SIGMA = np.min([OBJ_DIM]) // 4  # particle filter shift per generation
DISTRIBUTION_SIGMA = 0.5

def make_crosshairs(img, top_left_xy, bot_right_xy, ch_color, captured):
    obj_h, obj_w = CROSSHAIR_DIM
    center_y, center_x = img.shape[0] // 2, img.shape[1] // 2

    img = cv2.rectangle(img, top_left_xy, bot_right_xy, ch_color, 1)
    img = cv2.line(img, (center_x, img.shape[0] * 1 // 3), (center_x, center_y - obj_h // 2), ch_color, 1)
    img = cv2.line(img, (center_x, center_y + obj_h // 2), (center_x, img.shape[0] * 2 // 3), ch_color, 1)
    img = cv2.line(img, (img.shape[1] * 1 // 3, center_y), (center_x - obj_w // 2, center_y), ch_color, 1)
    img = cv2.line(img, (center_x + obj_w // 2, center_y), (img.shape[1] * 2 // 3, center_y), ch_color, 1)
    return img

def mark_target(img, center_xy, ch_color, captured):
    obj_h, obj_w = OBJ_DIM
    center_x, center_y = int(center_xy[0]), int(center_xy[1])

    tl_x = int(center_xy[0] - OBJ_DIM[1] // 2)
    tl_y = int(center_xy[1] - OBJ_DIM[0] // 2)
    br_x = int(center_xy[0] + OBJ_DIM[1] // 2)
    br_y = int(center_xy[1] + OBJ_DIM[0] // 2)

    img = cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), ch_color, 1)
    img = cv2.line(img, (center_x, 0), (center_x, center_y - obj_h // 2), ch_color, 1)
    img = cv2.line(img, (center_x, center_y - obj_h // 2), (center_x, img.shape[0]), ch_color, 1)
    img = cv2.line(img, (0, center_y), (center_x - obj_w // 2, center_y), ch_color, 1)
    img = cv2.line(img, (center_x + obj_w // 2, center_y), (img.shape[1], center_y), ch_color, 1)

    return img


# main code
captured = False
img_patch = np.zeros(OBJ_DIM)  # this is actually changed to a 3d image later

particles_xy, particles_scores, particles_patches = [], [], []

ret, img = cam.read()  # read first frame for init
if img is None:
    cam.release()
    sys.exit(0)

img_color = cv2.resize(img, (int(img.shape[1] * NEW_SIZE_FACTOR), int(img.shape[0] * NEW_SIZE_FACTOR)))
img_h, img_w, _ = img_color.shape
top_left_x = img_w // 2 - OBJ_DIM[1] // 2
top_left_y = img_h // 2 - OBJ_DIM[0] // 2
bot_right_x = img_w // 2 + OBJ_DIM[1] // 2
bot_right_y = img_h // 2 + OBJ_DIM[0] // 2

while True:
    _, img = cam.read()
    if img is None:
        cam.release()
        sys.exit(0)

    img_color = cv2.resize(img, (int(img.shape[1] * NEW_SIZE_FACTOR), int(img.shape[0] * NEW_SIZE_FACTOR)))
    img_color_clean = img_color.copy()
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    if captured:
        key = cv2.waitKey(10) & 0xFF
    else:
        key = cv2.waitKey(25) & 0xFF
    if key == 27:  # ESC
        break

    if (key == ord('c')) or (key == ord('c')):
        captured = True
        img_patch = img_color_clean[top_left_y:bot_right_y, top_left_x:bot_right_x]
        particles_xy = np.zeros((NUM_PARTICLES, 2))
        particles_xy[:, :] = [img_w // 2, img_h // 2]

    elif (key == ord('d')) or (key == ord('D')):
        captured = False
        img_patch = np.zeros(img_patch.shape)

    if captured:
        # some of these loops can be converted to numpy style broadcasting for faster execution
        # introduce noise for the next generation
        for i, p in enumerate(particles_xy):
            if i == 0:  # skip the first particle so it remains at the center
                continue
            p[0] += np.random.normal(0, PARTICLE_SIGMA)
            p[1] += np.random.normla(0, PARTICLE_SIGMA)

            # adjust for out of frame particles
            p[0] = OBJ_DIM[1] // 2 if p[0] < OBJ_DIM[1] // 2 else p[0]
            p[0] = img_w - OBJ_DIM[1] // 2 if p[0] > img_w - OBJ_DIM[1] // 2 else p[0]
            p[1] = OBJ_DIM[0] // 2 if p[1] < OBJ_DIM[0] // 2 else p[1]
            p[1] = img_h - OBJ_DIM[0] // 2 if p[1] > img_h - OBJ_DIM[0] // 2 else p[1]

        # display particles
        for p in particles_xy:
            img_color = cv2.circle(img_color, (int(p[0]), int(p[1])), 1, GREEN, -1)

        # get patches for esch particle
        particles_patches = []
        for p in particles_xy:
            patch_top_left_x = int(p[0] - OBJ_DIM[1] // 2)
            patch_top_left_y = int(p[1] - OBJ_DIM[0] // 2)
            patch_bot_right_x = int(p[0] + OBJ_DIM[1] // 2)
            patch_bot_right_y = int(p[1] + OBJ_DIM[0] // 2)
            temp_patch = img_color_clean[patch_bot_right_y:patch_bot_right_y, patch_top_left_x:patch_bot_right_x]

        # compare each patch with the model patch
        model_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
        # model_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)[:,:,0]
        model_patch = cv2.GaussianBlur(model_patch, (3, 3), 0)
        particles_scores = []
        for p in particles_patches:
            temp_patch = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
            # temp_patch = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)[:.:.0]
            temp_patch = cv2.GaussianBlur(temp_patch, (3, 3), 0)
            mse = np.mean((model_patch - temp_patch) ** 2)
            particles_scores.append(mse)

        # convert to a 'probabillity'
        particles_scores = np.array(particles_scores)
        # missing np.sqrt() is intentional
        particles_scores = 1.0 / (2.0 * np.pi * DISTRIBUTION_SIGMA) * np.exp(-particles_scores / (2.0 * DISTRIBUTION_SIGMA**2))
        particles_scores = particles_scores / np.sum(particles_scores)

        # resample
        new_pxy_idx = np.random.choice(range(NUM_PARTICLES), size=NUM_PARTICLES - 1, p=particles_scores,replace=True)
        best_idx = np.where(particles_scores == np.max(particles_scores))[0][0]
        best_xy = particles_xy[best_idx]
        new_set = particles_xy[new_pxy_idx]
        particles_xy = np.vstack((best_xy, new_set))

        # display best_xy/ mark target
        img_color = mark_target(img_color, best_xy, RED, 1)

        # update model patch
        img_patch = particles_patches[best_idx]

    img_color = make_crosshairs(img_color, (top_left_x, top_left_y), (bot_right_x, bot_right_y), GREEN, 1)
    cv2.imshow('Object Tracker', img_color)

# housekeeping
cam.release()
cv2.destroyAllWindow()