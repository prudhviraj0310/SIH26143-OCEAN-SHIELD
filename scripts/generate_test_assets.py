"""
Generates synthetic realistic test video feeds for offline demonstration & judging.
Produces 4 surveillance video scenarios:
1. test_data/border_breach.mp4: Person crossing border fence line.
2. test_data/crawling_intruder.mp4: Low-light scene with crawling infiltrator.
3. test_data/checkpoint_vehicle.mp4: Checkpoint car with Indian license plate.
4. test_data/wildlife_suppression.mp4: Stray cattle crossing fence (testing false-alarm filter).
"""

import os
import cv2
import numpy as np

def create_border_breach_video(filename="test_data/border_breach.mp4", duration_sec=8, fps=25):
    """Simulates a person walking from north to south, crossing the fence line."""
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    fence_y = int(h * 0.65)

    for i in range(total_frames):
        # Background: outdoor grassy terrain with fence texture
        frame = np.full((h, w, 3), (35, 60, 40), dtype=np.uint8)
        
        # Ground shading
        cv2.rectangle(frame, (0, fence_y), (w, h), (30, 45, 30), -1)
        # Barbed wire fence posts
        for x in range(30, w, 80):
            cv2.line(frame, (x, fence_y - 80), (x, fence_y + 40), (70, 70, 70), 3)
        # Horizontal wire strands
        cv2.line(frame, (0, fence_y - 60), (w, fence_y - 60), (90, 90, 90), 1)
        cv2.line(frame, (0, fence_y - 30), (w, fence_y - 30), (90, 90, 90), 1)
        cv2.line(frame, (0, fence_y), (w, fence_y), (90, 90, 90), 1)

        # Person walking down from y = 180 to y = 420
        progress = i / total_frames
        px = int(w * 0.45 + np.sin(progress * 4) * 25)
        py = int(180 + progress * 240)
        pw, ph = 40, 90

        # Draw person figure
        # Head
        cv2.circle(frame, (px, py - ph + 15), 14, (140, 180, 210), -1)
        # Torso (jacket)
        cv2.rectangle(frame, (px - 16, py - ph + 28), (px + 16, py - 20), (40, 50, 80), -1)
        # Legs
        leg_offset = int(np.sin(progress * 30) * 12)
        cv2.line(frame, (px - 8, py - 20), (px - 8 + leg_offset, py), (30, 30, 40), 5)
        cv2.line(frame, (px + 8, py - 20), (px + 8 - leg_offset, py), (30, 30, 40), 5)

        out.write(frame)

    out.release()
    print(f"Generated: {filename}")

def create_crawling_intruder_video(filename="test_data/crawling_intruder.mp4", duration_sec=8, fps=25):
    """Simulates a pitch-black night scene with an infiltrator crawling horizontally."""
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    ground_y = int(h * 0.75)

    for i in range(total_frames):
        # Low-light dark frame (night vision testing)
        frame = np.full((h, w, 3), (12, 16, 14), dtype=np.uint8)
        # Subtle terrain texture
        cv2.rectangle(frame, (0, ground_y), (w, h), (8, 10, 8), -1)

        progress = i / total_frames
        px = int(80 + progress * 460)
        py = ground_y - 15
        pw, ph = 90, 25  # Wide and low (Aspect ratio ~3.6 -> Crawling!)

        # Draw crawling prone figure
        # Body lying flat
        cv2.ellipse(frame, (px, py), (pw // 2, ph // 2), 0, 0, 360, (45, 55, 50), -1)
        # Head crawling forward
        cv2.circle(frame, (px + pw // 2, py - 4), 10, (60, 75, 70), -1)

        out.write(frame)

    out.release()
    print(f"Generated: {filename}")

def create_checkpoint_vehicle_video(filename="test_data/checkpoint_vehicle.mp4", duration_sec=8, fps=25):
    """Simulates a vehicle approaching border outpost with an Indian license plate."""
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    total_frames = duration_sec * fps

    for i in range(total_frames):
        # Road background
        frame = np.full((h, w, 3), (50, 55, 52), dtype=np.uint8)
        cv2.rectangle(frame, (80, 0), (560, h), (35, 38, 36), -1)
        # Road lane markings
        cv2.line(frame, (320, 0), (320, h), (180, 180, 180), 2, cv2.LINE_AA)

        # Vehicle scaling up as it approaches checkpoint
        progress = i / total_frames
        scale = 0.5 + progress * 0.5
        vw = int(160 * scale)
        vh = int(120 * scale)
        vx = 320 - vw // 2
        vy = int(60 + progress * 240)

        # Car Body
        cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (120, 40, 30), -1)
        # Windshield
        cv2.rectangle(frame, (vx + int(15*scale), vy + int(10*scale)), (vx + vw - int(15*scale), vy + int(45*scale)), (180, 210, 230), -1)
        # Headlights
        cv2.circle(frame, (vx + int(20*scale), vy + vh - int(25*scale)), int(12*scale), (200, 240, 255), -1)
        cv2.circle(frame, (vx + vw - int(20*scale), vy + vh - int(25*scale)), int(12*scale), (200, 240, 255), -1)

        # Indian Number Plate (White HSRP plate with black border)
        pw, ph = int(75 * scale), int(22 * scale)
        px = vx + (vw - pw) // 2
        py = vy + vh - ph - int(8 * scale)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 0), 1)
        
        # Plate Text
        font_scale = 0.35 * scale
        cv2.putText(frame, "DL 01 AB 1234", (px + int(4*scale), py + ph - int(6*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Generated: {filename}")

def create_wildlife_video(filename="test_data/wildlife_suppression.mp4", duration_sec=8, fps=25):
    """Simulates cattle/dog crossing fence to demonstrate false-alarm filtering."""
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    fence_y = int(h * 0.65)

    for i in range(total_frames):
        frame = np.full((h, w, 3), (35, 60, 40), dtype=np.uint8)
        cv2.line(frame, (0, fence_y), (w, fence_y), (90, 90, 90), 2)

        progress = i / total_frames
        cx = int(w * 0.3 + progress * 200)
        cy = int(fence_y - 60 + progress * 120)

        # Draw quadruped shape (cow/dog)
        cv2.ellipse(frame, (cx, cy), (45, 25), 0, 0, 360, (90, 110, 130), -1)
        # Head
        cv2.circle(frame, (cx + 35, cy - 15), 18, (90, 110, 130), -1)
        # Legs (4 legs)
        for offset in [-25, -10, 10, 25]:
            cv2.line(frame, (cx + offset, cy + 15), (cx + offset, cy + 40), (60, 70, 80), 4)

        out.write(frame)

    out.release()
    print(f"Generated: {filename}")

if __name__ == "__main__":
    os.makedirs("test_data", exist_ok=True)
    create_border_breach_video()
    create_crawling_intruder_video()
    create_checkpoint_vehicle_video()
    create_wildlife_video()
    print("[IBVAP] All 4 tactical surveillance scenario videos generated successfully.")
