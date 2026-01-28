import cv2
import numpy as np
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import subprocess
import tempfile


class CinemaASCII:
    def __init__(self):
        self.ascii_chars = np.array(list(" .'`^\",:;Il!i><~+_-?][}{1)(|\\//tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"))
        self.auto_palette = None
        self.prev_color_frame = None
        self.gamma = 1.3

    # -------- Palette Extraction --------
    def extract_palette(self, cap, k):
        samples = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0, total, 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, f = cap.read()
            if ret:
                small = cv2.resize(f, (64, 64))
                samples.append(small.reshape(-1, 3))

        samples = np.vstack(samples).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
        _, _, centers = cv2.kmeans(samples, k, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return np.array(centers, dtype=np.float32)

    # -------- Perceptual Color Distance --------
    def color_match(self, pixels, palette):
        diff = pixels[:, None, :] - palette[None, :, :]
        dist = (2 + diff[:, :, 0]/255) * diff[:, :, 0]**2 + \
               4 * diff[:, :, 1]**2 + \
               (2 + (255 - diff[:, :, 0])/255) * diff[:, :, 2]**2
        return palette[np.argmin(dist, axis=1)]

    # -------- Frame to ASCII --------
    def frame_to_ascii(self, frame, w, h, fs):
        frame = cv2.resize(frame, (w, h))

        # Spatial smoothing
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        rgb = frame[:, :, ::-1].astype(np.float32)

        # Temporal smoothing
        if self.prev_color_frame is not None:
            rgb = rgb * 0.7 + self.prev_color_frame * 0.3
        self.prev_color_frame = rgb.copy()

        # Brightness with gamma
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        gray = np.power(gray, self.gamma)

        # Palette mapping
        flat = rgb.reshape(-1, 3)
        mapped = self.color_match(flat, self.auto_palette).reshape(h, w, 3)

        char_w, char_h = fs // 2, fs
        img = Image.new("RGB", (w * char_w, h * char_h), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", fs)
        except:
            font = ImageFont.load_default()

        for y in range(h):
            for x in range(w):
                c = self.ascii_chars[int(gray[y, x] * (len(self.ascii_chars)-1))]
                color = tuple(mapped[y, x].astype(int))
                draw.text((x*char_w, y*char_h), c, fill=color, font=font)

        return np.array(img)

    # -------- Video Conversion --------
    def convert(self, inp, out, w, h, fs):
        cap = cv2.VideoCapture(inp, cv2.CAP_FFMPEG)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"{total} frames at {fps:.2f} FPS")

        self.auto_palette = self.extract_palette(cap, 24)

        with tempfile.TemporaryDirectory() as tmp:
            for i in range(total):
                ret, f = cap.read()
                if not ret:
                    break

                if i % 10 == 0:
                    print(f"\rFrame {i}/{total}", end="")

                af = self.frame_to_ascii(f, w, h, fs)
                cv2.imwrite(os.path.join(tmp, f"f_{i:06d}.png"),
                            cv2.cvtColor(af, cv2.COLOR_RGB2BGR))

            print("\nEncoding with audio")

            audio = os.path.join(tmp, "audio.aac")
            subprocess.run(
                ["ffmpeg", "-y", "-i", inp, "-vn", "-acodec", "aac", audio],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "f_%06d.png")]

            if os.path.exists(audio):
                cmd += ["-i", audio, "-c:a", "aac"]

            cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out]

            subprocess.run(cmd)

        print("Done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-w", type=int, default=120)
    ap.add_argument("-H", type=int, default=68)
    ap.add_argument("-fs", type=int, default=16)
    args = ap.parse_args()

    CinemaASCII().convert(args.input, args.output, args.w, args.H, args.fs)


if __name__ == "__main__":
    main()
