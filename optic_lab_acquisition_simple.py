import numpy as np
import time
import os
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from pyAndorSDK3 import AndorSDK3
# Utilities for image processing
#https://bi1.caltech.edu/2017/code/t04_quantitative_image_processing.html
import skimage.io
import skimage.exposure
from matplotlib.animation import FuncAnimation
from skimage.measure import label, regionprops

plt.close("all")
timestr = time.strftime("%Y-%m-%d_%H-%M")
print("Fecha:", timestr)

prismWollas = ""#  input("¿ Has puesto el prisma ? : ").strip()
estudio = "si Heat/si PBS"

if prismWollas in ["yes", "y", "Y", "Yes", "si", "Si", "SI"]:
    grad_prismWollas = float(input("Graduación del prism (en º): "))
    estudio = "no Heat/si PBS"

def unpack_mono12_packed(buffer, width, height):
    buffer = np.frombuffer(buffer, dtype=np.uint8)
    assert len(buffer) % 3 == 0, "El buffer no tiene un tamaño válido para Mono12Packed"

    b0 = buffer[0::3]
    b1 = buffer[1::3]
    b2 = buffer[2::3]

    pixel0 = (b0.astype(np.uint16) << 4) | (b1 & 0x0F)
    pixel1 = (b2.astype(np.uint16) << 4) | (b1 >> 4)

    pixels = np.empty(pixel0.size + pixel1.size, dtype=np.uint16)
    pixels[0::2] = pixel0
    pixels[1::2] = pixel1

    assert pixels.size == width * height, "Tamaño de imagen incompatible"
    return pixels.reshape((height, width))

def process_image(acquisition, width, height):
    raw_data = acquisition._np_data.tobytes()
    img = unpack_mono12_packed(raw_data, width, height)
    acquisition._np_data = img
    return acquisition

def count_clusters(img, thr):
    mask = img > thr
    labeled = label(mask, connectivity=2)
    return labeled.max()

def custom_acquire_series(cam, frame_count, width, height):
    timeout = 15000 # can determine the maximum exposure too
    cam.TriggerMode = "Software"
    cam.CycleMode = "Fixed"
    cam.FrameCount = frame_count

    imgsize = cam.ImageSizeBytes
    for _ in range(frame_count):
        buf = np.empty((imgsize,), dtype='B') # "B" es uint8
        cam.queue(buf, imgsize)

    series = deque() #¿necesario si no haces appendleft()?
    try:# start 
        cam.AcquisitionStart()  
        for frame in range(frame_count):
            cam.SoftwareTrigger()
            acq = cam.wait_buffer(timeout)
            acq = process_image(acq, width, height)
            series.append(acq)
            print(f"{(frame + 1) / frame_count * 100:.0f}% complete series", end="\r")
    finally:# stop
        cam.AcquisitionStop()
        cam.flush()
        # acq.show(cmap="gray")
        print(type(acq),type(series))

    return list(series)

# :: --- START LIVE PROJECTION
def live_image(cam, frame_count, width, height):
    timeout = 15000 # can determine the maximum exposure too
    cam.TriggerMode = "Software"
    cam.CycleMode = "Fixed"
    cam.FrameCount = frame_count
    imgsize = cam.ImageSizeBytes
    for _ in range(frame_count):
        buf = np.empty((imgsize,), dtype='B') # "B" es uint8
        cam.queue(buf, imgsize)
    series = deque() #¿necesario si no haces appendleft()?
    
    try:# start 
        cam.AcquisitionStart()  
        for _ in range(frame_count):
            cam.SoftwareTrigger()
            acq = cam.wait_buffer(timeout)
            acq = process_image(acq, width, height)
    finally:# stop
        cam.AcquisitionStop()
        cam.flush()
    return acq._np_data

def live_projection(cam,frame_count,width,height):
    fig,ax_img = plt.subplots(1,1,figsize=(8,8))
    thresh = 103
    img = live_image(cam,frame_count,width,height)
    img = img > thresh
    display = ax_img.imshow(img, origin="lower",cmap="gray")
    
    def update_live(frame):
        img_ = live_image(cam,frame_count,width,height)
        img_ = img_ > thresh
        display.set_data(img_)

    ani = FuncAnimation(fig, update_live, interval=1, cache_frame_data=False)

    def close(event):
        if event.key == 'q':
            plt.close(event.canvas.figure)

    cid =fig.canvas.mpl_connect("key_press_event", close)

    plt.show()
# :: --- END LIVE PROJECTION


def normalize_im(im):
    """
    Normalizes a given image such that the values range between 0 and 255, since 
    we work with uint8 data type.     
    """
    im_norm = (im - im.min()) / (im.max() - im.min())
    return im_norm

def plot_and_save_first_frame(acqs, gain, exposure, gate_width_sec, threshold, ZOOM=False):
    if not acqs:
        print("No hay frames para mostrar.")
        return

    img = acqs[0]._np_data

    # --- Main figure setup ---
    fig, (ax_img, ax_hist) = plt.subplots(
        1, 2, figsize=(14, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25, wspace=0.4)

    # --- Initial image ---
    thresh_val = threshold
    thresh_im = img > thresh_val
    im_display = ax_img.imshow(thresh_im, origin="lower", cmap="gray")
    counts = label(thresh_im, connectivity=2)
    ax_img.set_title(f"Thresholded image (thr={thresh_val}), counts={counts.max()}")

    if ZOOM:
        # --- Initial histogram (only thresholded pixels) ---
        valid_pixels = img[img > thresh_val]
        if valid_pixels.size > 0:
            hist, bins = skimage.exposure.histogram(valid_pixels)
        else:
            hist, bins = np.array([0]), np.array([0])
    else:
    # --- Initial histogram ---
        hist, bins = skimage.exposure.histogram(img)
    (hist_line,) = ax_hist.plot(bins, hist, color="black", linewidth=1)
    hist_thresh_line = ax_hist.axvline(thresh_val, color="red", linestyle="--", label="Threshold")
    ax_hist.set_title("Histogram of gray values")
    ax_hist.set_xlabel("Gray value (0–4095 for 12-bit image)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend()
    ax_hist.grid(True)

    # --- Colorbar for the image ---
    fig.colorbar(im_display, ax=ax_img, shrink=0.8, pad=0.02)

    # --- Slider setup ---
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, label='Threshold', valmin=95, valmax=125, valinit=thresh_val, valstep=1)

    # --- Patch collection for region boxes ---
    pc = PatchCollection([], alpha=0.8)
    ax_img.add_collection(pc)

    # --- Update function ---
    def update(thresh_val):
        # Update thresholded image
        new_thresh_im = img > thresh_val
        new_counts = label(new_thresh_im, connectivity=2)
        im_display.set_data(new_thresh_im)
        ax_img.set_title(f"Thresholded image (thr={thresh_val}), counts={new_counts.max()}")

        # Update patches (region boxes)
        new_patches = []
        if new_counts.max() < 500:
            for region in regionprops(new_counts):
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle(
                    (minc, minr), maxc - minc, maxr - minr, fill=False, linewidth=2
                )
                new_patches.append(rect)
        pc.set_paths(new_patches)

        # --- Update histogram ---
        if ZOOM:
            # --- Update histogram (only thresholded pixels) ---
            valid_pixels = img[img > thresh_val]
            if valid_pixels.size > 0:
                hist, bins = skimage.exposure.histogram(valid_pixels)
            else:
                hist, bins = np.array([0]), np.array([0])
        else:
            hist, bins = skimage.exposure.histogram(img)
        hist_line.set_data(bins, hist)
        hist_thresh_line.set_xdata([thresh_val, thresh_val])
        ax_hist.relim()
        ax_hist.autoscale_view()

        fig.canvas.draw_idle()

    # Initialize and connect slider
    update(thresh_val)
    slider.on_changed(update)
    plt.show()

def save_frames(acqs, gain, frames, exposure, gatemode, gate_width_sec):
    imgs = np.stack([acq._np_data for acq in acqs])
    name = f"gn{gain}_n{frames}_t{exposure}_gate_{gatemode}"
    if gatemode == "DDG":
        name += f"_width{gate_width_sec:.2e}"
    name = name.replace(".", "p")
    path = "acqui-pics//" + name + "_" + timestr + ".npz"
    np.savez_compressed(path, images=imgs)
    print(f"Frames guardados en {os.path.abspath(path)}")

def main():
    #photons_target = float(input("Cantidad deseada de fotones por píxel y segundo: "))
    #frame_count = int(input("Cantidad de fotos a capturar: "))
    #exposure_time = float(input("Tiempo de exposición (en segundos): "))
    #gatemode = input("Modo de Gate (CW On, CW Off, Fire only, Gate only, Fire and Gate, DDG): ").strip(
    frame_count = 1
    exposure_time = 2.5e-3 # in seconds
    Gain = 4095
    AOIHeight = 720# 500 #2160
    AOIWidth = 1380#1350 #2540 #2560 #2560
    AOITop = 90#150#250 #1
    AOILeft = 600#450 #1 
    threshold = 100
    #gate_width_sec = float(input("Duración del gate (en segundos): "))
    gate_width_sec = 0.05e-3#2e-9
    gate_width_ps = int(gate_width_sec * 1e12)  # convertir a picosegundos

    sdk3 = AndorSDK3()
    cam = sdk3.GetCamera(0)
    print("Cámara conectada:", cam.SerialNumber)
    cam.SensorCooling = True
    while cam.SensorTemperature > 3.0:
        print(f"Temperature: {cam.SensorTemperature:.2f}C")
        if cam.TemperatureStatus == "Fault":
            raise RuntimeError("Fallo en la refrigeración del sensor")
        time.sleep(5)
    print("Sensor estabilizado.")
    gatemode = "DDG"
    cam.GateMode = gatemode
    if gatemode == "DDG":
        cam.DDGOpticalWidthEnable = True
        cam.DDGOutputWidth = gate_width_ps
        print(f"DDGOutputWidth configurado a {gate_width_ps} ps")
    print(f"Test to know size {cam.ImageSizeBytes}")
    cam.MCPGain = Gain
    cam.AOIHeight = AOIHeight #2150 #2160
    cam.AOIWidth = AOIWidth #2540 #2560
    cam.AOITop = AOITop #1
    cam.AOILeft = AOILeft #1
    cam.ExposureTime = exposure_time
    cam.PixelEncoding = "Mono12Packed"
    cam.FrameRate = cam.max_FrameRate

    width, height = cam.AOIWidth, cam.AOIHeight
    gain_target = cam.MCPGain
    acqs = custom_acquire_series(cam, frame_count, width, height)
    
    
    #
    # print("\n Adquisición completada.")
    # print(acqs,",type:",type(acqs),",acq[0] type:",type(acqs[0]),",len(acq0)=",len(acqs),",img.np_data shape:",acqs[0]._np_data.shape)
    # plot_and_save_first_frame(acqs, gain_target, exposure_time, gate_width_sec,threshold)
    # save_frames(acqs, gain_target, frame_count, exposure_time, gatemode, gate_width_sec)
    plt.show()

    live_projection(cam,frame_count,width,height)

if __name__ == "__main__":
    main()
