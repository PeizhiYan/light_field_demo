##################################################
##################################################
## Author: LFR-project team
## 
## Demonstration GUI Application
##      Click-and-Refocus
##
##################################################
##################################################

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import os
from functools import partial
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim

W = 2048 # width
H = 1088 # height
resize = 2 # resize to 1/4 for demo

z0 = 100
z1 = 1.63

z_values = {
    '01': [1.86798, 4.42554],
    '02': [1.80353, 4.27669],
    '03': [1.80353, 4.27669],
    '04': [1.83155, 4.34117],
    '05': [1.83725, 4.35469],
    '06': [1.79141, 4.24794],
    '07': [1.79141, 4.24794],
    '08': [1.89206, 4.48125],
    '09': [1.84239, 4.36621],
    '10': [1.83473, 4.34871],
    '11': [1.83473, 4.34871],
    '12': [1.83703, 4.3535],
    '13': [1.86696, 4.42311],
    '14': [1.82547, 4.32677],
    '15': [1.82547, 4.32677],
    '16': [1.84489, 4.37213]
} # from DERS. format: [nearest, farthest]

cam_lookup = {
    '01': [0,0],
    '02': [0,1],
    '03': [0,2],
    '04': [0,3],
    '05': [1,0],
    '06': [1,1],
    '07': [1,2],
    '08': [1,3],
    '09': [2,0],
    '10': [2,1],
    '11': [2,2],
    '12': [2,3],
    '13': [3,0],
    '14': [3,1],
    '15': [3,2],
    '16': [3,3]
}

""" Define the Shift Matrix """
## Reference:
##  N. Sabater et al., "Dataset and Pipeline for Multi-view Light-Field Video", 
##  2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 
##  Honolulu, HI, 2017, pp. 1743-1753.
shiftMat = np.zeros([4,4,2]) # The base Shift Matrix is relative to view (1,1)
shiftMat[:,:,0] = np.array([[100,-0.36,-97.19,-195.55],
                           [98.67,0,-96.18,-197.85],
                           [99.17,0.21,-98.33,-197],
                           [99.08,-1.22,-99.26,-198.36]])/resize
shiftMat[:,:,1] = np.array([[98.28,98.14,98.07,97.35],
                            [-1.73,0,0.74,0.11],
                            [-99.93,-99.11,-101.12,-99.07],
                            [-197.68,-198.14,-198.89,-199.37]])/resize
def changeBaseView(s,t):
    # Compute New Shift Matrix for Camera at (s,t)
    shiftMat[:,:,0] = shiftMat[:,:,0] - shiftMat[s,t,0]
    shiftMat[:,:,1] = shiftMat[:,:,1] - shiftMat[s,t,1]

def compute_d(z):
    # compute d(z)
    d = ((1/z)-(1/z0))/((1/z1)-(1/z0)) 
    return d

def translateImg(img, offsetU, offsetV):
    # Use affine transform to shift image
    M = np.float32([[1,0,offsetU],[0,1,offsetV]])
    ret = cv2.warpAffine(img, M, (W//2,H//2))
    return ret

def shift_and_sum(img, s, t, refocusedImage, z):
    d = compute_d(z)
    offsetU = int(-1*d*shiftMat[s,t,0])
    offsetV = int(-1*d*shiftMat[s,t,1])
    refocusedImage = refocusedImage + translateImg(img, offsetU, offsetV)
    return refocusedImage

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W//resize, H//resize))
    return img

def read_depth(path):
    img = cv2.imread(path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W//resize, H//resize))
    return img

def refocus(path, frame, z):
    refocusedImage = np.zeros([H//2,W//2,3], dtype=float)
    divisor = 0
    for i in range(16):
        cam = "{:02d}".format(i+1)
        img = read_image(os.path.join(path, cam + '\\{:05d}.png'.format(frame)))
        (s,t) = cam_lookup[cam]
        refocusedImage = shift_and_sum(img, s, t, refocusedImage, z)
        divisor += 1
    refocusedImage = refocusedImage/divisor
    return np.array(refocusedImage, dtype=np.uint8)

class Application(tk.Frame):
    def __init__(self, root=None):
        super().__init__(root)
        self.root = root
        self.pack()
        self.current_frame = tk.IntVar()
        self.n_frames = 0 # numbe of frames
        self.ref_cam = 6 # default camera
        self.current_depth = 0
        self.z = tk.DoubleVar()
        self.search = tk.IntVar(0)
        self.video_path = 'C:\\Users\\yanpe\\Desktop\\data\\Painter_4x4_png'
        self.depth_path = 'C:\\Users\\yanpe\\Desktop\\data\\Painter_4x4_depth_png'
        self.current_view_image = None
        self.current_depth_image = None
        self.create_GUI()

    def create_GUI(self):
        # Video Path
        label1 = tk.Label(self.root, text="Video Files Path")
        label1.pack()
        label1.place(height=20, width=100, x=10, y=10)
        self.video_path_field = tk.Text(self.root)
        self.video_path_field.pack()
        self.video_path_field.place(height=20, width=800, x=120, y=10)
        
        # Depth File Path
        label2 = tk.Label(self.root, text="Depth Files Path")
        label2.pack()
        label2.place(height=20, width=100, x=10, y=40)
        self.depth_path_field = tk.Text(self.root)
        self.depth_path_field.pack()
        self.depth_path_field.place(height=20, width=800, x=120, y=40)
        
        # Load Button
        self.btn_load = tk.Button(self.root)
        self.btn_load["text"] = "Load Data"
        self.btn_load["command"] = self.load_data
        self.btn_load.pack()
        self.btn_load.place(height=50, width=110, x=930, y=10)

        # Frame Selector
        label3 = tk.Label(self.root, text="Frame Selector")
        label3.pack()
        label3.place(height=20, width=100, x=10, y=100)
        self.scale = tk.Scale(self.root, variable = self.current_frame, from_=1, orient='horizontal', length=1024, command=self.scale_event)
        self.scale.pack()
        self.scale.place(x = 10, y=120)

        # Image Display
        self.panel = tk.Label() #tk.Canvas(self.root, width=W//2, height=H//2)
        self.panel.pack()
        self.panel.place(width=W//2, height=H//2, x=10, y=200)
        self.panel.bind('<Button-1>', self.panel_event)

        # Search
        self.search_ckb = tk.Checkbutton(self.root, text='Search',variable=self.search, onvalue=1, offvalue=0)
        self.search_ckb.pack()
        self.search_ckb.place(x = 1050, y=100)

        # Depth Selector
        label6 = tk.Label(self.root, text="Depth Selector")
        label6.pack()
        label6.place(height=20, width=190, x=1050, y=200)
        self.zscale = tk.Scale(self.root, variable = self.z, from_=1, to=5, orient='horizontal', length=170, resolution=0.1, command=self.zscale_event)
        self.zscale.pack()
        self.zscale.place(x = 1050, y=220)

        # Selected Depth Value
        label5 = tk.Label(self.root, text="Selected Depth")
        label5.pack()
        label5.place(height=20, width=190, x=1050, y=300)
        self.labelz = tk.Label(self.root)
        self.labelz.config(font=("Courier", 15))
        self.labelz['text'] = 'z = {:.3f} meter'.format(0)
        self.labelz.pack()
        self.labelz.place(height=40, width=190, x=1050, y=330)

        # Reference Camera Selector
        label4 = tk.Label(self.root, text="Reference Camera Selector")
        label4.pack()
        label4.place(height=20, width=190, x=1050, y=400)
        self.btn_cams = []
        for i in range(16):
            btn_cam = tk.Button(self.root, command=partial(self.select_ref_cam, "{:02d}".format(i+1)))
            btn_cam["text"] = "{:02d}".format(i+1)
            if self.ref_cam - 1 == i:
                btn_cam.configure(bg = "cyan")
            else:
                btn_cam.configure(bg = "white")
            btn_cam.pack()
            btn_cam.place(height=40, width=40, x=1050+(i%4)*50, y=430+(i//4)*50)
            self.btn_cams.append(btn_cam)

        # Copyright
        labelcr = tk.Label(self.root, text="LFR Team - 2021\n Copy Right")
        labelcr.pack()
        labelcr.place(height=50, width=190, x=1050, y=695)
        
    def load_data(self):
        if len(self.video_path_field.get(1.0, 'end-1c')) > 0:
            self.video_path = self.video_path_field.get(1.0, 'end-1c')
            self.depth_path = self.depth_path_field.get(1.0, 'end-1c')
        else:
            self.video_path_field.insert(1.0, self.video_path)
            self.depth_path_field.insert(1.0, self.depth_path)
        n_frames_video = len(list(os.listdir(os.path.join(self.video_path,'01'))))
        print("video frames ", n_frames_video)
        n_frames_depth = len(list(os.listdir(os.path.join(self.depth_path,'01'))))
        print("depth frames ", n_frames_depth)
        self.n_frames = min(n_frames_depth, n_frames_video)
        self.scale.configure(to=self.n_frames)
        current_frame = self.current_frame.get()
        img_path = os.path.join(self.video_path, "{:02d}\\{:05d}.png".format(self.ref_cam, current_frame))
        self.current_view_image = read_image(img_path)
        dpt_path = os.path.join(self.depth_path, "{:02d}\\{:05d}.png".format(self.ref_cam, current_frame))
        self.current_depth_image = read_depth(dpt_path)
        img = ImageTk.PhotoImage(image=Image.fromarray(self.current_view_image))
        self.panel.configure(image = img)
        self.panel.image = img

    def zscale_event(self, val=None):
        z = self.z.get()
        self.current_depth = z
        self.labelz['text'] = 'z = {:.3f} meter'.format(z)
        img = refocus(path=self.video_path, frame=self.current_frame.get(), z=z)
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.panel.configure(image = img)
        self.panel.image = img

    def scale_event(self, val=None):
        current_frame = self.current_frame.get()
        img_path = os.path.join(self.video_path, "{:02d}\\{:05d}.png".format(self.ref_cam, current_frame))
        self.current_view_image = read_image(img_path)
        dpt_path = os.path.join(self.depth_path, "{:02d}\\{:05d}.png".format(self.ref_cam, current_frame))
        self.current_depth_image = read_depth(dpt_path)
        img = ImageTk.PhotoImage(image=Image.fromarray(self.current_view_image))
        self.panel.configure(image = img)
        self.panel.image = img

    def select_ref_cam(self, cam):
        print("reference camera changed to ", cam)
        self.btn_cams[self.ref_cam-1].configure(bg = 'white')
        self.ref_cam = int(cam)
        self.btn_cams[self.ref_cam-1].configure(bg = 'cyan')
        current_frame = self.current_frame.get()
        img_path = os.path.join(self.video_path, "{:02d}\\{:05d}.png".format(self.ref_cam, current_frame))
        self.current_view_image = read_image(img_path)
        dpt_path = os.path.join(self.depth_path, "{:02d}\\{:05d}.png".format(self.ref_cam, current_frame))
        self.current_depth_image = read_depth(dpt_path)
        img = ImageTk.PhotoImage(image=Image.fromarray(self.current_view_image))
        self.panel.configure(image = img)
        self.panel.image = img
        changeBaseView(s=cam_lookup[cam][0],t=cam_lookup[cam][1])

    def panel_event(self, event):
        """ Click and refocus """
        print("click event ", event.x, event.y)
        z = self.get_depth(self.current_depth_image, event.x, event.y)
        self.z.set(z)
        self.current_depth = z
        self.labelz['text'] = 'z = {:.3f} meter'.format(z)
        img_ = refocus(path=self.video_path, frame=self.current_frame.get(), z=z)
        img = ImageTk.PhotoImage(image=Image.fromarray(img_))
        self.panel.configure(image = img)
        self.panel.image = img
        SEARCH_STEPS = 6
        SEARCH_STRIDE = 0.1
        best_z = z
        if self.search.get() == 1:
            #print(img_.shape)
            #print(self.current_view_image.shape)
            best_ssim = ssim(self.get_patch(img_, event.x, event.y),
                            self.get_patch(self.current_view_image, event.x, event.y))
            for i in range(SEARCH_STEPS):
                test_z = z + SEARCH_STRIDE * SEARCH_STEPS//2 - i * SEARCH_STRIDE
                if test_z < 1 or test_z > 4:
                    continue
                self.labelz['text'] = 'searching...\nz = {:.3f} meter'.format(test_z)
                img_ = refocus(path=self.video_path, frame=self.current_frame.get(), z=test_z)
                img = ImageTk.PhotoImage(image=Image.fromarray(img_))
                self.panel.configure(image = img)
                self.panel.image = img
                self.root.update()
                test_ssim = ssim(self.get_patch(img_, event.x, event.y),
                            self.get_patch(self.current_view_image, event.x, event.y))
                print(i, test_z, test_ssim)
                if test_ssim > best_ssim:
                    best_z = test_z
                    best_ssim = test_ssim
        z = best_z
        self.z.set(z)
        self.current_depth = z
        self.labelz['text'] = 'z = {:.3f} meter'.format(z)
        img_ = refocus(path=self.video_path, frame=self.current_frame.get(), z=z)
        img = ImageTk.PhotoImage(image=Image.fromarray(img_))
        self.panel.configure(image = img)
        self.panel.image = img
    
    def get_patch(self, img, x, y):
        size = 100 #  pixels
        xa = x-size//2
        xb = x+size//2
        ya = y-size//2
        yb = y+size//2
        if xa < 0:
            xa = 0
            xb = size
        if ya < 0:
            ya = 0
            yb = size
        patch = np.array(img[ya:yb, xa:xb, 1], dtype=np.int8) # only green channel
        return patch
    
    def get_depth(self, depth_map, x, y):
        size = 6 #  pixels
        xa = x-size//2
        xb = x+size//2
        ya = y-size//2
        yb = y+size//2
        if xa < 0:
            xa = 0
            xb = size
        if ya < 0:
            ya = 0
            yb = size
        cam = "{:02d}".format(self.ref_cam)
        [z_min, z_max] = z_values[cam]
        z_range = z_max - z_min
        patch = np.array(depth_map[ya:yb, xa:xb], dtype=np.float)
        patch = (255-patch)/255.
        patch = patch * z_range + z_min
        z_min_ = np.min(patch)
        z = np.mean(patch)
        z_max_ = np.max(patch)
        #if z_max_ - z_min_ > 0.5:
        #    return z_max_
        return z


root = tk.Tk(className=" Click and Refocusing Demo - GUI Application")
#root.configure(background='gray')
root.geometry("1250x768")
app = Application(root=root)
app.mainloop()



