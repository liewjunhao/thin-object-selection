import tkinter as tk
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import copy
import ui.seg_model as seg_model
import dataloaders.helpers as helpers
import imageio

class InteractiveDemo:
    def __init__(self, master=tk.Tk()):
        self.master = master
        self.master.title("Deep Interactive Thin Object Selection Demo")
        self.max_size = 1200

        self.filename = 'images/ants.jpg'
        self.image = np.array(Image.open(self.filename))
        self.clicks = []        # store user clicks
        self.mask = []          # store segmentation mask
        self._resize_image()    # resize image to fit the canvas
        self.canvas_image = self.resize_image.copy()

        self.canvas = ImageTk.PhotoImage(Image.fromarray(self.canvas_image))
        self.label = tk.Label(self.master, image=self.canvas)
        self.label.pack(side=tk.BOTTOM)
        self.label.bind('<Button-1>', self.on_click)

        self.btn_open_image = tk.Button(self.master, text='Open Image', command=self._open_image)
        self.btn_open_image.pack(side=tk.LEFT)
        self.btn_reset = tk.Button(self.master, text='Reset', command=self._reset)
        self.btn_reset.pack(side=tk.LEFT)
        self.btn_save_mask = tk.Button(self.master, text='Save Mask', command=self._save_mask)
        self.btn_save_mask.pack(side=tk.LEFT)

    def _resize_image(self):
        """ Resize image to fit canvas. """
        h, w = self.image.shape[:2]
        max_side = np.maximum(h, w)
        if max_side > self.max_size:
            self.sc = self.max_size / max_side
            self.resize_image = cv2.resize(self.image, (0,0), fx=self.sc, fy=self.sc, interpolation=cv2.INTER_LINEAR)
        else:
            self.sc = 1
            self.resize_image = self.image.copy()

    def _init_seg_model(self, net, cfg, device):
        """ Initialize segmentation model. """
        self.model = seg_model.SegModel(net, cfg, device)

    def _reset(self):
        self.clicks = [] # store user clicks
        self.mask = []   # store segmentation mask
        self.canvas_image = self.resize_image.copy() # rest image
        self._update_canvas()

    def _open_image(self):
        """ Open an image. """
        def _open_image_dialog():
            filename = filedialog.askopenfilename(title='Open')
            return filename
        self.filename = _open_image_dialog()
        self.image = np.array(Image.open(self.filename))
        self._resize_image()
        self._reset()
        self._update_canvas()

    def _update_canvas(self):
        """ Update the canvas. """
        self.canvas = ImageTk.PhotoImage(Image.fromarray(self.canvas_image))
        self.label.config(image=self.canvas)
        self.master.update_idletasks()

    def on_click(self, event):
        self._draw_click([int(event.x), int(event.y)]) # draw the latest click
        self.clicks.append([int(event.x / self.sc), int(event.y / self.sc)])
        # Automatically perform segmentation when there are 4 clicks
        if len(self.clicks) == 4:
            self.segment()

    def segment(self):
        """ Perform segmentation. """
        clicks = np.array(self.clicks).astype(np.int)
        self.mask = self.model._segment(self.image, clicks)
        self._visualize_mask()

    def _draw_click(self, click):
        """ Draw user clicks on the canvas image. """
        cv2.circle(self.canvas_image, (click[0], click[1]), 8, [255,0,0], -1)
        self._update_canvas()

    def _visualize_mask(self):
        """ Visualize segmentation mask. """
        mask = cv2.resize(self.mask, (0,0), fx=self.sc, fy=self.sc, interpolation=cv2.INTER_LINEAR)
        self.canvas_image = helpers.mask_image(self.canvas_image, mask>0.5, color=[0,255,255], alpha=0.8)
        self._update_canvas()

    def _save_mask(self):
        """ Save segmentation mask. """
        mask = (self.mask * 255).astype(np.uint8)
        names = self.filename.split('/')
        img_name = names[-1].split('.')[0] + '_mask.png'
        filename = names[:-1]
        filename.append(img_name)
        filename = '/'.join(filename)
        imageio.imwrite(filename, mask)
        print('Mask is saved to {}!'.format(filename))

    def mainloop(self):
        self.master.mainloop()