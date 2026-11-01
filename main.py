import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageFilter


def rgb2hsv(img):
    img = img.astype(np.float64) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.max(img, axis=-1)
    mn = np.min(img, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff != 0

    idx = mask & (mx == r)
    h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx])) % 360
    idx = mask & (mx == g)
    h[idx] = 60 * ((b[idx] - r[idx]) / diff[idx]) + 120
    idx = mask & (mx == b)
    h[idx] = 60 * ((r[idx] - g[idx]) / diff[idx]) + 240

    s = np.where(mx != 0, diff / np.where(mx == 0, 1, mx), 0)
    v = mx
    return h, s, v


def hsv2rgb(h, s, v):
    h = h % 360
    c = v * s
    x = c * (1 - np.abs((h / 60.0) % 2 - 1))
    m = v - c

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    hh = (h // 60).astype(int)

    m0 = hh == 0;
    r[m0], g[m0], b[m0] = c[m0], x[m0], 0
    m1 = hh == 1;
    r[m1], g[m1], b[m1] = x[m1], c[m1], 0
    m2 = hh == 2;
    r[m2], g[m2], b[m2] = 0, c[m2], x[m2]
    m3 = hh == 3;
    r[m3], g[m3], b[m3] = 0, x[m3], c[m3]
    m4 = hh == 4;
    r[m4], g[m4], b[m4] = x[m4], 0, c[m4]
    m5 = hh == 5;
    r[m5], g[m5], b[m5] = c[m5], 0, x[m5]

    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def gauss_kernel(size=15, sigma=2.0):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def convolve_channel(ch, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(ch, ((ph, ph), (pw, pw)), mode='edge')
    out = np.zeros_like(ch, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + ch.shape[0], j:j + ch.shape[1]]
    return out


def low_pass_filter(img, ksize=15, sigma=2.0):
    img = img.copy()
    h, s, v = rgb2hsv(img)
    k = gauss_kernel(ksize, sigma)
    W = img.shape[1]

    s_blur = convolve_channel(s, k)
    v_blur = convolve_channel(v, k)
    h_blur = convolve_channel(h, k)

    left = slice(0, W // 2)
    right = slice(W // 2, W)

    s[:, left] = s_blur[:, left]
    v[:, left] = v_blur[:, left]

    s[:, right] = s_blur[:, right]
    h[:, right] = h_blur[:, right]

    return hsv2rgb(h, s, v)


def high_pass_difference(img):
    blurred_img = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=2.5))
    img_array = np.array(blurred_img)

    gray = 0.2125 * img_array[..., 0] + 0.7154 * img_array[..., 1] + 0.0721 * img_array[..., 2]
    gray = gray.astype(np.float64) / 255.0

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    gpx = convolve_channel(gray, prewitt_x)
    gpy = convolve_channel(gray, prewitt_y)
    gp = np.sqrt(gpx ** 2 + gpy ** 2)

    gsx = convolve_channel(gray, sobel_x)
    gsy = convolve_channel(gray, sobel_y)
    gs = np.sqrt(gsx ** 2 + gsy ** 2)

    diff = np.abs(gp - gs)
    mx = diff.max()
    if mx > 0:
        diff = diff / mx * 255

    diff = diff[:img.shape[0], :img.shape[1]]

    return diff.astype(np.uint8)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Свёрточные фильтры — ФНЧ и ФВЧ (вариант 20)")
        root.geometry("1000x750")

        self.img = None
        self.low = None
        self.high = None

        top = tk.Frame(root)
        top.pack(pady=5)

        tk.Button(top, text="Открыть изображение", command=self.open_file).grid(row=0, column=0, padx=4)
        tk.Button(top, text="Применить ФНЧ", command=self.apply_low).grid(row=0, column=1, padx=4)
        tk.Button(top, text="Применить ФВЧ", command=self.apply_high).grid(row=0, column=2, padx=4)

        tk.Label(top, text="Размер ядра ФНЧ").grid(row=1, column=0)
        self.e_size = tk.Entry(top, width=6);
        self.e_size.insert(0, "15")
        self.e_size.grid(row=1, column=1)

        tk.Label(top, text="Сигма ФНЧ").grid(row=1, column=2)
        self.e_sigma = tk.Entry(top, width=6);
        self.e_sigma.insert(0, "2.0")
        self.e_sigma.grid(row=1, column=3)

        tk.Button(top, text="Сохранить ФНЧ", command=lambda: self.save(self.low)).grid(row=2, column=0, columnspan=2,
                                                                                       pady=4)
        tk.Button(top, text="Сохранить ФВЧ", command=lambda: self.save(self.high)).grid(row=2, column=2, columnspan=2,
                                                                                        pady=4)

        panels = tk.Frame(root);
        panels.pack(fill="both", expand=True)

        self.lbl_in = self._panel(panels, "Исходное изображение", 0, 0)
        self.lbl_low = self._panel(panels, "ФНЧ фильтр", 0, 1)
        self.lbl_hi = self._panel(panels, "ФВЧ", 1, 0)

        desc = tk.Label(panels, justify="left", anchor="nw", font=("Arial", 9),
                        text=("Описание фильтров:\n\n"
                              "ФНЧ:\n"
                              "• Левая половина: размытие S и V\n"
                              "• Правая половина: размытие S и H\n\n"
                              "ФВЧ:\n"
                              "• Вычислены градиенты Прюитта и Собела\n"
                              "• Результат — модуль разности"))
        desc.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)

        for i in range(2):
            panels.rowconfigure(i, weight=1)
        for j in range(2):
            panels.columnconfigure(j, weight=1)

    def _panel(self, parent, title, r, c):
        f = tk.Frame(parent, relief="groove", bd=1)
        f.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
        tk.Label(f, text=title).pack()
        lbl = tk.Label(f, bg="white")
        lbl.pack(fill="both", expand=True)
        return lbl

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.bmp *.png *.jpg *.jpeg *.ppm *.pgm *.tif *.tiff")])
        if not path:
            return
        self.img = np.array(Image.open(path).convert("RGB"))
        self._show(self.img, self.lbl_in)

    def apply_low(self):
        if self.img is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте файл")
            return
        size = int(self.e_size.get())
        if size % 2 == 0:
            size += 1
        sigma = float(self.e_sigma.get())
        self.low = low_pass_filter(self.img, size, sigma)
        self._show(self.low, self.lbl_low)

    def apply_high(self):
        if self.img is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте файл")
            return
        self.high = high_pass_difference(self.img)
        self._show(self.high, self.lbl_hi, mode="L")

    def save(self, data):
        if data is None:
            messagebox.showwarning("Нет результата", "Сначала примените фильтр")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".ppm",
            filetypes=[
                ("PPM Image", "*.ppm"),
                ("PNG Image", "*.png"),
                ("BMP Image", "*.bmp")
            ]
        )

        if path:
            if data.ndim == 2:
                im = Image.fromarray(data, mode="L")
            else:
                im = Image.fromarray(data, mode="RGB")

            try:
                im.save(path)
            except Exception as e:
                messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файл:\n{e}")

    def _show(self, arr, widget, mode="RGB"):
        if mode == "L" or (arr.ndim == 2):
            im = Image.fromarray(arr)
        else:
            im = Image.fromarray(arr, mode="RGB")

        im.thumbnail((450, 350))
        ph = ImageTk.PhotoImage(im)
        widget.config(image=ph)
        widget.image = ph


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()