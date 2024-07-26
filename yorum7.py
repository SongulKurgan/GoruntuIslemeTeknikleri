import tkinter as tk  # GUI oluşturma
from tkinter import filedialog, messagebox, Toplevel, simpledialog  # Dosya seçici açma
from PIL import Image, ImageTk  # resim işleme
import numpy as np  # dizi işlemleri
import matplotlib.pyplot as plt  # grafik çizimi
from collections import deque 
import cv2

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Aracı")  # Pencere başlığını ayarlıyor
        self.root.geometry("800x600")  # Pencerenin boyutlarını ayarlıyor
        self.root.configure(bg="#f0f0f0")  # Pencerenin arka plan rengini ayarlıyor

        # Orijinal ve işlenmiş görüntüler için çerçeve
        self.frame_images = tk.Frame(root, bg="#f0f0f0")
        self.frame_images.pack(pady=10)  # Çerçeveyi pencereye yerleştiriyor

        # Orijinal görüntü çerçevesi
        self.frame_original = tk.LabelFrame(self.frame_images, text="Orijinal Görüntü", bg="#f0f0f0")
        self.frame_original.grid(row=0, column=0, padx=10)  # Çerçeveyi yerleştiriyor
        self.canvas_original = tk.Canvas(self.frame_original, width=256, height=256, bg="white")
        self.canvas_original.pack()  # Orijinal görüntü için tuvali oluşturuyor

        # İşlenmiş görüntü çerçevesi
        self.frame_processed = tk.LabelFrame(self.frame_images, text="İşlenmiş Görüntü", bg="#f0f0f0")
        self.frame_processed.grid(row=0, column=1, padx=10)  # Çerçeveyi yerleştirir
        self.canvas_processed = tk.Canvas(self.frame_processed, width=256, height=256, bg="white")
        self.canvas_processed.pack()  # İşlenmiş görüntü için tuvali oluşturur

        # RGB değerlerini gösteren çerçeve
        self.rgb_frame = tk.Frame(root, bg="#f0f0f0")
        self.rgb_frame.pack(pady=5)  # Çerçeveyi pencereye yerleştirir
        
        self.color_box = tk.Canvas(self.rgb_frame, width=20, height=20, bg="white", highlightthickness=1, highlightbackground="black")
        self.color_box.pack(side=tk.LEFT, padx=5)  # Renk kutusunu ekler
        
        self.rgb_label = tk.Label(self.rgb_frame, text="RGB Değerleri: (R, G, B)", bg="#f0f0f0")
        self.rgb_label.pack(side=tk.LEFT, padx=5)  # RGB değerlerini gösterecek etiketi ekler

        # Kontrol butonları çerçevesi
        self.btn_frame = tk.Frame(root, bg="#f0f0f0")
        self.btn_frame.pack(pady=10)  # Çerçeveyi pencereye yerleştirir

        # Görüntü yükleme butonu
        self.btn_load = tk.Button(self.btn_frame, text="Görüntü Yükle", command=self.load_image, bg="#4CAF50", fg="white", width=20)
        self.btn_load.grid(row=0, column=0, padx=10, pady=5)  # Butonu ekler
        
        # Histogram gösterme butonu
        self.btn_histogram = tk.Button(self.btn_frame, text="Histogram Göster", command=self.show_histogram, bg="#2196F3", fg="white", width=20)
        self.btn_histogram.grid(row=0, column=1, padx=10, pady=5)  # Butonu ekler
        
        
        # Histogram eşitleme değeri için etiket ve giriş kutusu
        self.label_equalize = tk.Label(self.btn_frame, text="Histogram Eşitleme Değeri:", bg="#f0f0f0")
        self.label_equalize.grid(row=1, column=0, padx=10, pady=5)  # Etiketi ekler
        
        self.entry_equalize = tk.Entry(self.btn_frame, width=23)
        self.entry_equalize.grid(row=1, column=1, padx=10, pady=5)  # Giriş kutusunu ekler
        
        # Histogram eşitleme butonu
        self.btn_equalize = tk.Button(self.btn_frame, text="Histogram Eşitle", command=self.equalize_histogram, bg="#FF9800", fg="white", width=20)
        self.btn_equalize.grid(row=2, column=0, columnspan=2,padx=10, pady=5)  # Butonu ekler
        
        self.btn_hist_intensity = tk.Button(self.btn_frame, text="Histogram Yoğunluğu", command=self.show_histogram_intensity, bg="#8BC34A", fg="white", width=20)
        self.btn_hist_intensity.grid(row=3, column=1, padx=10, pady=5)
        
        # Kontrast geliştirme butonu
        self.btn_contrast = tk.Button(self.btn_frame, text="Kontrast Geliştirme", command=self.enhance_contrast, bg="#FFEB3B", fg="black", width=20)
        self.btn_contrast.grid(row=3, column=0, padx=10, pady=5)
        
        self.btn_gray = tk.Button(self.btn_frame, text="Gray'e Dönüştür", command=self.convert_to_gray, bg="#9E9E9E", fg="white", width=20)
        self.btn_gray.grid(row=4, column=0, padx=10, pady=5)
        
        self.btn_rgb = tk.Button(self.btn_frame, text="RGB'ye Dönüştür", command=self.convert_to_rgb, bg="#FF5722", fg="white", width=20)
        self.btn_rgb.grid(row=4, column=1,  padx=10, pady=5)
        
        # Segmentasyon butonu
        self.btn_segment = tk.Button(self.btn_frame, text="Segmentasyon", command=self.segment_image, bg="#3F51B5", fg="white", width=20)
        self.btn_segment.grid(row=5, column=0,  padx=10, pady=5)  # Butonu ekler
        
        self.btn_segment_contour = tk.Button(self.btn_frame, text="Segmentasyon Contour", command=self.segment_contour, bg="#3F51B5", fg="white", width=20)
        self.btn_segment_contour.grid(row=5, column=1,  padx=10, pady=5)
        
        # Filtre uygulama butonu
        self.btn_filter = tk.Button(self.btn_frame, text="Filtre Uygula", command=self.show_filter_menu, bg="#F44336", fg="white", width=20)
        self.btn_filter.grid(row=6, column=0,  padx=10, pady=5)   # Butonu ekler 
        
        # Kenar tespiti butonu
        self.btn_edge_detection = tk.Button(self.btn_frame, text="Kenar Algılama", command=self.show_edge_menu, bg="#607D8B", fg="white", width=20)
        self.btn_edge_detection.grid(row=6, column=1,  padx=10, pady=5)
        
         # Görüntü özelliklerini gösterme butonu
        self.btn_properties = tk.Button(self.btn_frame, text="Görüntü Özelliklerini Göster", command=self.show_image_properties, bg="#795548", fg="white", width=20)
        self.btn_properties.grid(row=7, column=0, padx=10, pady=5)  # Butonu ekler
        
         # Görüntüyü ikiliye dönüştürme butonu
        self.btn_convert_binary = tk.Button(self.btn_frame, text="Görüntüyü İkiliye Dönüştür", command=self.convert_to_binary, bg="#FFC107", fg="white", width=20)
        self.btn_convert_binary.grid(row=7, column=1, padx=10, pady=5)  # Butonu ekler
        
        # Görüntü matrisini gösterme butonu
        self.btn_show_matrix = tk.Button(self.btn_frame, text="Görüntü Matrisini Göster", command=self.show_image_matrix, bg="#9C27B0", fg="white", width=20)
        self.btn_show_matrix.grid(row=8, column=0, padx=10, pady=5)  # Butonu ekler
        
        # Morfolojik işlemler butonu
        self.btn_morphology = tk.Button(self.btn_frame, text="Morfolojik İşlemler", command=self.show_morphology_menu, bg="#3F51B5", fg="white", width=20)
        self.btn_morphology.grid(row=8, column=1,  padx=10, pady=5)  # Butonu ekler
        
        self.btn_region_growing = tk.Button(self.btn_frame, text="Region Growing Uygula", command=self.region_growing_menu, bg="#3F51B5", fg="white", width=20)
        self.btn_region_growing.grid(row=9, column=1, padx=10, pady=5)  # Butonu ekler
        
        # Özel Filtre butonu
        self.btn_custom_filter = tk.Button(self.btn_frame, text="Özel Filtre Uygula", command=self.apply_custom_filter_ui, bg="#673AB7", fg="white", width=20)
        self.btn_custom_filter.grid(row=9, column=0, padx=10, pady=5)  # Butonu ekler

        self.img = None  # Yüklenen görüntüyü tutar
        self.img_gray = None  # Gri tonlamaya dönüştürülmüş görüntüyü tutar
        self.img_display_original = None  # Orijinal görüntünün ekranda gösterilen hali
        self.img_display_processed = None  # İşlenmiş görüntünün ekranda gösterilen hali

    def load_image(self):
        # Kullanıcının yüklemesi için bir dosya seçici açar ve resmi yükler
        file_path = filedialog.askopenfilename()
        self.img = Image.open(file_path)  # Görüntüyü yükler
        self.img_gray = self.img.convert("L")  # Resmi gri tonlamaya dönüştürür
        self.display_image(self.img, self.canvas_original)  # Orijinal resmi görüntüler
        self.canvas_original.bind("<Motion>", self.show_rgb_values)  # Fare hareket ederken RGB değerlerini gösterir

    def display_image(self, img, canvas):
        # Belirtilen tuval üzerinde resmi görüntüler
        img_tk = ImageTk.PhotoImage(img)  # Resmi PhotoImage nesnesine dönüştürür
        canvas.create_image(0, 0, anchor="nw", image=img_tk)  # Resmi tuvale çizer
        canvas.image = img_tk  # Resmi tutar, böylece Garbage Collection tarafından silinmez

    def show_rgb_values(self, event):
        # Fare hareket ederken ilgili pikselin RGB değerlerini gösterir
        if self.img:
            x, y = event.x, event.y  # Fare koordinatlarını alır
            if 0 <= x < self.img.width and 0 <= y < self.img.height:
                r, g, b = self.img.getpixel((x, y))  # Pikselin RGB değerlerini alır
                self.rgb_label.config(text=f"RGB Değerleri: ({r}, {g}, {b})")  # RGB etiketini günceller
                self.color_box.create_rectangle(0, 0, 20, 20, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")  # Renk kutusunu günceller

    def show_histogram(self):
        # Gri tonlama histogramı
        histogram_gray = self.calculate_histogram(self.img_gray)
        
        # RGB histogramlarını hesaplar
        histogram_r, histogram_g, histogram_b = self.calculate_rgb_histogram(self.img)
        
        # Alt grafikler oluştur
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        # Gri tonlama histogramı
        axs[0, 0].bar(range(256), histogram_gray, color='gray')
        axs[0, 0].set_title("Gray Histogram")
        axs[0, 0].set_xlabel("Piksel Değeri")
        axs[0, 0].set_ylabel("Frekans")
        
        # Kırmızı kanal histogramı
        axs[0, 1].bar(range(256), histogram_r, color='red')
        axs[0, 1].set_title("Red Histogram")
        axs[0, 1].set_xlabel("Piksel Değeri")
        axs[0, 1].set_ylabel("Frekans")
        
        # Yeşil kanal histogramı
        axs[1, 0].bar(range(256), histogram_g, color='green')
        axs[1, 0].set_title("Green Histogram")
        axs[1, 0].set_xlabel("Piksel Değeri")
        axs[1, 0].set_ylabel("Frekans")
        
        # Mavi kanal histogramı
        axs[1, 1].bar(range(256), histogram_b, color='blue')
        axs[1, 1].set_title("Blue Histogram")
        axs[1, 1].set_xlabel("Piksel Değeri")
        axs[1, 1].set_ylabel("Frekans")
        
        plt.tight_layout()
        plt.show()

    def calculate_histogram(self, image):
        # Bir görüntünün histogramını hesaplar
        histogram = [0] * 256
        for pixel in image.getdata():  # Piksel değerlerini iterasyonla alır
            histogram[pixel] += 1  # Histogramı günceller
        return histogram

    def calculate_rgb_histogram(self, image):
        # RGB histogramlarını ayrı ayrı hesaplar
        histogram_r = [0] * 256
        histogram_g = [0] * 256
        histogram_b = [0] * 256
        for r, g, b in image.getdata():  # Piksel değerlerini iterasyonla alır
            histogram_r[r] += 1  # Kırmızı kanal histogramını günceller
            histogram_g[g] += 1  # Yeşil kanal histogramını günceller
            histogram_b[b] += 1  # Mavi kanal histogramını günceller
        return histogram_r, histogram_g, histogram_b

    def display_histogram(self, histogram, title, color="black"):
        # Histogram grafiğini gösterir
        plt.figure()  # Yeni bir grafik oluşturur
        plt.bar(range(256), histogram, color=color)  # Histogram çubuğunu çizer
        plt.title(title)  # Grafiğin başlığını ayarlar
        plt.xlabel("Piksel Değeri")  # X ekseni etiketini ayarlar
        plt.ylabel("Frekans")  # Y ekseni etiketini ayarlar
        plt.show()  # Grafiği gösterir

    def equalize_histogram(self):
        # Histogram eşitleme uygular (görüntünün kontrastını artırarak daha fazla ayrıntının görünmesini sağlar)
        alpha = float(self.entry_equalize.get())  # Kullanıcıdan alpha değerini alır
        equalized_img = self.histogram_equalization(self.img_gray, alpha)  # Histogram eşitleme uygular
        self.display_image(equalized_img, self.canvas_processed)  # Eşitlenmiş görüntüyü gösterir

    def histogram_equalization(self, image, alpha):
        # Histogram eşitleme işlemini manuel olarak gerçekleştirir
        histogram = self.calculate_histogram(image)  # Histogramı hesaplar
        cdf = [sum(histogram[:i + 1]) for i in range(len(histogram))]  # Kümülatif dağılım fonksiyonunu (CDF) hesaplar
        cdf_min = min(cdf)  # CDF'nin minimum değerini alır
        cdf_normalized = [(x - cdf_min) / (image.size[0] * image.size[1] - cdf_min) for x in cdf]  # CDF'yi normalize eder
        equalized_image = Image.new("L", image.size)  # Yeni bir boş görüntü oluşturur
        img_array = np.array(image)  # Görüntüyü numpy dizisine dönüştürür
        eq_array = np.zeros_like(img_array)  # Eşitlenmiş görüntü için boş bir dizi oluşturur
        for i in range(img_array.shape[0]):  # Satırları iterasyonla alır
            for j in range(img_array.shape[1]):  # Sütunları iterasyonla alır
                eq_array[i, j] = int(cdf_normalized[img_array[i, j]] * 255 * alpha)  # Eşitlenmiş piksel değerini hesaplar
        equalized_image = Image.fromarray(eq_array)  # Eşitlenmiş görüntüyü oluşturur
        return equalized_image

    def convert_to_binary(self):
        # Görüntüyü belirli bir eşik değerine göre ikili (binary) görüntüye dönüştürür
        if self.img_gray:
            threshold = 128  # Sabit bir eşik değeri kullanıyoruz, isteğe göre değiştirilebilir
            img_arr = np.array(self.img_gray)  # Görüntüyü diziye dönüştürür
            binary_img_arr = (img_arr > threshold) * 255  # İkili görüntüyü oluşturur
            self.img_binary = Image.fromarray(binary_img_arr.astype(np.uint8))  # Görüntüyü oluşturur

            # Yeni bir pencere oluştur ve ikili görüntüyü burada göster
            binary_window = Toplevel(self.root)
            binary_window.title("İkili Görüntü")
            binary_window.geometry("300x300")
            binary_window.configure(bg="#f0f0f0")

            binary_canvas = tk.Canvas(binary_window, width=256, height=256, bg="white")
            binary_canvas.pack(padx=10, pady=10)
            self.display_image(self.img_binary, binary_canvas)
    
    
    def show_filter_menu(self):
        # Filtreleme menüsünü gösterir
        filter_menu = tk.Menu(self.root, tearoff=0)
        filter_menu.add_command(label="Bulanıklaştırma", command=self.apply_blur)
        filter_menu.add_command(label="Yumuşatma", command=self.apply_smoothing)
        filter_menu.add_command(label="Keskinleştirme", command=self.apply_sharpening)
        try:
            filter_menu.tk_popup(self.btn_filter.winfo_rootx(), self.btn_filter.winfo_rooty() + self.btn_filter.winfo_height())
        finally:
            filter_menu.grab_release()

    def apply_blur(self):
        # Bulanıklaştırma filtresi uygular
        blur_kernel = np.ones((5, 5), np.float32) / 25  # Bulanıklaştırma çekirdeğini tanımlar
        filtered_image = self.apply_kernel(np.array(self.img), blur_kernel)  # Filtreyi uygular
        self.display_image(Image.fromarray(filtered_image), self.canvas_processed)  # Filtrelenmiş görüntüyü gösterir

    def apply_smoothing(self):
        # Yumuşatma filtresi uygular
        smoothing_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32) / 9  # Yumuşatma çekirdeğini tanımlar
        filtered_image = self.apply_kernel(np.array(self.img), smoothing_kernel)  # Filtreyi uygular
        self.display_image(Image.fromarray(filtered_image), self.canvas_processed)  # Filtrelenmiş görüntüyü gösterir

    def apply_sharpening(self):
        # Keskinleştirme filtresi uygular
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Keskinleştirme çekirdeğini tanımlar
        filtered_image = self.apply_kernel(np.array(self.img), sharpen_kernel)  # Filtreyi uygular
        self.display_image(Image.fromarray(filtered_image), self.canvas_processed)  # Filtrelenmiş görüntüyü gösterir

    def apply_kernel(self, image, kernel):
        # Kernel filtresi uygular
        kernel_size = len(kernel)  # Kernel boyutunu alır
        pad_size = kernel_size // 2  # Yastıklama boyutunu hesaplar
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')  # Görüntüyü yastıklar
        filtered_image = np.zeros_like(image)  # Filtrelenmiş görüntü için boş bir dizi oluşturur

        for channel in range(3):  # Her renk kanalı için işlemi gerçekleştir
            for i in range(image.shape[0]):  # Satırları iterasyonla alır
                for j in range(image.shape[1]):  # Sütunları iterasyonla alır
                    region = padded_image[i:i+kernel_size, j:j+kernel_size, channel]  # Çekirdek boyutunda bölgeyi alır
                    filtered_value = np.sum(region * kernel)  # Bölge ve çekirdek çarpımının toplamını hesaplar
                    filtered_image[i, j, channel] = np.clip(filtered_value, 0, 255)  # Filtrelenmiş değeri 0-255 aralığına sınırlar

        return filtered_image

    def segment_image(self):
        # Otomatik eşikleme kullanarak segmentasyon uygular (görüntüdeki piksel yoğunluklarının iki ayrı sınıfa ait olduğu varsayımına dayanarak optimal bir eşik değeri belirler)
        segmented_img = self.otsu_threshold_segmentation(np.array(self.img_gray))  # Segmentasyon uygular
        self.display_image(Image.fromarray(segmented_img), self.canvas_processed)  # Segment edilmiş görüntüyü gösterir

    def otsu_threshold_segmentation(self, image):
        # Otsu'nun yöntemi ile segmentasyon yapar (bir görüntüdeki piksel yoğunluklarının bimodal bir dağılım gösterdiği varsayımına dayanır, yani genellikle iki farklı sınıfa ait piksellerin olduğu düşünülür (örneğin, nesne ve arka plan).)
        histogram = self.calculate_histogram(self.img_gray)  # Histogramı hesaplar
        total = sum(histogram)  # Histogram toplamını alır
        sumB, wB, maximum, sum1 = 0, 0, 0, sum(i * histogram[i] for i in range(256))  # Otsu'nun yöntemi için başlangıç değerleri
        level = 0  # Eşik değeri başlangıç olarak 0
        for i in range(256):
            wB += histogram[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += i * histogram[i]
            mB = sumB / wB
            mF = (sum1 - sumB) / wF
            between = wB * wF * (mB - mF) ** 2
            if between > maximum:
                level = i
                maximum = between

        segmented_image = np.zeros_like(image)
        segmented_image[image > level] = 255
        return segmented_image
    
    def edge_detection(self):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        edge_img = self.apply_edge_detection(np.array(self.img_gray), sobel_x, sobel_y)
        self.display_image(Image.fromarray(edge_img), self.canvas_processed)
    
    def apply_edge_detection(self, image, kernel_x, kernel_y):
        kernel_size = len(kernel_x)
        pad_size = kernel_size // 2
        padded_image = np.pad(image, pad_size, mode='constant')
        edge_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size]
                gx = np.sum(region * kernel_x)
                gy = np.sum(region * kernel_y)
                edge_image[i, j] = np.sqrt(gx**2 + gy**2)
                edge_image[i, j] = np.clip(edge_image[i, j], 0, 255)
        return edge_image.astype(np.uint8)

    def show_edge_menu(self):
        edge_menu = tk.Menu(self.root, tearoff=0)
        edge_menu.add_command(label="Sobel Kenar Algılama", command=self.apply_sobel_edge)
        edge_menu.add_command(label="Canny Kenar Algılama", command=self.apply_canny_edge)
        edge_menu.add_command(label="Roberts Kenar Algılama", command=self.apply_roberts_edge)
        try:
            edge_menu.tk_popup(self.btn_edge_detection.winfo_rootx(), self.btn_edge_detection.winfo_rooty() + self.btn_edge_detection.winfo_height())
        finally:
            edge_menu.grab_release()

    def apply_sobel_edge(self):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        edge_img = self.apply_edge_detection(np.array(self.img_gray), sobel_x, sobel_y)
        self.display_image(Image.fromarray(edge_img), self.canvas_processed)

    def apply_canny_edge(self):
        edge_img = cv2.Canny(np.array(self.img_gray), 50, 150)
        self.display_image(Image.fromarray(edge_img), self.canvas_processed)

    def apply_roberts_edge(self):
        edge_img = self.roberts_edge_detection(np.array(self.img_gray))
        self.display_image(Image.fromarray(edge_img), self.canvas_processed)

    def roberts_edge_detection(self, image):
        # Roberts kenar algılama kernel'leri
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        
        # Görüntüyü pad'lemek
        pad_size = 1  # Roberts kernel'leri 2x2 olduğu için padding boyutu 1
        padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
        
        # Boş bir edge görüntüsü oluşturmak
        edge_image = np.zeros_like(image, dtype=np.float32)
        
        # Konvolüsyon işlemini gerçekleştirmek
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i+2, j:j+2]
                gx = np.sum(region * kernel_x)
                gy = np.sum(region * kernel_y)
                edge_image[i, j] = np.sqrt(gx**2 + gy**2)
        
        # Sonuç görüntüsünü 0-255 aralığına normalize etmek
        edge_image = np.clip(edge_image, 0, 255)
        return edge_image.astype(np.uint8)
    
    def show_image_properties(self):
        # Yüklenen görüntünün özelliklerini gösterir
        if self.img:
            img_arr = np.array(self.img)  # Görüntüyü diziye dönüştürür
            dimensions = img_arr.shape  # Görüntünün boyutlarını alır
            min_val = img_arr.min()  # En düşük piksel değerini alır
            max_val = img_arr.max()  # En yüksek piksel değerini alır
            mean_val = img_arr.mean()  # Ortalama piksel değerini alır

            # Yeni bir pencere oluştur ve özellikleri burada göster
            properties_window = Toplevel(self.root)
            properties_window.title("Görüntü Özellikleri")
            properties_window.geometry("300x200")
            properties_window.configure(bg="#f0f0f0")

            properties_text = (f"Boyutlar: {dimensions}\n"
                               f"Minimum Piksel Değeri: {min_val}\n"
                               f"Maksimum Piksel Değeri: {max_val}\n"
                               f"Ortalama Piksel Değeri: {mean_val:.2f}")

            properties_label = tk.Label(properties_window, text=properties_text, bg="#f0f0f0")
            properties_label.pack(padx=10, pady=10)
            
    def convert_to_binary(self):
        # Görüntüyü belirli bir eşik değerine göre ikili (binary) görüntüye dönüştürür
        if self.img_gray:
            threshold = 128  # Sabit bir eşik değeri kullanıyoruz, isteğe göre değiştirilebilir
            img_arr = np.array(self.img_gray)  # Görüntüyü diziye dönüştürür
            binary_img_arr = (img_arr > threshold) * 255  # İkili görüntüyü oluşturur
            self.img_binary = Image.fromarray(binary_img_arr.astype(np.uint8))  # Görüntüyü oluşturur

            # Yeni bir pencere oluştur ve ikili görüntüyü burada göster
            binary_window = Toplevel(self.root)
            binary_window.title("İkili Görüntü")
            binary_window.geometry("300x300")
            binary_window.configure(bg="#f0f0f0")

            binary_canvas = tk.Canvas(binary_window, width=256, height=256, bg="white")
            binary_canvas.pack(padx=10, pady=10)
            self.display_image(self.img_binary, binary_canvas)
    
    
    def show_image_matrix(self):
        # Yüklenen görüntünün matrislerini gösterir
        if self.img:
            img_arr = np.array(self.img)  # Görüntüyü diziye dönüştürür

            # Yeni bir pencere oluştur ve matrisleri burada göster
            matrix_window = Toplevel(self.root)
            matrix_window.title("Görüntü Matrisleri")
            matrix_window.geometry("1200x800")
            matrix_window.configure(bg="#f0f0f0")

            # Başlık etiketini oluştur
            header_label = tk.Label(matrix_window, text="Görüntü Matrisleri", bg="#f0f0f0", font=("Arial", 16, "bold"))
            header_label.grid(row=0, column=0, columnspan=6, pady=10)

            # Kırmızı kanal matrisi
            r_matrix = img_arr[:, :, 0]
            r_label = tk.Label(matrix_window, text="Kırmızı Kanal Matrisi", bg="#f0f0f0", font=("Arial", 14, "bold"))
            r_label.grid(row=1, column=0, padx=10, pady=10)
            r_text = tk.Text(matrix_window, wrap='none', width=40, height=20, font=("Arial", 10))
            r_text.grid(row=2, column=0, padx=10, pady=10)
            r_scroll_x = tk.Scrollbar(matrix_window, orient='horizontal', command=r_text.xview)
            r_scroll_x.grid(row=3, column=0, sticky='ew')
            r_scroll_y = tk.Scrollbar(matrix_window, orient='vertical', command=r_text.yview)
            r_scroll_y.grid(row=2, column=1, sticky='ns')
            r_text.config(xscrollcommand=r_scroll_x.set, yscrollcommand=r_scroll_y.set)
            for row in r_matrix:
                r_text.insert('end', ' '.join(map(str, row)) + '\n')

            # Yeşil kanal matrisi
            g_matrix = img_arr[:, :, 1]
            g_label = tk.Label(matrix_window, text="Yeşil Kanal Matrisi", bg="#f0f0f0", font=("Arial", 14, "bold"))
            g_label.grid(row=1, column=2, padx=10, pady=10)
            g_text = tk.Text(matrix_window, wrap='none', width=40, height=20, font=("Arial", 10))
            g_text.grid(row=2, column=2, padx=10, pady=10)
            g_scroll_x = tk.Scrollbar(matrix_window, orient='horizontal', command=g_text.xview)
            g_scroll_x.grid(row=3, column=2, sticky='ew')
            g_scroll_y = tk.Scrollbar(matrix_window, orient='vertical', command=g_text.yview)
            g_scroll_y.grid(row=2, column=3, sticky='ns')
            g_text.config(xscrollcommand=g_scroll_x.set, yscrollcommand=g_scroll_y.set)
            for row in g_matrix:
                g_text.insert('end', ' '.join(map(str, row)) + '\n')

            # Mavi kanal matrisi
            b_matrix = img_arr[:, :, 2]
            b_label = tk.Label(matrix_window, text="Mavi Kanal Matrisi", bg="#f0f0f0", font=("Arial", 14, "bold"))
            b_label.grid(row=1, column=4, padx=10, pady=10)
            b_text = tk.Text(matrix_window, wrap='none', width=40, height=20, font=("Arial", 10))
            b_text.grid(row=2, column=4, padx=10, pady=10)
            b_scroll_x = tk.Scrollbar(matrix_window, orient='horizontal', command=b_text.xview)
            b_scroll_x.grid(row=3, column=4, sticky='ew')
            b_scroll_y = tk.Scrollbar(matrix_window, orient='vertical', command=b_text.yview)
            b_scroll_y.grid(row=2, column=5, sticky='ns')
            b_text.config(xscrollcommand=b_scroll_x.set, yscrollcommand=b_scroll_y.set)
            for row in b_matrix:
                b_text.insert('end', ' '.join(map(str, row)) + '\n')
                
    def show_morphology_menu(self):
        # Morfolojik işlemler menüsünü gösterir
        morphology_menu = tk.Menu(self.root, tearoff=0)
        morphology_menu.add_command(label="Erozyon", command=self.apply_erosion)
        morphology_menu.add_command(label="Genişleme", command=self.apply_dilation)
        morphology_menu.add_command(label="Açılma", command=self.apply_opening)
        morphology_menu.add_command(label="Kapanma", command=self.apply_closing)
        try:
            morphology_menu.tk_popup(self.btn_morphology.winfo_rootx(), self.btn_morphology.winfo_rooty() + self.btn_morphology.winfo_height())
        finally:
            morphology_menu.grab_release()

    def apply_erosion(self):
        # Erozyon işlemi uygular
        kernel = np.ones((5, 5), np.uint8)  # Erozyon çekirdeğini tanımlar
        eroded_image = self.morphological_operation(np.array(self.img), kernel, operation='erosion')
        self.display_image(Image.fromarray(eroded_image), self.canvas_processed)  # Erozyon uygulanmış görüntüyü gösterir

    def apply_dilation(self):
        # Genişleme işlemi uygular
        kernel = np.ones((5, 5), np.uint8)  # Genişleme çekirdeğini tanımlar
        dilated_image = self.morphological_operation(np.array(self.img), kernel, operation='dilation')
        self.display_image(Image.fromarray(dilated_image), self.canvas_processed)  # Genişleme uygulanmış görüntüyü gösterir

    def apply_opening(self):
        # Açılma işlemi uygular
        kernel = np.ones((5, 5), np.uint8)  # Açılma çekirdeğini tanımlar
        eroded_image = self.morphological_operation(np.array(self.img), kernel, operation='erosion')
        opened_image = self.morphological_operation(eroded_image, kernel, operation='dilation')
        self.display_image(Image.fromarray(opened_image), self.canvas_processed)  # Açılma uygulanmış görüntüyü gösterir

    def apply_closing(self):
        # Kapanma işlemi uygular
        kernel = np.ones((5, 5), np.uint8)  # Kapanma çekirdeğini tanımlar
        dilated_image = self.morphological_operation(np.array(self.img), kernel, operation='dilation')
        closed_image = self.morphological_operation(dilated_image, kernel, operation='erosion')
        self.display_image(Image.fromarray(closed_image), self.canvas_processed)  # Kapanma uygulanmış görüntüyü gösterir

    def morphological_operation(self, image, kernel, operation):
        # Morfolojik işlemleri uygular
        kernel_size = kernel.shape[0]
        pad_size = kernel_size // 2
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
        result_image = np.zeros_like(image)

        for channel in range(3):  # Her renk kanalı için işlemi gerçekleştir
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded_image[i:i+kernel_size, j:j+kernel_size, channel]
                    if operation == 'erosion':
                        result_image[i, j, channel] = np.min(region * kernel)
                    elif operation == 'dilation':
                        result_image[i, j, channel] = np.max(region * kernel)

        return result_image
    
    def region_growing_menu(self):
        # Region Growing için tohum noktayı ve eşik değerini al
        self.region_growing_window = Toplevel(self.root)
        self.region_growing_window.title("Region Growing Parametreleri")
        self.region_growing_window.geometry("300x200")

        tk.Label(self.region_growing_window, text="Tohum Nokta (x,y):").pack(pady=5)
        self.seed_entry = tk.Entry(self.region_growing_window)
        self.seed_entry.pack(pady=5)

        tk.Label(self.region_growing_window, text="Eşik Değeri:").pack(pady=5)
        self.threshold_entry = tk.Entry(self.region_growing_window)
        self.threshold_entry.pack(pady=5)

        tk.Button(self.region_growing_window, text="Uygula", command=self.apply_region_growing).pack(pady=5)

    def apply_region_growing(self):
        # Tohum nokta ve eşik değerini al ve region growing uygula
        try:
            seed_point = tuple(map(int, self.seed_entry.get().split(',')))
            threshold = int(self.threshold_entry.get())

            if self.img_gray:
                segmented_img = self.region_growing(np.array(self.img_gray), seed_point, threshold)
                self.display_image(Image.fromarray(segmented_img), self.canvas_processed)
        except ValueError:
            tk.messagebox.showerror("Hata", "Lütfen doğru formatta tohum nokta (x,y) ve eşik değeri giriniz.")

    def region_growing(self, image, seed, threshold):
        # Region growing algoritmasını uygula
        h, w = image.shape
        segmented = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)
        seed_value = image[seed]
        queue = deque([seed])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4 yönlü komşuluk

        while queue:
            x, y = queue.popleft()
            if visited[x, y]:
                continue
            visited[x, y] = True

            if abs(int(image[x, y]) - int(seed_value)) < threshold:
                segmented[x, y] = 255  # Segment edilen pikseli beyaz yap
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                        queue.append((nx, ny))

        return segmented
    
    def convert_to_rgb(self):
        if self.img:
            img_rgb = self.img.convert("RGB")
            self.display_image(img_rgb, self.canvas_processed)

    def convert_to_gray(self):
        if self.img:
            img_gray = self.img.convert("L")
            self.display_image(img_gray, self.canvas_processed)

    def show_histogram_intensity(self):
        if self.img_gray:
            histogram = self.calculate_histogram(self.img_gray)
            self.display_histogram(histogram, "Yoğunluk Histogramı", color='gray')

    def segment_contour(self):
        segmented_img = self.active_contour_model(np.array(self.img))
        self.display_image(Image.fromarray(segmented_img), self.canvas_processed)

    def active_contour_model(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        ret, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(gray)
        cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
        return contour_img

    def apply_custom_filter_ui(self):
        # Kullanıcıdan matris boyutunu ve matris değerlerini al
        size = simpledialog.askinteger("Matris Boyutu", "Lütfen matris boyutunu girin (örn. 3, 5, 7):")
        if size:
            kernel_values = simpledialog.askstring("Matris Değerleri", f"Lütfen {size}x{size} matris değerlerini virgülle ayırarak girin:")
            if kernel_values:
                try:
                    kernel = np.array([float(x) for x in kernel_values.split(',')]).reshape((size, size))
                    filtered_image = self.apply_custom_filter(self.img, kernel)
                    self.display_image(filtered_image, self.canvas_processed)
                except:
                    messagebox.showerror("Hata", "Lütfen doğru formatta matris değerlerini girin.")

    def apply_custom_filter(self, image, kernel):
        # Özelleştirilmiş filtre uygula
        if image is not None and kernel is not None:
            image_array = np.array(image)
            filtered_array = self.convolve(image_array, kernel)
            filtered_image = Image.fromarray(filtered_array.astype('uint8'))
            return filtered_image
        return image

    def convolve(self, image, kernel):
        # Konvolüsyon işlemi
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        if len(image.shape) == 3:  # RGB görüntüler için
            padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
            result = np.zeros_like(image)
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    for c in range(image.shape[2]):
                        result[y, x, c] = (kernel * padded_image[y:y + kernel_height, x:x + kernel_width, c]).sum()
        else:  # Gri tonlama görüntüler için
            padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
            result = np.zeros_like(image)
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    result[y, x] = (kernel * padded_image[y:y + kernel_height, x:x + kernel_width]).sum()
                    
        return result
    
    def enhance_contrast(self):
        # Görüntüyü numpy dizisine dönüştür
        img_array = np.array(self.img_gray)

        # Minimum ve maksimum piksel değerlerini bul
        min_val = np.min(img_array)
        max_val = np.max(img_array)

        # Kontrast geliştirme işlemi (lineer kontrast genişletme)
        contrast_enhanced = (img_array - min_val) * (255 / (max_val - min_val))
        contrast_enhanced = contrast_enhanced.astype(np.uint8)

        # İşlenmiş görüntüyü göster
        enhanced_img = Image.fromarray(contrast_enhanced)
        self.display_image(enhanced_img, self.canvas_processed)


if __name__ == "__main__":
    root = tk.Tk()  # Ana uygulama penceresi oluşturur
    app = ImageProcessor(root)  # ImageProcessor sınıfından bir örnek oluşturur
    root.mainloop()  # Tkinter ana döngüsünü başlatır
