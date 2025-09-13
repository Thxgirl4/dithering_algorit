from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("arte.webp")
img_original = img.copy()

if img.mode == 'L':
    arr = np.array(img, dtype=np.float32)
    height, width = arr.shape

    burkes_matrix = {
        (0, 1): 8 / 32,
        (0, 2): 4 / 32,
        (1, -2): 2 / 32,
        (1, -1): 4 / 32,
        (1, 0): 8 / 32,
        (1, 1): 4 / 32,
        (1, 2): 2 / 32,
    } #  kernel de difusão de erro ref: 

    colormap = np.array([0, 255], dtype=np.float32)

    arr_dith = arr.copy()

    for y in range(height):
        for x in range(width):
            old_pixel = arr_dith[y, x]

            new_pixel = 255 if old_pixel > 128 else 0

            quant_err = old_pixel - new_pixel

            arr_dith[y, x] = new_pixel

            for (dy, dx), weight in burkes_matrix.items():
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    arr_dith[ny, nx] += quant_err * weight

    final_img = Image.fromarray(arr_dith.astype(np.uint8)) 
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_original, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(final_img, cmap='gray')
    axs[1].set_title('Dithering Burkes')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

elif img.mode == 'RGB':
    arr = np.array(img, dtype=np.float32)
    height, width, _ = arr.shape

    burkes_matrix = {
        (0, 1): 8 / 32,
        (0, 2): 4 / 32,
        (1, -2): 2 / 32,
        (1, -1): 4 / 32,
        (1, 0): 8 / 32,
        (1, 1): 4 / 32,
        (1, 2): 2 / 32,
    } # kernel de difusao ref: 

    colormap = np.array([
        [0, 0, 0],       # Preto
        [255, 0, 0],     # Vermelho
        [0, 255, 0],     # Verde
        [0, 0, 255],    
        [255, 255, 0],   
        [255, 0, 255],   
        [0, 255, 255],   
        [255, 255, 255]  
    ], dtype=np.float32)

    def find_closest_color(pixel):
        distances = np.sum((colormap - pixel) ** 2, axis=1)
        return colormap[np.argmin(distances)]

    arr_dith = arr.copy()

    for y in range(height):
            for x in range(width):
                old_pixel = arr_dith[y, x]
                new_pixel = find_closest_color(old_pixel)
                quant_err = old_pixel - new_pixel
                arr_dith[y, x] = new_pixel
                for (dy, dx), weight in burkes_matrix.items():
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        arr_dith[ny, nx] += quant_err * weight
        
    final_img = np.clip(arr_dith, 0, 255).astype(np.uint8)  

    img_grayscale = img.convert('L')
    arr_grayscale = np.array(img_grayscale, dtype=np.float32)

    burkes_matrix_bw = {
        (0, 1): 8 / 32,
        (0, 2): 4 / 32,
        (1, -2): 2 / 32,
        (1, -1): 4 / 32,
        (1, 0): 8 / 32,
        (1, 1): 4 / 32,
        (1, 2): 2 / 32,
    }
    
    arr_dith_bw = arr_grayscale.copy()
    height_bw, width_bw = arr_dith_bw.shape

    for y in range(height_bw):
        for x in range(width_bw):
            old_pixel = arr_dith_bw[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            quant_err = old_pixel - new_pixel
            arr_dith_bw[y, x] = new_pixel
            for (dy, dx), weight in burkes_matrix_bw.items():
                ny, nx = y + dy, x + dx
                if 0 <= ny < height_bw and 0 <= nx < width_bw:
                    arr_dith_bw[ny, nx] += quant_err * weight

    final_img_bw = arr_dith_bw.astype(np.uint8) 
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_original)
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(final_img_bw, cmap='gray')
    axs[1].set_title('Dithering Burkes')
    axs[1].axis('off')

    axs[2].imshow(final_img)
    axs[2].set_title('Dithering Burkes 8 cores')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()        