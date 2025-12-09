import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_gradients(image):
 
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    orientation = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)  # в градусах
    orientation = (orientation + 360) % 360  # нормализация 0-360
    
    return magnitude, orientation


def bilinear_interpolation_gradient(mag_block, ori_block):
 
    # Средний градиент между верхними двумя пикселями
    top_avg_mag = (mag_block[0, 0] + mag_block[0, 1]) / 2
    top_avg_ori = (ori_block[0, 0] + ori_block[0, 1]) / 2
    
    # Средний градиент между нижними двумя пикселями
    bottom_avg_mag = (mag_block[1, 0] + mag_block[1, 1]) / 2
    bottom_avg_ori = (ori_block[1, 0] + ori_block[1, 1]) / 2
    
    # Итоговый усреднённый градиент
    final_mag = (top_avg_mag + bottom_avg_mag) / 2
    final_ori = (top_avg_ori + bottom_avg_ori) / 2
    
    return final_mag, final_ori


def create_descriptor(keypoint, image, grid_size=16, cell_size=1):
  
    x, y = int(keypoint[0]), int(keypoint[1])
    
    # Размер окрестности: grid_size * cell_size + 1
    patch_size = grid_size * cell_size + 1
    
    # Проверка выхода за границы
    half_size = patch_size // 2
    height, width = image.shape
    
    if (x - half_size < 0 or x + half_size >= width or 
        y - half_size < 0 or y + half_size >= height):
        return None
    
    # Вычисляем градиенты для всего изображения
    mag_full, ori_full = compute_gradients(image)
    
    # Извлекаем патч окрестности точки
    patch_mag = mag_full[y-half_size:y+half_size+1, x-half_size:x+half_size+1]
    patch_ori = ori_full[y-half_size:y+half_size+1, x-half_size:x+half_size+1]
    
    descriptor = []
    
    # Проходим по сетке grid_size x grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            # Вычисляем индексы для блока 2x2
            top_left_i = i * cell_size
            top_left_j = j * cell_size
            
            # Извлекаем блок 2x2
            mag_block = patch_mag[top_left_i:top_left_i+2, top_left_j:top_left_j+2]
            ori_block = patch_ori[top_left_i:top_left_i+2, top_left_j:top_left_j+2]
            
            # Билинейная интерполяция
            avg_mag, avg_ori = bilinear_interpolation_gradient(mag_block, ori_block)
            
            # Добавляем в дескриптор (можно квантовать ориентацию)
            # Здесь просто добавляем магнитуду и ориентацию
            descriptor.append(avg_mag)
            descriptor.append(avg_ori)
    
    # Нормализация дескриптора (для устойчивости к изменениям освещения)
    descriptor = np.array(descriptor)
    descriptor_norm = np.linalg.norm(descriptor)
    if descriptor_norm > 0:
        descriptor = descriptor / descriptor_norm
    
    # Ограничиваем максимальные значения (как в SIFT)
    descriptor = np.clip(descriptor, 0, 0.2)
    
    # Повторная нормализация
    descriptor_norm = np.linalg.norm(descriptor)
    if descriptor_norm > 0:
        descriptor = descriptor / descriptor_norm
    
    return descriptor


def create_descriptor_with_histograms(keypoint, image, grid_size=4, cell_size=4, bins=8):
  
    x, y = int(keypoint[0]), int(keypoint[1])
    
    # Размер окрестности: grid_size * cell_size
    patch_size = grid_size * cell_size
    
    half_size = patch_size // 2
    height, width = image.shape
    
    if (x - half_size < 0 or x + half_size >= width or 
        y - half_size < 0 or y + half_size >= height):
        return None
    
    # Вычисляем градиенты
    mag_full, ori_full = compute_gradients(image)
    
    # Извлекаем патч
    patch_mag = mag_full[y-half_size:y+half_size, x-half_size:x+half_size]
    patch_ori = ori_full[y-half_size:y+half_size, x-half_size:x+half_size]
    
    descriptor = []
    
    # Разбиваем на блоки grid_size x grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            # Извлекаем ячейку cell_size x cell_size
            cell_mag = patch_mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ori = patch_ori[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            
            # Создаём гистограмму для ячейки
            hist = np.zeros(bins)
            
            # Проходим по всем пикселям в ячейке
            for m in range(cell_size):
                for n in range(cell_size):
                    mag = cell_mag[m, n]
                    ori = cell_ori[m, n]
                    
                    # Квантование ориентации
                    bin_idx = int(ori / (360 / bins)) % bins
                    hist[bin_idx] += mag
            
            # Добавляем гистограмму в дескриптор
            descriptor.extend(hist)
    
    descriptor = np.array(descriptor)
    
    # Нормализация
    descriptor_norm = np.linalg.norm(descriptor)
    if descriptor_norm > 0:
        descriptor = descriptor / descriptor_norm
    
    # Ограничение и повторная нормализация
    descriptor = np.clip(descriptor, 0, 0.2)
    descriptor_norm = np.linalg.norm(descriptor)
    if descriptor_norm > 0:
        descriptor = descriptor / descriptor_norm
    
    return descriptor


def harris_keypoints(image, max_points=100):
  
    # Параметры детектора Харриса
    block_size = 2
    ksize = 3
    k = 0.04
    
    # Вычисление углов Харриса
    dst = cv2.cornerHarris(image, block_size, ksize, k)
    
    # Нормализация и threshold
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Порог для отбора ключевых точек
    threshold = 0.01 * dst_norm.max()
    keypoints = []
    
    # Находим координаты ключевых точек
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if dst_norm[i, j] > threshold:
                keypoints.append([j, i])  # (x, y)
    
    # Ограничиваем количество точек
    if len(keypoints) > max_points:
        # Сортируем по силе отклика
        strengths = [dst_norm[y, x] for x, y in keypoints]
        sorted_indices = np.argsort(strengths)[::-1]
        keypoints = [keypoints[i] for i in sorted_indices[:max_points]]
    
    return keypoints


def visualize_keypoints_and_descriptors(image, keypoints, descriptors_sample=None):
    """
    Визуализация ключевых точек и примеров дескрипторов
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Исходное изображение с ключевыми точками
    axes[0].imshow(image, cmap='gray')
    axes[0].scatter([kp[0] for kp in keypoints], 
                    [kp[1] for kp in keypoints], 
                    c='red', s=10, marker='o')
    axes[0].set_title(f'Keypoints ({len(keypoints)} points)')
    axes[0].axis('off')
    
    # 2. Пример дескриптора (график)
    if descriptors_sample is not None and len(descriptors_sample) > 0:
        axes[1].plot(descriptors_sample[0])
        axes[1].set_title('Descriptor (first keypoint)')
        axes[1].set_xlabel('Descriptor element')
        axes[1].set_ylabel('Value')
    
    # 3. Визуализация градиентов в окрестности точки
    if len(keypoints) > 0:
        kp = keypoints[0]
        x, y = int(kp[0]), int(kp[1])
        patch_size = 20
        
        if (x - patch_size >= 0 and x + patch_size < image.shape[1] and
            y - patch_size >= 0 and y + patch_size < image.shape[0]):
            
            patch = image[y-patch_size:y+patch_size, x-patch_size:x+patch_size]
            axes[2].imshow(patch, cmap='gray')
            axes[2].set_title(f'Patch around keypoint (size: {patch_size}x{patch_size})')
            axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


# Пример использования
def main():
    # Загрузка тестового изображения
    image = cv2.imread('photo_2025-11-25_08-59-04.jpg', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        # Создаём тестовое изображение, если файла нет
        print("Создаю тестовое изображение...")
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
        cv2.circle(image, (100, 100), 30, 128, -1)
    
    # 1. Детекция ключевых точек Харриса
    keypoints = harris_keypoints(image, max_points=50)
    print(f"Найдено {len(keypoints)} ключевых точек")
    
    # 2. Создание дескрипторов для каждой точки
    descriptors_simple = []
    descriptors_hist = []
    
    for kp in keypoints[:10]:  # Ограничим для примера
        # Простой дескриптор с билинейной интерполяцией
        desc_simple = create_descriptor(kp, image, grid_size=16, cell_size=1)
        if desc_simple is not None:
            descriptors_simple.append(desc_simple)
        
        # Дескриптор с гистограммами
        desc_hist = create_descriptor_with_histograms(kp, image, grid_size=4, cell_size=4, bins=8)
        if desc_hist is not None:
            descriptors_hist.append(desc_hist)
    
    print(f"Создано {len(descriptors_simple)} простых дескрипторов")
    print(f"Создано {len(descriptors_hist)} дескрипторов с гистограммами")
    
    if len(descriptors_simple) > 0:
        print(f"Размер простого дескриптора: {descriptors_simple[0].shape}")
        print(f"Размер дескриптора с гистограммами: {descriptors_hist[0].shape}")
    
    # 3. Визуализация
    visualize_keypoints_and_descriptors(image, keypoints, descriptors_simple)
    
    # 4. Пример сравнения двух дескрипторов
    if len(descriptors_simple) >= 2:
        dist = np.linalg.norm(descriptors_simple[0] - descriptors_simple[1])
        print(f"Расстояние между первыми двумя дескрипторами: {dist:.4f}")


if __name__ == "__main__":
    main()