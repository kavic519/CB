"""
车辆图像匹配核心模块（实时计算模式）
支持 SIFT 和 ORB 特征提取与匹配
"""

import cv2
import numpy as np
import os
import time
from glob import glob


def get_train_image_paths(train_dir='image', max_per_plate=None):
    """
    获取训练集所有图像路径
    :param train_dir: 训练集根目录
    :param max_per_plate: 每个车牌最多取几张，None 表示全取
    :return: [(plate_id, image_path), ...]
    """
    paths = []
    if not os.path.exists(train_dir):
        return paths
    for plate_dir in sorted(os.listdir(train_dir)):
        plate_path = os.path.join(train_dir, plate_dir)
        if not os.path.isdir(plate_path):
            continue
        images = sorted(glob(os.path.join(plate_path, '*.jpg')) +
                        glob(os.path.join(plate_path, '*.png')) +
                        glob(os.path.join(plate_path, '*.jpeg')))
        if max_per_plate:
            images = images[:max_per_plate]
        for img_path in images:
            paths.append((plate_dir, img_path))
    return paths


def create_detector(method='SIFT'):
    """创建特征检测器"""
    method = method.upper()
    if method == 'SIFT':
        return cv2.SIFT_create()
    elif method == 'ORB':
        return cv2.ORB_create(nfeatures=500)
    else:
        raise ValueError(f"不支持的方法: {method}")


def extract_features(image_path, method='SIFT'):
    """
    对单张图像提取特征
    :return: (keypoints, descriptors, grayscale_image)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    detector = create_detector(method)
    kp, desc = detector.detectAndCompute(img, None)
    return kp, desc, img


def match_features(query_desc, train_desc, method='SIFT', ratio_thresh=0.75):
    """
    使用 BFMatcher + Lowe's ratio test 计算好的匹配点数量
    :return: good_matches_count
    """
    if query_desc is None or train_desc is None:
        return 0
    if len(query_desc) < 2 or len(train_desc) < 2:
        return 0

    method = method.upper()
    if method == 'SIFT':
        norm = cv2.NORM_L2
    else:
        norm = cv2.NORM_HAMMING

    bf = cv2.BFMatcher(norm)
    try:
        matches = bf.knnMatch(query_desc, train_desc, k=2)
    except cv2.error:
        return 0

    good = 0
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good += 1
        elif len(m_n) == 1:
            good += 1
    return good


def draw_matches(query_path, train_path, method='SIFT', max_matches=50):
    """
    绘制匹配结果图
    :return: BGR 图像数组
    """
    q_kp, q_desc, q_img = extract_features(query_path, method)
    t_kp, t_desc, t_img = extract_features(train_path, method)

    if q_desc is None or t_desc is None or len(q_desc) < 2 or len(t_desc) < 2:
        # 无法匹配，返回拼接图
        if q_img is not None and t_img is not None:
            h = max(q_img.shape[0], t_img.shape[0])
            w = q_img.shape[1] + t_img.shape[1]
            canvas = np.zeros((h, w), dtype=np.uint8)
            canvas[:q_img.shape[0], :q_img.shape[1]] = q_img
            canvas[:t_img.shape[0], q_img.shape[1]:] = t_img
            return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        return None

    method = method.upper()
    norm = cv2.NORM_L2 if method == 'SIFT' else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)
    try:
        matches = bf.knnMatch(q_desc, t_desc, k=2)
    except cv2.error:
        return None

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        elif len(m_n) == 1:
            good_matches.append(m_n[0])

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]

    img_matches = cv2.drawMatches(
        q_img, q_kp, t_img, t_kp, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return img_matches


def find_best_match(query_image_path, method='SIFT', train_dir='image',
                    max_per_plate=None, progress_callback=None):
    """
    在训练集中查找最佳匹配图像（实时计算模式）
    :param query_image_path: 待匹配图像路径
    :param method: 'SIFT' 或 'ORB'
    :param train_dir: 训练集目录
    :param max_per_plate: 每个车牌最多取几张，None 表示全取
    :param progress_callback: 进度回调函数(current, total)
    :return: dict {
        'best_plate': 最佳匹配车牌号,
        'best_path': 最佳匹配图像路径,
        'best_score': 匹配点数,
        'elapsed_ms': 总耗时毫秒,
        'searched': 搜索的图像数量,
        'top_results': [(plate, path, score), ...] 前N个结果
    }
    """
    start = time.perf_counter()

    # 1. 提取查询图像特征
    q_kp, q_desc, _ = extract_features(query_image_path, method)
    if q_desc is None:
        return {
            'best_plate': None,
            'best_path': None,
            'best_score': 0,
            'elapsed_ms': 0,
            'searched': 0,
            'top_results': []
        }

    # 2. 获取训练集图像
    train_paths = get_train_image_paths(train_dir, max_per_plate)
    total = len(train_paths)

    # 3. 遍历匹配
    results = []
    for idx, (plate_id, img_path) in enumerate(train_paths):
        _, t_desc, _ = extract_features(img_path, method)
        score = match_features(q_desc, t_desc, method)
        results.append((plate_id, img_path, score))
        if progress_callback:
            progress_callback(idx + 1, total)

    # 4. 排序取最佳
    results.sort(key=lambda x: x[2], reverse=True)
    best = results[0] if results else (None, None, 0)

    elapsed = (time.perf_counter() - start) * 1000

    return {
        'best_plate': best[0],
        'best_path': best[1],
        'best_score': best[2],
        'elapsed_ms': elapsed,
        'searched': total,
        'top_results': results[:5]
    }


if __name__ == '__main__':
    # 简单测试
    test_dir = 'test'
    if os.path.exists(test_dir):
        plates = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        if plates:
            sample_plate = plates[0]
            sample_images = glob(os.path.join(test_dir, sample_plate, '*.jpg'))
            if sample_images:
                query = sample_images[0]
                print(f"测试查询图像: {query}")
                for m in ['SIFT', 'ORB']:
                    res = find_best_match(query, method=m, max_per_plate=3)
                    print(f"[{m}] 耗时: {res['elapsed_ms']:.1f}ms, 匹配到: {res['best_plate']}, 点数: {res['best_score']}")
