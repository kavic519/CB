"""
图像检索系统：基于 BoF / VLAD 特征编码 + KNN 检索
步骤：
1. 批量提取 SIFT/ORB 描述子
2. K-Means 生成视觉字典（码本）
3. BoF / VLAD 编码
4. KNN 检索 (K=10)
5. 评估：Recall, Precision, mAP, 检索时间
"""

import os
import cv2
import numpy as np
import pickle
import time
from glob import glob
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


# ===================== 1. 批量特征提取 =====================

def get_image_paths(root_dir):
    """获取目录下所有图像路径，返回 [(label, path), ...]"""
    paths = []
    for label in sorted(os.listdir(root_dir)):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for p in glob(os.path.join(label_dir, ext)):
                paths.append((label, p))
    return paths


def extract_sift_desc(image_path, max_features=500):
    """提取单张图像的 SIFT 描述子"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp, desc = sift.detectAndCompute(img, None)
    return desc  # (N, 128) or None


def extract_orb_desc(image_path, max_features=500):
    """提取单张图像的 ORB 描述子"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    orb = cv2.ORB_create(nfeatures=max_features)
    kp, desc = orb.detectAndCompute(img, None)
    return desc  # (N, 32) or None


def batch_extract_features(image_list, method='SIFT', max_features=500):
    """
    批量提取特征
    :param image_list: [(label, path), ...]
    :return: dict {path: (label, descriptors)}
    """
    features = {}
    extractor = extract_sift_desc if method == 'SIFT' else extract_orb_desc
    total = len(image_list)
    print(f"[{method}] 开始提取 {total} 张图像的特征...")
    t0 = time.time()
    for i, (label, path) in enumerate(image_list):
        desc = extractor(path, max_features)
        if desc is not None and len(desc) > 0:
            features[path] = (label, desc)
        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  进度: {i+1}/{total}")
    print(f"[{method}] 特征提取完成，耗时 {time.time()-t0:.2f}s，成功 {len(features)}/{total}")
    return features


# ===================== 2. 生成视觉字典 =====================

def build_codebook(all_descriptors, n_words=256):
    """
    用 K-Means 生成视觉字典
    :param all_descriptors: 所有描述子拼接后的数组 (M, D)
    :param n_words: 视觉单词数量
    :return: kmeans 模型
    """
    print(f"\n[Codebook] 开始 K-Means 聚类，视觉单词数={n_words}...")
    print(f"[Codebook] 总描述子数量: {all_descriptors.shape}")
    t0 = time.time()
    kmeans = MiniBatchKMeans(n_clusters=n_words, batch_size=10000,
                              random_state=42, n_init=3, max_iter=100)
    kmeans.fit(all_descriptors)
    print(f"[Codebook] K-Means 完成，耗时 {time.time()-t0:.2f}s")
    return kmeans


# ===================== 3. BoF / VLAD 编码 =====================

def encode_bof(descriptors, kmeans):
    """Bag of Features 编码"""
    if descriptors is None or len(descriptors) == 0:
        n_words = kmeans.n_clusters
        return np.zeros(n_words, dtype=np.float32)
    words = kmeans.predict(descriptors)
    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def encode_vlad(descriptors, kmeans):
    """VLAD 编码"""
    n_words = kmeans.n_clusters
    D = kmeans.cluster_centers_.shape[1]
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(n_words * D, dtype=np.float32)

    words = kmeans.predict(descriptors)
    vlad = np.zeros((n_words, D), dtype=np.float32)

    for i in range(n_words):
        mask = (words == i)
        if np.sum(mask) > 0:
            vlad[i] = np.sum(descriptors[mask] - kmeans.cluster_centers_[i], axis=0)

    vlad = vlad.flatten()
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    norm = np.linalg.norm(vlad)
    if norm > 0:
        vlad /= norm
    return vlad


def compute_idf(raw_counts_matrix):
    """
    计算 IDF 权重向量
    :param raw_counts_matrix: (N_images, n_words) 原始频次矩阵
    :return: (n_words,) 的 IDF 向量
    """
    N = raw_counts_matrix.shape[0]
    # 统计每个视觉单词在多少张图像中出现过（频次 > 0 即算出现）
    df = np.sum(raw_counts_matrix > 0, axis=0).astype(np.float32)
    # 加 1 平滑，防止除零
    idf = np.log(N / (df + 1e-8))
    return idf


def encode_bof_tfidf(descriptors, kmeans, idf):
    """
    TF-IDF 加权的 BoF 编码
    公式: h_j = (h_j / sum_i h_i) * log(N / f_j)
    其中 h_j / sum_i h_i 是归一化频次（TF），log(N/f_j) 是 IDF
    :param idf: (n_words,) IDF 权重向量
    """
    n_words = kmeans.n_clusters
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(n_words, dtype=np.float32)

    words = kmeans.predict(descriptors)
    hist, _ = np.histogram(words, bins=np.arange(n_words + 1))
    hist = hist.astype(np.float32)

    total = np.sum(hist)
    if total > 0:
        tf = hist / total  # TF = 归一化频次
        tfidf = tf * idf   # TF-IDF
    else:
        tfidf = np.zeros(n_words, dtype=np.float32)

    return tfidf


def batch_encode_features(features_dict, kmeans, encoding='bof', idf=None):
    """批量编码"""
    if encoding == 'bof':
        encode_fn = lambda desc, km: encode_bof(desc, km)
    elif encoding == 'vlad':
        encode_fn = lambda desc, km: encode_vlad(desc, km)
    elif encoding == 'tfidf':
        if idf is None:
            raise ValueError("TF-IDF 编码需要提供 idf 向量")
        encode_fn = lambda desc, km: encode_bof_tfidf(desc, km, idf)
    else:
        raise ValueError(f"不支持的编码方式: {encoding}")

    codes = []
    labels = []
    paths = []
    print(f"\n[{encoding.upper()}] 开始编码 {len(features_dict)} 张图像...")
    t0 = time.time()
    for path, (label, desc) in features_dict.items():
        code = encode_fn(desc, kmeans)
        codes.append(code)
        labels.append(label)
        paths.append(path)
    codes = np.array(codes, dtype=np.float32)
    print(f"[{encoding.upper()}] 编码完成，耗时 {time.time()-t0:.2f}s，编码维度={codes.shape[1]}")
    return codes, labels, paths


# ===================== 4. KNN 检索 =====================

def build_index(train_codes):
    """构建 KNN 索引"""
    print("\n[Index] 构建 KNN 索引...")
    nn = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='auto')
    nn.fit(train_codes)
    return nn


def knn_search(nn, query_code, train_labels, train_paths, k=10):
    """KNN 检索，返回 top-k 结果"""
    distances, indices = nn.kneighbors(query_code.reshape(1, -1), n_neighbors=k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((train_labels[idx], train_paths[idx], float(dist)))
    return results


# ===================== 5. 评估指标 =====================

def compute_ap(retrieved_labels, query_label):
    """
    计算 Average Precision (AP)
    :param retrieved_labels: 检索返回的 label 列表（已按相关性排序）
    :param query_label: 查询图像的 label
    :return: AP 值
    """
    correct = 0
    precisions = []
    for i, label in enumerate(retrieved_labels):
        if label == query_label:
            correct += 1
            precision_at_k = correct / (i + 1)
            precisions.append(precision_at_k)
    if len(precisions) == 0:
        return 0.0
    return np.mean(precisions)


def evaluate_retrieval(test_codes, test_labels, test_paths, nn_index,
                       train_labels, train_paths, k=10):
    """
    评估检索性能
    :return: dict with metrics
    """
    print(f"\n[Evaluate] 开始评估 {len(test_codes)} 个查询...")
    t0 = time.time()

    aps = []
    precisions_at_k = []
    recalls = []
    query_times = []

    for i, (code, q_label, q_path) in enumerate(zip(test_codes, test_labels, test_paths)):
        t_q = time.time()
        results = knn_search(nn_index, code, train_labels, train_paths, k=k)
        query_times.append((time.time() - t_q) * 1000)  # ms

        retrieved_labels = [r[0] for r in results]

        # AP
        ap = compute_ap(retrieved_labels, q_label)
        aps.append(ap)

        # Precision@K
        relevant_in_topk = sum(1 for l in retrieved_labels if l == q_label)
        precision = relevant_in_topk / k
        precisions_at_k.append(precision)

        # Recall
        # 训练集中该 label 的总数
        total_relevant = sum(1 for l in train_labels if l == q_label)
        if total_relevant > 0:
            recall = relevant_in_topk / total_relevant
        else:
            recall = 0.0
        recalls.append(recall)

    total_time = (time.time() - t0) * 1000

    metrics = {
        'mAP': np.mean(aps),
        'mean_precision@10': np.mean(precisions_at_k),
        'mean_recall@10': np.mean(recalls),
        'total_query_time_ms': total_time,
        'mean_query_time_ms': np.mean(query_times),
        'num_queries': len(test_codes)
    }
    return metrics


# ===================== 主流程 =====================

def run_pipeline(method='SIFT', n_words=256, encoding='bof', k=10):
    """
    运行完整检索流程
    """
    print("=" * 60)
    print(f"配置: method={method}, codebook_size={n_words}, encoding={encoding}, K={k}")
    print("=" * 60)

    # 1. 获取图像路径
    train_list = get_image_paths('image')
    test_list = get_image_paths('test')
    print(f"训练集: {len(train_list)} 张, 测试集: {len(test_list)} 张")

    # 2. 提取特征
    train_features = batch_extract_features(train_list, method=method)
    test_features = batch_extract_features(test_list, method=method)

    # 3. 生成码本（只用训练集）
    all_train_desc = np.vstack([desc for _, desc in train_features.values()])
    kmeans = build_codebook(all_train_desc, n_words=n_words)

    # 保存码本
    codebook_path = f'codebook_{method}_{n_words}.pkl'
    with open(codebook_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"码本已保存: {codebook_path}")

    # 4. 编码
    idf = None
    if encoding == 'tfidf':
        # 先提取原始频次以计算 IDF
        print("\n[TF-IDF] 提取原始频次以计算 IDF...")
        raw_counts_matrix = []
        for path, (label, desc) in train_features.items():
            if desc is not None and len(desc) > 0:
                words = kmeans.predict(desc)
                hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
                raw_counts_matrix.append(hist.astype(np.float32))
            else:
                raw_counts_matrix.append(np.zeros(kmeans.n_clusters, dtype=np.float32))
        raw_counts_matrix = np.array(raw_counts_matrix)
        idf = compute_idf(raw_counts_matrix)
        idf_path = f'idf_{method}_{n_words}.pkl'
        with open(idf_path, 'wb') as f:
            pickle.dump(idf, f)
        print(f"IDF 权重已保存: {idf_path}")

    train_codes, train_labels, train_paths = batch_encode_features(
        train_features, kmeans, encoding=encoding, idf=idf)
    test_codes, test_labels, test_paths = batch_encode_features(
        test_features, kmeans, encoding=encoding, idf=idf)

    # 保存训练集编码
    train_encode_path = f'train_encode_{method}_{n_words}_{encoding}.pkl'
    with open(train_encode_path, 'wb') as f:
        pickle.dump({'codes': train_codes, 'labels': train_labels, 'paths': train_paths}, f)
    print(f"训练集编码已保存: {train_encode_path}")

    # 保存测试集编码
    test_encode_path = f'test_encode_{method}_{n_words}_{encoding}.pkl'
    with open(test_encode_path, 'wb') as f:
        pickle.dump({'codes': test_codes, 'labels': test_labels, 'paths': test_paths}, f)
    print(f"测试集编码已保存: {test_encode_path}")

    # 5. 构建索引并检索
    nn_index = build_index(train_codes)
    metrics = evaluate_retrieval(test_codes, test_labels, test_paths,
                                  nn_index, train_labels, train_paths, k=k)

    # 6. 输出结果
    print("\n" + "=" * 60)
    print("评估结果:")
    print(f"  mAP@10:           {metrics['mAP']:.4f}")
    print(f"  Mean Precision@10: {metrics['mean_precision@10']:.4f}")
    print(f"  Mean Recall@10:    {metrics['mean_recall@10']:.4f}")
    print(f"  总检索时间:        {metrics['total_query_time_ms']:.2f} ms")
    print(f"  平均单次检索时间:  {metrics['mean_query_time_ms']:.2f} ms")
    print(f"  查询数量:          {metrics['num_queries']}")
    print("=" * 60)

    return metrics


def compare_experiments():
    """对比不同配置的组合实验"""
    configs = [
        ('SIFT', 256, 'bof'),
        ('SIFT', 256, 'tfidf'),
        ('SIFT', 256, 'vlad'),
        ('ORB', 256, 'bof'),
        ('ORB', 256, 'tfidf'),
        ('ORB', 256, 'vlad'),
    ]
    results = []
    for method, n_words, encoding in configs:
        metrics = run_pipeline(method=method, n_words=n_words, encoding=encoding, k=10)
        results.append({
            'method': method,
            'n_words': n_words,
            'encoding': encoding,
            **metrics
        })

    print("\n\n" + "=" * 80)
    print("对比实验汇总:")
    print("=" * 80)
    print(f"{'Method':<8} {'Words':<8} {'Encoding':<8} {'mAP':<10} {'Prec@10':<10} {'Recall@10':<12} {'AvgTime(ms)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:<8} {r['n_words']:<8} {r['encoding']:<8} "
              f"{r['mAP']:<10.4f} {r['mean_precision@10']:<10.4f} "
              f"{r['mean_recall@10']:<12.4f} {r['mean_query_time_ms']:<12.4f}")
    print("=" * 80)
    return results


# ===================== GUI 友好接口 =====================

def get_cache_paths(method, n_words, encoding):
    """获取缓存文件路径"""
    codebook_path = f'codebook_{method}_{n_words}.pkl'
    train_encode_path = f'train_encode_{method}_{n_words}_{encoding}.pkl'
    test_encode_path = f'test_encode_{method}_{n_words}_{encoding}.pkl'
    return codebook_path, train_encode_path, test_encode_path


def check_cache_exists(method, n_words, encoding):
    """检查码本和训练集编码是否存在"""
    cb_path, train_path, _ = get_cache_paths(method, n_words, encoding)
    return os.path.exists(cb_path) and os.path.exists(train_path)


def build_codebook_and_encode(method='SIFT', n_words=256, encoding='bof',
                               progress_callback=None):
    """
    构建码本并对训练集编码（GUI 用）
    :param progress_callback: fn(stage, current, total) stage: 'extract'/'kmeans'/'encode'
    :return: True/False
    """
    try:
        train_list = get_image_paths('image')
        total = len(train_list)

        # 1. 提取特征
        features = {}
        extractor = extract_sift_desc if method == 'SIFT' else extract_orb_desc
        for i, (label, path) in enumerate(train_list):
            desc = extractor(path)
            if desc is not None and len(desc) > 0:
                features[path] = (label, desc)
            if progress_callback:
                progress_callback('extract', i + 1, total)

        # 2. K-Means
        all_desc = np.vstack([desc for _, desc in features.values()])
        if progress_callback:
            progress_callback('kmeans', 0, 1)
        kmeans = build_codebook(all_desc, n_words=n_words)
        if progress_callback:
            progress_callback('kmeans', 1, 1)

        # 保存码本
        cb_path, _, _ = get_cache_paths(method, n_words, encoding)
        with open(cb_path, 'wb') as f:
            pickle.dump(kmeans, f)

        # 3. 计算 IDF（如果需要）
        idf = None
        if encoding == 'tfidf':
            if progress_callback:
                progress_callback('idf', 0, 1)
            raw_counts_matrix = []
            for path, (label, desc) in features.items():
                if desc is not None and len(desc) > 0:
                    words = kmeans.predict(desc)
                    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
                    raw_counts_matrix.append(hist.astype(np.float32))
                else:
                    raw_counts_matrix.append(np.zeros(kmeans.n_clusters, dtype=np.float32))
            raw_counts_matrix = np.array(raw_counts_matrix)
            idf = compute_idf(raw_counts_matrix)
            idf_path = f'idf_{method}_{n_words}.pkl'
            with open(idf_path, 'wb') as f:
                pickle.dump(idf, f)
            if progress_callback:
                progress_callback('idf', 1, 1)

        # 4. 编码训练集
        train_codes, train_labels, train_paths = batch_encode_features(
            features, kmeans, encoding=encoding, idf=idf)
        _, train_encode_path, _ = get_cache_paths(method, n_words, encoding)
        with open(train_encode_path, 'wb') as f:
            pickle.dump({'codes': train_codes, 'labels': train_labels, 'paths': train_paths}, f)

        return True
    except Exception as err:
        print(f"[build_codebook_and_encode] 错误: {err}")
        return False


def load_train_index(method='SIFT', n_words=256, encoding='bof'):
    """
    加载训练集编码并构建 KNN 索引
    :return: (nn_index, train_codes, train_labels, train_paths) or None
    """
    _, train_encode_path, _ = get_cache_paths(method, n_words, encoding)
    if not os.path.exists(train_encode_path):
        return None

    with open(train_encode_path, 'rb') as f:
        data = pickle.load(f)
    train_codes = data['codes']
    train_labels = data['labels']
    train_paths = data['paths']

    nn_index = build_index(train_codes)
    return nn_index, train_codes, train_labels, train_paths


def load_codebook(method='SIFT', n_words=256):
    """加载码本"""
    cb_path, _, _ = get_cache_paths(method, n_words, 'bof')
    if not os.path.exists(cb_path):
        return None
    with open(cb_path, 'rb') as f:
        return pickle.load(f)


def encode_single_image(image_path, method='SIFT', n_words=256, encoding='bof'):
    """
    对单张查询图像进行编码
    :return: (query_code, query_label) or (None, None)
    """
    cb_path, _, _ = get_cache_paths(method, n_words, encoding)
    if not os.path.exists(cb_path):
        return None, None

    with open(cb_path, 'rb') as f:
        kmeans = pickle.load(f)

    extractor = extract_sift_desc if method == 'SIFT' else extract_orb_desc
    desc = extractor(image_path)
    if desc is None or len(desc) == 0:
        return None, None

    if encoding == 'tfidf':
        idf_path = f'idf_{method}_{n_words}.pkl'
        if not os.path.exists(idf_path):
            return None, None
        with open(idf_path, 'rb') as f:
            idf = pickle.load(f)
        code = encode_bof_tfidf(desc, kmeans, idf)
    elif encoding == 'vlad':
        code = encode_vlad(desc, kmeans)
    else:
        code = encode_bof(desc, kmeans)

    # 从路径推断 label
    label = os.path.basename(os.path.dirname(image_path))
    return code, label


def knn_search_single(query_code, nn_index, train_labels, train_paths, k=10):
    """单张图像 KNN 检索"""
    distances, indices = nn_index.kneighbors(query_code.reshape(1, -1), n_neighbors=k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            'label': train_labels[idx],
            'path': train_paths[idx],
            'distance': float(dist)
        })
    return results


if __name__ == '__main__':
    # 运行单次实验
    # run_pipeline(method='SIFT', n_words=256, encoding='bof', k=10)

    # 运行对比实验
    compare_experiments()
