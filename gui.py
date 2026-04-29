"""
车辆图像匹配 GUI
支持模式:
  1. 原始特征匹配 (SIFT/ORB 逐张 BFMatcher)
  2. BoF 编码 + KNN 检索
  3. VLAD 编码 + KNN 检索
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
import time
import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from matcher import find_best_match, draw_matches
from image_retrieval import (
    check_cache_exists, build_codebook_and_encode,
    load_train_index, encode_single_image, knn_search_single
)


class VehicleMatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("车辆图像检索系统 — SIFT / ORB + BoF / VLAD")
        self.root.geometry("1400x950")
        self.root.configure(bg="#f5f5f5")

        self.query_image_path = None
        self.query_photo = None
        self.result_photo = None
        self.topk_photo_refs = []   # 防止缩略图被垃圾回收
        self.topk_labels = []       # Top-K 图像 Label 控件

        # 缓存加载的索引
        self._cached_index_key = None
        self._cached_index = None
        # 保存最近一次检索结果用于PR曲线
        self._last_results = None
        self._last_query_label = None
        self._last_total_relevant = 0

        self._build_ui()
        self._update_codebook_status()

    # ===================== UI 构建 =====================

    def _build_ui(self):
        # ===== 顶部控制区 =====
        control_frame = tk.Frame(self.root, bg="#e0e0e0", padx=10, pady=10)
        control_frame.pack(fill=tk.X, side=tk.TOP)

        # --- 特征算法 ---
        tk.Label(control_frame, text="特征算法:", bg="#e0e0e0",
                 font=("Microsoft YaHei", 11)).pack(side=tk.LEFT, padx=5)
        self.method_var = tk.StringVar(value="SIFT")
        tk.Radiobutton(control_frame, text="SIFT", variable=self.method_var,
                       value="SIFT", bg="#e0e0e0", font=("Microsoft YaHei", 10),
                       command=self._update_codebook_status).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(control_frame, text="ORB", variable=self.method_var,
                       value="ORB", bg="#e0e0e0", font=("Microsoft YaHei", 10),
                       command=self._update_codebook_status).pack(side=tk.LEFT, padx=2)

        tk.Frame(control_frame, bg="#e0e0e0", width=15).pack(side=tk.LEFT)

        # --- 编码方式 ---
        tk.Label(control_frame, text="编码方式:", bg="#e0e0e0",
                 font=("Microsoft YaHei", 11)).pack(side=tk.LEFT, padx=5)
        self.encoding_var = tk.StringVar(value="none")
        tk.Radiobutton(control_frame, text="原始匹配", variable=self.encoding_var,
                       value="none", bg="#e0e0e0", font=("Microsoft YaHei", 10),
                       command=self._update_codebook_status).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(control_frame, text="BoF", variable=self.encoding_var,
                       value="bof", bg="#e0e0e0", font=("Microsoft YaHei", 10),
                       command=self._update_codebook_status).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(control_frame, text="VLAD", variable=self.encoding_var,
                       value="vlad", bg="#e0e0e0", font=("Microsoft YaHei", 10),
                       command=self._update_codebook_status).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(control_frame, text="BoF+TF-IDF", variable=self.encoding_var,
                       value="tfidf", bg="#e0e0e0", font=("Microsoft YaHei", 10),
                       command=self._update_codebook_status).pack(side=tk.LEFT, padx=2)

        tk.Frame(control_frame, bg="#e0e0e0", width=15).pack(side=tk.LEFT)

        # --- K 值 ---
        tk.Label(control_frame, text="K:", bg="#e0e0e0",
                 font=("Microsoft YaHei", 11)).pack(side=tk.LEFT, padx=5)
        self.k_var = tk.Spinbox(control_frame, from_=1, to=50, width=5,
                                font=("Microsoft YaHei", 10))
        self.k_var.delete(0, tk.END)
        self.k_var.insert(0, "10")
        self.k_var.pack(side=tk.LEFT, padx=2)

        tk.Frame(control_frame, bg="#e0e0e0", width=15).pack(side=tk.LEFT)

        # --- 码本状态 ---
        self.status_label = tk.Label(control_frame, text="码本: 检查中...",
                                     bg="#e0e0e0", font=("Microsoft YaHei", 10),
                                     fg="#666666")
        self.status_label.pack(side=tk.LEFT, padx=5)

        tk.Frame(control_frame, bg="#e0e0e0", width=15).pack(side=tk.LEFT)

        # --- 按钮 ---
        btn_font = ("Microsoft YaHei", 10, "bold")
        tk.Button(control_frame, text="选择图像", command=self._select_image,
                  font=btn_font, bg="#2196F3", fg="white", padx=10).pack(side=tk.LEFT, padx=5)
        self.match_btn = tk.Button(control_frame, text="开始检索", command=self._start_match,
                                   font=btn_font, bg="#4CAF50", fg="white", padx=10,
                                   state=tk.DISABLED)
        self.match_btn.pack(side=tk.LEFT, padx=5)
        self.build_btn = tk.Button(control_frame, text="构建码本", command=self._build_codebook,
                                   font=btn_font, bg="#FF9800", fg="white", padx=10,
                                   state=tk.DISABLED)
        self.build_btn.pack(side=tk.LEFT, padx=5)
        self.pr_btn = tk.Button(control_frame, text="绘制PR曲线", command=self._draw_pr_curve,
                                font=btn_font, bg="#9C27B0", fg="white", padx=10,
                                state=tk.DISABLED)
        self.pr_btn.pack(side=tk.LEFT, padx=5)
        self.hist_btn = tk.Button(control_frame, text="编码直方图", command=self._draw_histograms,
                                  font=btn_font, bg="#00BCD4", fg="white", padx=10,
                                  state=tk.DISABLED)
        self.hist_btn.pack(side=tk.LEFT, padx=5)
        self.avg_pr_btn = tk.Button(control_frame, text="平均PR曲线", command=self._draw_avg_pr_curve,
                                    font=btn_font, bg="#E91E63", fg="white", padx=10)
        self.avg_pr_btn.pack(side=tk.LEFT, padx=5)
        self.idf_cmp_btn = tk.Button(control_frame, text="IDF对比", command=self._draw_idf_compare,
                                     font=btn_font, bg="#795548", fg="white", padx=10)
        self.idf_cmp_btn.pack(side=tk.LEFT, padx=5)

        # ===== 图像展示区 =====
        image_frame = tk.Frame(self.root, bg="#f5f5f5", height=294)
        image_frame.pack(fill=tk.X, padx=10, pady=10)
        image_frame.pack_propagate(False)

        # 左侧：查询图像
        left_frame = tk.LabelFrame(image_frame, text="待匹配图像",
                                   font=("Microsoft YaHei", 12), bg="#ffffff", padx=5, pady=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.query_label = tk.Label(left_frame, bg="#eeeeee", text="未选择图像",
                                    font=("Microsoft YaHei", 12))
        self.query_label.pack(fill=tk.BOTH, expand=True)
        self.query_info_label = tk.Label(left_frame, text="", bg="#ffffff",
                                         font=("Microsoft YaHei", 10))
        self.query_info_label.pack(fill=tk.X, pady=(5, 0))

        # 右侧：最佳匹配
        right_frame = tk.LabelFrame(image_frame, text="最佳匹配结果 (Top-1)",
                                    font=("Microsoft YaHei", 12), bg="#ffffff", padx=5, pady=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.result_label = tk.Label(right_frame, bg="#eeeeee", text="等待检索...",
                                     font=("Microsoft YaHei", 12))
        self.result_label.pack(fill=tk.BOTH, expand=True)
        self.result_info_label = tk.Label(right_frame, text="", bg="#ffffff",
                                          font=("Microsoft YaHei", 10))
        self.result_info_label.pack(fill=tk.X, pady=(5, 0))

        # ===== Top-10 图像缩略图展示区 =====
        topk_frame = tk.LabelFrame(self.root, text="Top-10 检索结果图像",
                                   font=("Microsoft YaHei", 12), bg="#f5f5f5", padx=5, pady=5)
        topk_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 5))
        topk_frame.pack_propagate(False)
        topk_frame.config(height=250)

        # Canvas + Frame 实现横向滚动
        self.topk_canvas = tk.Canvas(topk_frame, bg="#f5f5f5", highlightthickness=0)
        hbar = tk.Scrollbar(topk_frame, orient=tk.HORIZONTAL, command=self.topk_canvas.xview)
        self.topk_canvas.configure(xscrollcommand=hbar.set)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.topk_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.topk_container = tk.Frame(self.topk_canvas, bg="#f5f5f5")
        self.topk_canvas.create_window((0, 0), window=self.topk_container, anchor="nw")
        self.topk_container.bind("<Configure>", lambda e: self.topk_canvas.configure(
            scrollregion=self.topk_canvas.bbox("all")))

        # 创建 10 个缩略图槽位（固定像素大小）
        for i in range(10):
            slot = tk.Frame(self.topk_container, bg="#f5f5f5", padx=5, pady=2,
                            width=128, height=170)
            slot.pack(side=tk.LEFT, fill=tk.Y)
            slot.pack_propagate(False)

            # 图像区域（固定 118x118）
            img_frame = tk.Frame(slot, bg="#dddddd", width=118, height=118)
            img_frame.pack(side=tk.TOP, pady=(2, 0))
            img_frame.pack_propagate(False)
            lbl = tk.Label(img_frame, bg="#dddddd", text=f"#{i+1}",
                           font=("Microsoft YaHei", 10, "bold"), fg="#888888")
            lbl.pack(fill=tk.BOTH, expand=True)

            # 信息文字
            info = tk.Label(slot, text="", bg="#f5f5f5",
                            font=("Microsoft YaHei", 8), wraplength=118)
            info.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
            self.topk_labels.append((lbl, info))

        # ===== 结果表格区 =====
        table_frame = tk.LabelFrame(self.root, text="Top-K 检索结果",
                                    font=("Microsoft YaHei", 12), bg="#f5f5f5", padx=10, pady=5)
        table_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))

        cols = ("rank", "plate", "distance", "path")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=6)
        self.tree.heading("rank", text="排名")
        self.tree.heading("plate", text="车牌号")
        self.tree.heading("distance", text="距离")
        self.tree.heading("path", text="文件路径")
        self.tree.column("rank", width=50, anchor="center")
        self.tree.column("plate", width=100, anchor="center")
        self.tree.column("distance", width=100, anchor="center")
        self.tree.column("path", width=800, anchor="w")
        self.tree.pack(fill=tk.X)

        # 统计信息
        self.stats_label = tk.Label(table_frame, text="", bg="#f5f5f5",
                                    font=("Microsoft YaHei", 11, "bold"), fg="#333333")
        self.stats_label.pack(pady=(5, 0))

        # 进度条
        self.progress = ttk.Progressbar(table_frame, orient=tk.HORIZONTAL,
                                        mode='determinate', length=400)
        self.progress.pack(pady=5)
        self.progress_label = tk.Label(table_frame, text="", bg="#f5f5f5",
                                       font=("Microsoft YaHei", 10))
        self.progress_label.pack()

    # ===================== 事件处理 =====================

    def _update_codebook_status(self):
        """更新码本状态显示"""
        encoding = self.encoding_var.get()
        method = self.method_var.get()

        if encoding == "none":
            self.status_label.config(text="码本: 不使用编码", fg="#666666")
            self.build_btn.config(state=tk.DISABLED)
            return

        exists = check_cache_exists(method, 256, encoding)
        if exists:
            self.status_label.config(text=f"码本: {method}+{encoding.upper()} 已就绪", fg="green")
            self.build_btn.config(state=tk.DISABLED)
        else:
            self.status_label.config(text=f"码本: {method}+{encoding.upper()} 未构建", fg="red")
            self.build_btn.config(state=tk.NORMAL)

    def _select_image(self):
        initial_dir = os.path.abspath("test") if os.path.exists("test") else "."
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="选择待匹配的车辆图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self.query_image_path = path
        self._show_image(path, self.query_label, max_size=(406, 350))

        plate_id = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)
        self.query_info_label.config(text=f"车牌: {plate_id}  |  文件名: {fname}")
        self.match_btn.config(state=tk.NORMAL)
        self.pr_btn.config(state=tk.NORMAL)
        self.hist_btn.config(state=tk.NORMAL)
        self.result_label.config(image='', text="等待检索...")
        self.result_info_label.config(text="")
        self.stats_label.config(text="")
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._clear_topk_images()

    def _show_image(self, path_or_array, label, max_size=(406, 350)):
        try:
            if isinstance(path_or_array, str):
                img = Image.open(path_or_array)
            else:
                arr = cv2.cvtColor(path_or_array, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(arr)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label.config(image=photo, text="")
            label.image = photo
            return photo
        except Exception as e:
            label.config(text=f"无法显示图像: {e}")
            return None

    def _start_match(self):
        if not self.query_image_path:
            messagebox.showwarning("提示", "请先选择待匹配图像")
            return

        encoding = self.encoding_var.get()
        method = self.method_var.get()

        # 检查编码模式是否需要码本
        if encoding != "none":
            if not check_cache_exists(method, 256, encoding):
                messagebox.showwarning("提示", f"码本未构建，请先点击【构建码本】按钮")
                return

        self.match_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.progress_label.config(text="正在检索，请稍候...")
        self.stats_label.config(text="")
        self.result_label.config(image='', text="检索中...")
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            k = int(self.k_var.get())
        except ValueError:
            k = 10

        thread = threading.Thread(target=self._match_worker,
                                  args=(method, encoding, k), daemon=True)
        thread.start()

    def _match_worker(self, method, encoding, k):
        try:
            if encoding == "none":
                self._do_raw_match(method, k)
            else:
                self._do_encoded_match(method, encoding, k)
        except Exception as err:
            msg = str(err)
            self.root.after(0, lambda: self._on_error(msg))

    # ===================== 原始匹配 =====================

    def _do_raw_match(self, method, k):
        """原始特征匹配（使用 matcher.py）"""
        max_per = None  # 编码模式用全部

        def progress_cb(current, total):
            pct = int(current / total * 100)
            self.root.after(0, lambda: self._update_progress(pct, current, total))

        result = find_best_match(
            self.query_image_path,
            method=method,
            max_per_plate=max_per,
            progress_callback=progress_cb
        )
        self.root.after(0, lambda: self._on_raw_match_done(result, method))

    def _on_raw_match_done(self, result, method):
        self.progress['value'] = 100
        self.progress_label.config(text="检索完成")

        best_path = result['best_path']
        best_plate = result['best_plate']
        best_score = result['best_score']
        elapsed = result['elapsed_ms']
        searched = result['searched']
        top_results = result.get('top_results', [])

        # 显示 Top-1 图像和匹配连线
        if best_path and os.path.exists(best_path):
            match_img = draw_matches(self.query_image_path, best_path, method)
            if match_img is not None:
                self._show_image(match_img, self.result_label, max_size=(406, 350))
            else:
                self._show_image(best_path, self.result_label, max_size=(406, 350))
            self.result_info_label.config(
                text=f"车牌: {best_plate}  |  文件: {os.path.basename(best_path)}"
            )
            # 填入表格和 Top-K 缩略图
            self.tree.insert("", tk.END, values=("1", best_plate, f"{best_score}pts", best_path))
            self._show_topk_images([{'label': best_plate, 'path': best_path, 'distance': best_score}])
        else:
            self.result_label.config(image='', text="未找到匹配结果")
            self.result_info_label.config(text="")

        # 计算 Precision / Recall / AP（基于 top_results 前10个）
        query_label = os.path.basename(os.path.dirname(self.query_image_path))
        k = min(10, len(top_results))
        topk = top_results[:k]
        correct_count = sum(1 for r in topk if r[0] == query_label)
        precision = correct_count / k if k > 0 else 0

        # 统计训练集中该车牌总数（从 image 目录统计）
        train_plate_dir = os.path.join('image', query_label)
        total_relevant = len([f for f in os.listdir(train_plate_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(train_plate_dir) else 0
        recall = correct_count / total_relevant if total_relevant > 0 else 0

        ap_hits = 0
        ap_precisions = []
        for i, r in enumerate(topk):
            if r[0] == query_label:
                ap_hits += 1
                ap_precisions.append(ap_hits / (i + 1))
        ap = sum(ap_precisions) / len(ap_precisions) if ap_precisions else 0.0

        self.stats_label.config(
            text=(f"模式: 原始{method}匹配  |  搜索图像: {searched}  |  "
                  f"匹配点数: {best_score}  |  "
                  f"Precision@{k}: {precision:.2%}  |  "
                  f"Recall@{k}: {recall:.2%}  |  "
                  f"AP: {ap:.4f}  |  耗时: {elapsed:.1f} ms")
        )
        self.match_btn.config(state=tk.NORMAL)

    # ===================== 编码匹配 =====================

    def _do_encoded_match(self, method, encoding, k):
        """编码 + KNN 检索"""
        n_words = 256

        # 加载索引（带缓存）
        cache_key = f"{method}_{n_words}_{encoding}"
        if self._cached_index_key != cache_key or self._cached_index is None:
            self.root.after(0, lambda: self.progress_label.config(text="加载索引中..."))
            index_data = load_train_index(method, n_words, encoding)
            if index_data is None:
                raise RuntimeError("无法加载训练集索引")
            self._cached_index_key = cache_key
            self._cached_index = index_data

        nn_index, train_codes, train_labels, train_paths = self._cached_index

        # 编码查询图像
        self.root.after(0, lambda: self.progress_label.config(text="编码查询图像..."))
        query_code, query_label = encode_single_image(
            self.query_image_path, method, n_words, encoding
        )
        if query_code is None:
            raise RuntimeError("查询图像编码失败")

        # KNN 检索
        self.root.after(0, lambda: self.progress_label.config(text="KNN 检索中..."))
        t0 = time.perf_counter()
        results = knn_search_single(query_code, nn_index, train_labels, train_paths, k=k)
        elapsed = (time.perf_counter() - t0) * 1000

        # 统计训练集中该车牌的总数（用于计算 Recall）
        total_relevant = sum(1 for l in train_labels if l == query_label)

        # 保存结果供PR曲线使用
        self._last_results = results
        self._last_query_label = query_label
        self._last_total_relevant = total_relevant

        self.root.after(0, lambda: self._on_encoded_match_done(
            results, query_label, elapsed, method, encoding, total_relevant
        ))

    def _on_encoded_match_done(self, results, query_label, elapsed, method, encoding, total_relevant=0):
        self.progress['value'] = 100
        self.progress_label.config(text="检索完成")

        if not results:
            self.result_label.config(image='', text="未找到匹配结果")
            self.stats_label.config(text="")
            self.match_btn.config(state=tk.NORMAL)
            return

        # 显示 Top-1
        top1 = results[0]
        top1_path = top1['path']
        top1_label = top1['label']
        top1_dist = top1['distance']

        if os.path.exists(top1_path):
            self._show_image(top1_path, self.result_label, max_size=(406, 350))
            self.result_info_label.config(
                text=f"车牌: {top1_label}  |  距离: {top1_dist:.4f}  |  文件: {os.path.basename(top1_path)}"
            )

        # 填入表格
        for i, r in enumerate(results):
            self.tree.insert("", tk.END, values=(
                i + 1, r['label'], f"{r['distance']:.4f}", r['path']
            ))

        # 显示 Top-10 缩略图
        self._show_topk_images(results)

        # 计算指标
        correct_count = sum(1 for r in results if r['label'] == query_label)
        precision = correct_count / len(results) if results else 0
        recall = correct_count / total_relevant if total_relevant > 0 else 0

        # 计算每个截断点 K=1..10 的 AP@K，再取平均得到 mAP
        def ap_at_k(res, label, k, total_rel):
            hits = 0
            precisions = []
            for i in range(min(k, len(res))):
                if res[i]['label'] == label:
                    hits += 1
                    precisions.append(hits / (i + 1))
            if not precisions:
                return 0.0
            # 标准 AP@K：除以 min(总相关数, K)
            denom = min(total_rel, k) if total_rel > 0 else len(precisions)
            return sum(precisions) / denom

        ap_list = [ap_at_k(results, query_label, k, total_relevant) for k in range(1, 11)]
        map_value = sum(ap_list) / len(ap_list)

        # 当前 K 对应的 AP
        current_ap = ap_at_k(results, query_label, len(results), total_relevant)

        self.stats_label.config(
            text=(f"模式: {method}+{encoding.upper()}  |  查询车牌: {query_label}  |  "
                  f"Precision@{len(results)}: {precision:.2%}  |  "
                  f"Recall@{len(results)}: {recall:.2%}  |  "
                #   f"mAP@10: {map_value:.4f}  |  AP@{len(results)}: {current_ap:.4f}  |  "
                  f"mAP@10: {map_value:.4f}  |  "

                  f"耗时: {elapsed:.2f} ms")
        )
        self.match_btn.config(state=tk.NORMAL)

    # ===================== 构建码本 =====================

    def _build_codebook(self):
        encoding = self.encoding_var.get()
        method = self.method_var.get()
        if encoding == "none":
            return

        self.build_btn.config(state=tk.DISABLED)
        self.match_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.progress_label.config(text="正在构建码本，请稍候...")
        self.stats_label.config(text="")

        thread = threading.Thread(target=self._build_worker,
                                  args=(method, encoding), daemon=True)
        thread.start()

    def _build_worker(self, method, encoding):
        def progress_cb(stage, current, total):
            pct = int(current / total * 100)
            stage_name = {"extract": "提取特征", "kmeans": "K-Means 聚类", "encode": "编码"}.get(stage, stage)
            self.root.after(0, lambda: self._update_progress(pct, current, total, stage_name))

        success = build_codebook_and_encode(method, 256, encoding, progress_callback=progress_cb)
        self.root.after(0, lambda: self._on_build_done(success, method, encoding))

    def _on_build_done(self, success, method, encoding):
        if success:
            self.progress['value'] = 100
            self.progress_label.config(text="码本构建完成")
            messagebox.showinfo("完成", f"{method} + {encoding.upper()} 码本构建成功！")
        else:
            self.progress_label.config(text="码本构建失败")
            messagebox.showerror("错误", "码本构建过程中发生错误")
        self._update_codebook_status()
        self.match_btn.config(state=tk.NORMAL if self.query_image_path else tk.DISABLED)

    # ===================== 通用辅助 =====================

    def _update_progress(self, pct, current, total, stage=""):
        self.progress['value'] = pct
        if stage:
            self.progress_label.config(text=f"{stage}: {current} / {total}  ({pct}%)")
        else:
            self.progress_label.config(text=f"进度: {current} / {total}  ({pct}%)")

    def _clear_topk_images(self):
        """清空 Top-K 缩略图"""
        self.topk_photo_refs.clear()
        for i, (lbl, info) in enumerate(self.topk_labels):
            lbl.config(image='', text=f"#{i+1}", bg="#dddddd")
            info.config(text="")

    def _show_topk_images(self, results):
        """显示 Top-K 检索结果缩略图"""
        self._clear_topk_images()
        thumb_size = (100, 100)
        for i, r in enumerate(results[:10]):
            if i >= len(self.topk_labels):
                break
            lbl, info = self.topk_labels[i]
            path = r.get('path', '')
            label_text = r.get('label', '')
            dist_text = r.get('distance', '')
            if isinstance(dist_text, float):
                dist_str = f"{dist_text:.3f}"
            elif isinstance(dist_text, int):
                dist_str = f"{dist_text}pts"
            else:
                dist_str = str(dist_text)

            if path and os.path.exists(path):
                try:
                    img = Image.open(path)
                    img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.topk_photo_refs.append(photo)
                    lbl.config(image=photo, text="", bg="#ffffff")
                except Exception:
                    lbl.config(text=f"#{i+1}\n加载失败", bg="#ffcccc")
            else:
                lbl.config(text=f"#{i+1}\n无图像", bg="#ffcccc")

            info.config(text=f"{label_text}\n{dist_str}")

    def _draw_pr_curve(self):
        """绘制 Precision-Recall 曲线"""
        if not self.query_image_path:
            messagebox.showwarning("提示", "请先选择待匹配图像")
            return

        method = self.method_var.get()
        query_label = os.path.basename(os.path.dirname(self.query_image_path))

        def worker():
            try:
                self.root.after(0, lambda: self.progress_label.config(text="正在计算PR曲线..."))

                # 收集所有可用编码方式的PR数据
                encodings_to_test = []
                for enc in ['bof', 'tfidf', 'vlad']:
                    if check_cache_exists(method, 256, enc):
                        encodings_to_test.append(enc)

                if not encodings_to_test:
                    self.root.after(0, lambda: messagebox.showwarning("提示", "没有可用的编码缓存"))
                    return

                pr_data = {}  # {encoding: (recalls, precisions)}
                for enc in encodings_to_test:
                    # 加载索引
                    idx = load_train_index(method, 256, enc)
                    if idx is None:
                        continue
                    nn_index, _, train_labels, train_paths = idx

                    # 编码查询图像
                    qcode, _ = encode_single_image(self.query_image_path, method, 256, enc)
                    if qcode is None:
                        continue

                    # KNN 检索 K=10
                    results = knn_search_single(qcode, nn_index, train_labels, train_paths, k=946)
                    total_rel = sum(1 for l in train_labels if l == query_label)

                    recalls = []
                    precisions = []
                    hits = 0
                    for i, r in enumerate(results):
                        if r['label'] == query_label:
                            hits += 1
                        precisions.append(hits / (i + 1))
                        recalls.append(hits / total_rel if total_rel > 0 else 0)

                    pr_data[enc] = (recalls, precisions)

                self.root.after(0, lambda: self._show_pr_curve(pr_data, query_label, method))
            except Exception as err:
                msg = str(err)
                self.root.after(0, lambda: self._on_error(msg))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _show_pr_curve(self, pr_data, query_label, method):
        """弹出 matplotlib 窗口显示 PR 曲线"""
        if not pr_data:
            messagebox.showwarning("提示", "没有可用的PR数据")
            return

        plt.figure(figsize=(8, 6))
        colors = {'bof': 'blue', 'tfidf': 'green', 'vlad': 'red'}
        labels = {'bof': 'BoF', 'tfidf': 'BoF+TF-IDF', 'vlad': 'VLAD'}

        for enc, (recalls, precisions) in pr_data.items():
            # 添加 (0,1) 起始点使曲线更完整
            r = [0.0] + recalls
            p = [1.0] + precisions

            r_arr = np.array(r)
            p_arr = np.array(p)
            if len(r_arr) >= 3:
                # 去除重复的 recall 值（make_interp_spline 要求 x 无重复）
                # 保留每个 recall 值对应的最后一个 precision
                unique_r = []
                unique_p = []
                seen = set()
                for i in range(len(r_arr) - 1, -1, -1):
                    rv = round(float(r_arr[i]), 6)
                    if rv not in seen:
                        seen.add(rv)
                        unique_r.append(float(r_arr[i]))
                        unique_p.append(float(p_arr[i]))
                unique_r.reverse()
                unique_p.reverse()
                r_arr = np.array(unique_r)
                p_arr = np.array(unique_p)

                if len(r_arr) >= 3:
                    x_new = np.linspace(r_arr.min(), r_arr.max(), 300)
                    spl = make_interp_spline(r_arr, p_arr, k=2)
                    y_new = spl(x_new)
                    y_new = np.clip(y_new, 0.0, 1.0)
                    plt.plot(x_new, y_new, color=colors.get(enc, 'black'),
                             linestyle='-', label=labels.get(enc, enc), linewidth=2)
                else:
                    plt.plot(r_arr, p_arr, color=colors.get(enc, 'black'),
                             linestyle='-', label=labels.get(enc, enc), linewidth=2)
            else:
                plt.plot(r_arr, p_arr, color=colors.get(enc, 'black'),
                         linestyle='-', label=labels.get(enc, enc), linewidth=2)

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve\nQuery: {query_label} | Method: {method}', fontsize=14)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.show()

    def _draw_histograms(self):
        """绘制查询图像和 Top-10 近邻的特征编码直方图"""
        if not self.query_image_path:
            messagebox.showwarning("提示", "请先选择待匹配图像")
            return

        # 仅支持编码模式（BoF / TF-IDF / VLAD）
        encoding = self.encoding_var.get()
        if encoding == "none":
            messagebox.showwarning("提示", "原始匹配模式不支持编码直方图，请选择 BoF / TF-IDF / VLAD 编码方式")
            return

        method = self.method_var.get()
        n_words = 256

        # 检查是否有缓存的检索结果
        if not self._last_results:
            messagebox.showwarning("提示", "请先执行一次检索，再查看编码直方图")
            return

        def worker():
            try:
                self.root.after(0, lambda: self.progress_label.config(text="正在加载编码数据..."))

                # 加载训练集编码
                train_path = f'train_encode_{method}_{n_words}_{encoding}.pkl'
                if not os.path.exists(train_path):
                    self.root.after(0, lambda: messagebox.showwarning("提示", f"找不到编码文件: {train_path}"))
                    return
                with open(train_path, 'rb') as f:
                    train_data = pickle.load(f)

                # 查询图像编码
                qcode, q_label = encode_single_image(self.query_image_path, method, n_words, encoding)
                if qcode is None:
                    self.root.after(0, lambda: messagebox.showwarning("提示", "查询图像编码失败"))
                    return

                # 收集查询图像 + Top-10 的编码和标签
                codes_list = [qcode]
                q_name = os.path.basename(self.query_image_path)
                titles = [f"Query: {q_label}\n({q_name})"]
                colors_list = ['#FF5722']  # 查询图像用橙色

                for i, r in enumerate(self._last_results[:10]):
                    # 在训练集中查找对应编码
                    found = False
                    for j, tp in enumerate(train_data['paths']):
                        if tp == r['path']:
                            codes_list.append(train_data['codes'][j])
                            r_name = os.path.basename(r['path'])
                            titles.append(f"#{i+1}: {r['label']}\n({r_name})")
                            colors_list.append('#2196F3' if r['label'] == q_label else '#757575')
                            found = True
                            break
                    if not found:
                        # 如果找不到，实时编码
                        c, _ = encode_single_image(r['path'], method, n_words, encoding)
                        codes_list.append(c if c is not None else np.zeros_like(qcode))
                        titles.append(f"#{i+1}: {r['label']}\n(os.path.basename(r['path']))")
                        colors_list.append('#2196F3' if r['label'] == q_label else '#757575')

                self.root.after(0, lambda: self._show_histograms(codes_list, titles, colors_list, method, encoding))
            except Exception as err:
                msg = str(err)
                self.root.after(0, lambda: self._on_error(msg))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _show_histograms(self, codes_list, titles, colors_list, method, encoding):
        """弹出 matplotlib 窗口显示编码直方图"""
        n = len(codes_list)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = np.array(axes).reshape(1, -1)
        axes = axes.flatten()

        for i, (code, title, color) in enumerate(zip(codes_list, titles, colors_list)):
            ax = axes[i]
            code = np.array(code)

            if encoding == 'vlad':
                # VLAD 编码：按视觉单词分组，显示每组的 L2 范数（残差强度）
                n_words = 256
                D = len(code) // n_words
                vlad_matrix = code.reshape(n_words, D)
                norms = np.linalg.norm(vlad_matrix, axis=1)  # 每个视觉单词的残差强度
                x = np.arange(n_words)
                y = norms
                ax.set_xlabel("Visual Word Index", fontsize=8)
                ax.set_ylabel("Residual Norm", fontsize=8)
            else:
                # BoF / TF-IDF：直接显示直方图
                x = np.arange(len(code))
                y = code
                ax.set_xlabel("Dimension", fontsize=8)
                ax.set_ylabel("Value", fontsize=8)

            ax.bar(x, y, color=color, alpha=0.7, width=0.8)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle='--', alpha=0.3)

        # 隐藏多余的子图
        for j in range(n, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f"Feature Encoding Histograms\nMethod: {method} | Encoding: {encoding.upper()}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def _draw_avg_pr_curve(self):
        """绘制所有算法组合的平均 PR 曲线（不含 TF-IDF）"""
        # 收集所有可用的 (method, encoding) 组合，排除 tfidf
        all_configs = []
        for method in ['SIFT', 'ORB']:
            for encoding in ['bof', 'vlad']:
                if check_cache_exists(method, 256, encoding):
                    all_configs.append((method, encoding))

        if not all_configs:
            messagebox.showwarning("提示", "没有可用的编码缓存，请先构建码本")
            return

        self.avg_pr_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.progress_label.config(text="正在计算平均PR曲线...")

        thread = threading.Thread(target=self._avg_pr_worker_all, args=(all_configs,), daemon=True)
        thread.start()

    def _draw_idf_compare(self):
        """绘制 IDF 对比：使用IDF前后各两种算法的 PR 曲线"""
        configs = []
        for method in ['SIFT', 'ORB']:
            for encoding in ['bof', 'tfidf']:
                if check_cache_exists(method, 256, encoding):
                    configs.append((method, encoding))

        if len(configs) < 4:
            messagebox.showwarning("提示", "需要构建 BoF 和 TF-IDF 两种编码的码本才能对比")
            return

        self.idf_cmp_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.progress_label.config(text="正在计算IDF对比曲线...")

        thread = threading.Thread(target=self._avg_pr_worker_all, args=(configs,), daemon=True)
        thread.start()

    def _avg_pr_worker_all(self, all_configs):
        """对所有算法组合计算平均 PR 曲线和 mAP"""
        try:
            results = []  # list of (method, encoding, recall_points, avg_precisions, mAP, num_queries)
            total_configs = len(all_configs)

            for ci, (method, encoding) in enumerate(all_configs):
                self.root.after(0, lambda m=method, e=encoding, c=ci, t=total_configs:
                    self.progress_label.config(text=f"计算 {m}+{e.upper()} ({c+1}/{t})..."))

                index_data = load_train_index(method, 256, encoding)
                if index_data is None:
                    continue
                nn_index, train_codes, train_labels, train_paths = index_data
                N = len(train_codes)

                # 批量检索所有 N 个近邻
                distances, indices = nn_index.kneighbors(train_codes, n_neighbors=N)

                # 对每个查询计算 PR 曲线和 AP
                all_interp_precisions = []
                all_aps = []

                for idx in range(N):
                    q_label = train_labels[idx]
                    total_rel = sum(1 for l in train_labels if l == q_label) - 1

                    if total_rel <= 0:
                        continue

                    # 排除自身，收集邻居标签
                    neighbor_labels = []
                    for ni in indices[idx]:
                        if ni == idx:
                            continue
                        neighbor_labels.append(train_labels[ni])

                    # 构建 (recall, precision) 点列表，同时计算 AP
                    pr_points = []
                    hits = 0
                    for k, lbl in enumerate(neighbor_labels, start=1):
                        if lbl == q_label:
                            hits += 1
                            prec = hits / k
                            rec = hits / total_rel
                            pr_points.append((rec, prec))

                    # AP = sum(Precision@correct_k) / total_rel
                    ap = sum(prec for rec, prec in pr_points) / total_rel if total_rel > 0 else 0.0
                    all_aps.append(ap)

                    # 11 点插值
                    interp_prec = np.zeros(11, dtype=np.float32)
                    for ri in range(11):
                        r_target = ri / 10.0
                        max_p = 0.0
                        for rec, prec in pr_points:
                            if rec >= r_target and prec > max_p:
                                max_p = prec
                        interp_prec[ri] = max_p

                    all_interp_precisions.append(interp_prec)

                if len(all_interp_precisions) > 0:
                    avg_precisions = np.mean(all_interp_precisions, axis=0).tolist()
                    recall_points = [r / 10.0 for r in range(11)]
                    map_value = float(np.mean(all_aps))
                    results.append((method, encoding, recall_points, avg_precisions, map_value, len(all_interp_precisions)))

                pct = int((ci + 1) / total_configs * 100)
                self.root.after(0, lambda p=pct: self.progress.config(value=p))

            self.root.after(0, lambda: self._show_avg_pr_curve_all(results))
        except Exception as err:
            msg = str(err)
            self.root.after(0, lambda: self._on_error(msg))
        finally:
            self.root.after(0, lambda: self.avg_pr_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.idf_cmp_btn.config(state=tk.NORMAL))

    def _show_avg_pr_curve_all(self, results):
        """弹出 matplotlib 窗口显示所有算法组合的平均 PR 曲线"""
        self.progress['value'] = 100
        self.progress_label.config(text="平均PR曲线计算完成")

        if not results:
            messagebox.showwarning("提示", "没有可用的PR数据")
            return

        # 判断是 IDF 对比还是全方法对比
        is_idf_compare = all(enc in ('bof', 'tfidf') for _, enc, _, _, _, _ in results)

        if is_idf_compare:
            # IDF 对比模式：左侧 PR 曲线 + 右侧 mAP 柱状图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # 左图：PR 曲线
            colors_idf = {'SIFT': 'blue', 'ORB': 'red'}
            map_data = []  # for bar chart

            for method, encoding, recall_points, avg_precisions, map_value, num_queries in results:
                r_fixed = [0.0] + recall_points[1:]
                p_fixed = [1.0] + avg_precisions[1:]
                r_arr = np.array(r_fixed)
                p_arr = np.array(p_fixed)

                color = colors_idf.get(method, 'black')
                ls = '--' if encoding == 'tfidf' else '-'
                label = f"{method}+{'TF-IDF' if encoding == 'tfidf' else 'BoF'}"

                if len(r_arr) >= 3:
                    x_new = np.linspace(0.0, 1.0, 300)
                    spl = make_interp_spline(r_arr, p_arr, k=2)
                    y_new = spl(x_new)
                    y_new = np.clip(y_new, 0.0, 1.0)
                    ax1.plot(x_new, y_new, color=color, linestyle=ls, linewidth=2.5, label=label)
                else:
                    ax1.plot(r_arr, p_arr, color=color, linestyle=ls, linewidth=2.5, label=label)

                map_data.append((label, map_value, color, ls == '--'))

            ax1.set_xlabel('Recall', fontsize=11)
            ax1.set_ylabel('Precision', fontsize=11)
            ax1.set_title('PR Curve Comparison', fontsize=12)
            ax1.legend(loc='lower left', fontsize=9)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_xlim(-0.05, 1.05)
            ax1.set_ylim(-0.05, 1.05)

            # 右图：mAP 柱状图
            labels_bar = [d[0] for d in map_data]
            values_bar = [d[1] for d in map_data]
            colors_bar = [d[2] if not d[3] else d[2] for d in map_data]
            hatches = ['' if not d[3] else '//' for d in map_data]

            bars = ax2.bar(range(len(labels_bar)), values_bar, color=colors_bar, alpha=0.7, edgecolor='black')
            for bar, h in zip(bars, hatches):
                if h:
                    bar.set_hatch(h)

            ax2.set_xticks(range(len(labels_bar)))
            ax2.set_xticklabels(labels_bar, rotation=15, ha='right', fontsize=9)
            ax2.set_ylabel('mAP', fontsize=11)
            ax2.set_title('mAP Comparison', fontsize=12)
            ax2.set_ylim(0, 1.05)
            ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

            # 在柱子上方标注数值
            for i, (bar, val) in enumerate(zip(bars, values_bar)):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9)

            fig.suptitle('IDF Weighting Comparison (11-point interpolation)', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        else:
            # 全方法对比模式（不含 TF-IDF）：单图显示 PR 曲线
            plt.figure(figsize=(10, 7))
            colors = {
                ('SIFT', 'bof'): 'blue',
                ('SIFT', 'vlad'): 'red',
                ('ORB', 'bof'): 'cyan',
                ('ORB', 'vlad'): 'orange',
            }
            linestyles = {'bof': '-', 'vlad': '-.'}

            for method, encoding, recall_points, avg_precisions, map_value, num_queries in results:
                r_fixed = [0.0] + recall_points[1:]
                p_fixed = [1.0] + avg_precisions[1:]
                r_arr = np.array(r_fixed)
                p_arr = np.array(p_fixed)

                color = colors.get((method, encoding), 'black')
                ls = linestyles.get(encoding, '-')
                label = f"{method}+{encoding.upper()} (mAP={map_value:.3f})"

                if len(r_arr) >= 3:
                    x_new = np.linspace(0.0, 1.0, 300)
                    spl = make_interp_spline(r_arr, p_arr, k=2)
                    y_new = spl(x_new)
                    y_new = np.clip(y_new, 0.0, 1.0)
                    plt.plot(x_new, y_new, color=color, linestyle=ls, linewidth=2, label=label)
                else:
                    plt.plot(r_arr, p_arr, color=color, linestyle=ls, linewidth=2, label=label)

            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Average Precision-Recall Curve (11-point interpolation)\nBoF vs VLAD Comparison', fontsize=14)
            plt.legend(loc='lower left', fontsize=9)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
            plt.show()

    def _on_error(self, msg):
        self.progress_label.config(text="检索出错")
        messagebox.showerror("错误", f"检索过程中发生错误:\n{msg}")
        self.match_btn.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = VehicleMatcherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
