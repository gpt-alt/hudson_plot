import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import linalg
from scipy.stats import gaussian_kde
import matplotlib

matplotlib.use('TkAgg')


class HudsonDiagramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hudson Diagram")

        # 自动适应窗口大小
        self.root.geometry("")  # 自动适配内容
        self.root.update_idletasks()
        self.root.minsize(800, 600)

        self.data = None
        self.current_file = None
        self.setup_ui()

    def setup_ui(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 绘图区域
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.fig = Figure(figsize=(10, 7), facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 初始化显示空的Hudson框架
        self.draw_empty_hudson()

        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="", padding="15")
        control_frame.pack(fill=tk.X, pady=(0, 5))

        # 第一行：散点类型和颜色
        row_1 = ttk.Frame(control_frame)
        row_1.pack(fill=tk.X, pady=5)

        ttk.Label(row_1, text="散点类型", font=("Arial", 10)).grid(row=0, column=0, padx=(0, 10), sticky='w')
        self.plot_type = tk.StringVar(value="o")
        plot_combo = ttk.Combobox(row_1, textvariable=self.plot_type, values=["o", "density"], width=20, state="readonly")
        plot_combo.grid(row=0, column=1, padx=(0, 50))

        ttk.Label(row_1, text="颜色", font=("Arial", 10)).grid(row=0, column=2, padx=(0, 10), sticky='w')
        self.color = tk.StringVar(value="r")
        color_combo = ttk.Combobox(row_1, textvariable=self.color, values=["r", "b", "g", "k"], width=20, state="readonly")
        color_combo.grid(row=0, column=3, padx=(0, 50))

        ttk.Label(row_1, text="Slider_Size", font=("Arial", 10)).grid(row=0, column=4, padx=(0, 10), sticky='w')

        # 滑块框架
        slider_frame = ttk.Frame(row_1)
        slider_frame.grid(row=0, column=5, sticky='ew')

        self.size_var = tk.DoubleVar(value=50)
        self.size_slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.size_var,
                                    showvalue=False, length=200, sliderlength=20)
        self.size_slider.pack(side=tk.LEFT)

        # 滑块刻度标签
        tick_frame = ttk.Frame(row_1)
        tick_frame.grid(row=1, column=5, sticky='ew')
        tick_labels = ttk.Frame(tick_frame)
        tick_labels.pack(fill=tk.X)
        for i, val in enumerate([0, 20, 40, 60, 80, 100]):
            lbl = ttk.Label(tick_labels, text=str(val), font=("Arial", 8))
            lbl.place(relx=i / 5, anchor='n')

        # 第二行：按钮区域
        row2 = ttk.Frame(control_frame)
        row2.pack(fill=tk.X, pady=(15, 5))

        # 左侧空白
        ttk.Label(row2, text="").pack(side=tk.LEFT, expand=True)

        # 选择数据文件按钮
        self.file_button = ttk.Button(row2, text="选择数据文件", command=self.load_file, width=15)
        self.file_button.pack(side=tk.LEFT, padx=5)

        # 绘图按钮
        self.plot_button = ttk.Button(row2, text="绘图", command=self.plot_diagram, width=10)
        self.plot_button.pack(side=tk.LEFT, padx=5)

        # 右侧空白
        ttk.Label(row2, text="").pack(side=tk.LEFT, expand=True)

        # 底部标签
        footer = ttk.Label(main_frame, text="玉面小子", font=("Arial", 9), foreground="#666666")
        footer.pack(side=tk.RIGHT, pady=5)

    def load_file(self):
        """加载数据文件"""
        filename = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("Data files", "*.dat"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.data = self.read_moment_tensor(filename)
                self.current_file = filename
                messagebox.showinfo("成功", f"已成功加载 {len(self.data)} 条数据！")
            except Exception as e:
                messagebox.showerror("错误", f"读取文件失败:\n{str(e)}")

    def read_moment_tensor(self, filename):
        """读取矩张量数据"""
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释行和空行
                if line and not line.startswith('%') and not line.startswith('#'):
                    try:
                        values = [float(x) for x in line.split()]
                        if len(values) >= 6:
                            data.append(values)
                    except ValueError:
                        continue

        if len(data) == 0:
            raise ValueError("文件中没有找到有效数据")

        return np.array(data)

    def ae_srctype(self, M):
        """计算k和T参数"""
        try:
            # 计算特征值
            eigenvalues = linalg.eigvals(M)
            eigenvalues = np.sort(np.real(eigenvalues))[::-1]  # 降序排列

            # 计算平均值并中心化
            mean_val = np.mean(eigenvalues)
            centered = eigenvalues - mean_val

            # 计算k和T
            denom = np.abs(mean_val) + max(np.abs(centered[0]), np.abs(centered[2]))
            if denom == 0:
                return None, None

            k = mean_val / denom
            max_abs = max(np.abs(centered[0]), np.abs(centered[2]))
            T = 0 if max_abs == 0 else 2 * centered[1] / max_abs

            return k, T

        except:
            return None, None

    def ae_uvt(self, T, k):
        """将T和k转换为u和v坐标"""
        T = np.atleast_1d(T)
        k = np.atleast_1d(k)

        tau = T * (1 - np.abs(k))
        u = np.full_like(T, np.nan, dtype=float)
        v = np.full_like(k, np.nan, dtype=float)

        # 第2和第4象限
        idx = ((tau > 0) & (k < 0)) | ((tau < 0) & (k > 0))
        u[idx] = tau[idx]
        v[idx] = k[idx]

        # 第1象限 A区
        idx = (tau < 4 * k) & (tau >= 0) & (k >= 0)
        u[idx] = tau[idx] / (1 - tau[idx] / 2)
        v[idx] = k[idx] / (1 - tau[idx] / 2)

        # 第1象限 B区
        idx = (tau >= 4 * k) & (tau >= 0) & (k >= 0)
        u[idx] = tau[idx] / (1 - 2 * k[idx])
        v[idx] = k[idx] / (1 - 2 * k[idx])

        # 第3象限
        idx = (tau >= 4 * k) & (tau <= 0) & (k <= 0)
        u[idx] = tau[idx] / (1 + tau[idx] / 2)
        v[idx] = k[idx] / (1 + tau[idx] / 2)

        idx = (tau < 4 * k) & (tau <= 0) & (k <= 0)
        u[idx] = tau[idx] / (1 + 2 * k[idx])
        v[idx] = k[idx] / (1 + 2 * k[idx])

        return u, v

    def draw_hudson_frame(self):
        """绘制Hudson图框架"""
        # 外框
        bx = [0, 4 / 3, 0, -4 / 3, 0]
        by = [1, 1 / 3, -1, -1 / 3, 1]
        self.ax.plot(bx, by, 'k-', linewidth=2)

        # 中心线
        self.ax.plot([-1, 1], [0, 0], 'k-', linewidth=1)
        self.ax.plot([0, 0], [-1, 1], 'k-', linewidth=1)

        # 等值线
        V_V = np.linspace(-1, 1, 200)
        levels = [-0.5, 0, 0.5]
        gray_color = [0.5, 0.5, 0.5]

        for level in levels:
            u_1, v_1 = self.ae_uvt(V_V, level * np.ones_like(V_V))
            u_2, v_2 = self.ae_uvt(level * np.ones_like(V_V), V_V)
            valid_1 = ~np.isnan(u_1) & ~np.isnan(v_1)
            valid_2 = ~np.isnan(u_2) & ~np.isnan(v_2)
            self.ax.plot(u_1[valid_1], v_1[valid_1], '--', color=gray_color, linewidth=1)
            self.ax.plot(u_2[valid_2], v_2[valid_2], '--', color=gray_color, linewidth=1)

    def add_labels(self):
        """添加标签"""
        labels = [
            (1, 1, 'Explosion', 'center', 'bottom', 0.03),
            (1, -1, 'Implosion', 'center', 'top', -0.03),
            (1, 0, 'CLVD (-)', 'center', 'top', -0.03),
            (1, -5 / 9, 'Anticrack', 'left', 'top', -0.03),
            (-1, 0, 'CLVD (+)', 'center', 'top', -0.03),
            (-1, 5 / 9, 'Tensile Crack', 'right', 'bottom', 0.03),
            (0, 0, 'DC', 'center', 'bottom', 0.03),
            (-1, 1 / 3, 'LVD (+)', 'right', 'bottom', 0),
            (1, -1 / 3, 'LVD (-)', 'left', 'top', 0),
        ]

        for T_val, k_val, text, ha, va, offset in labels:
            u, v = self.ae_uvt(np.array([T_val]), np.array([k_val]))
            if not np.isnan(u[0]) and not np.isnan(v[0]):
                self.ax.text(u[0], v[0] + offset, text, horizontalalignment=ha, verticalalignment=va, fontsize=12,
                             fontname='Times New Roman', fontweight='normal')

    def draw_empty_hudson(self):
        """绘制空的Hudson框架"""
        self.ax.clear()
        self.draw_hudson_frame()
        self.add_labels()

        # 设置坐标轴
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('u', fontsize=13, fontname='Times New Roman')
        self.ax.set_ylabel('v', fontsize=13, fontname='Times New Roman')
        self.ax.grid(False)

        # 设置背景为白色
        self.ax.set_facecolor('white')
        self.fig.patch.set_facecolor('white')

        self.canvas.draw()

    def plot_diagram(self):
        """绘制Hudson图"""
        if self.data is None:
            messagebox.showwarning("警告", "请先加载数据文件！")
            return

        try:
            self.ax.clear()

            # 计算u, v坐标
            u_list = []
            v_list = []
            mag_list = []

            for i in range(len(self.data)):
                M_matrix = np.array([
                    [self.data[i, 0], self.data[i, 5], self.data[i, 4]],
                    [self.data[i, 5], self.data[i, 1], self.data[i, 3]],
                    [self.data[i, 4], self.data[i, 3], self.data[i, 2]]
                ])

                k, T = self.ae_srctype(M_matrix)
                if k is not None and T is not None:
                    u, v = self.ae_uvt(np.array([T]), np.array([k]))
                    if not np.isnan(u[0]) and not np.isnan(v[0]):
                        u_list.append(u[0])
                        v_list.append(v[0])
                        # 如果有震级数据
                        if self.data.shape[1] >= 7:
                            mag_list.append(self.data[i, 6])
                        else:
                            mag_list.append(1.0)

            if len(u_list) == 0:
                messagebox.showwarning("警告", "没有有效的数据点可以绘制！")
                self.draw_empty_hudson()
                return

            u_arr = np.array(u_list)
            v_arr = np.array(v_list)
            mag_arr = np.array(mag_list)

            # 绘制图形
            plot_type = self.plot_type.get()

            if plot_type == "o":
                # 散点图
                color_map = {'r': 'red', 'b': 'blue', 'g': 'green', 'k': 'black'}
                color = color_map.get(self.color.get(), 'red')
                size = mag_arr * self.size_var.get()

                self.ax.scatter(u_arr, v_arr, s=size, c=color, alpha=0.6,
                                edgecolors='black', linewidths=0.5, zorder=3)
            else:
                # 密度图
                if len(u_arr) > 1:
                    try:
                        xlin = np.linspace(-4 / 3, 4 / 3, 300)
                        ylin = np.linspace(-1, 1, 300)
                        X, Y = np.meshgrid(xlin, ylin)

                        # 使用核密度估计
                        positions = np.vstack([X.ravel(), Y.ravel()])
                        values = np.vstack([u_arr, v_arr])
                        kernel = gaussian_kde(values)
                        Z = np.reshape(kernel(positions).T, X.shape)

                        # 遮罩处理 - 只显示菱形内部
                        from matplotlib.path import Path
                        bx = np.array([0, 4 / 3, 0, -4 / 3, 0])
                        by = np.array([1, 1 / 3, -1, -1 / 3, 1])
                        path = Path(np.column_stack([bx, by]))
                        points = np.column_stack([X.ravel(), Y.ravel()])
                        mask = ~path.contains_points(points).reshape(X.shape)
                        Z[mask] = np.nan

                        # 绘制密度图
                        im = self.ax.pcolormesh(X, Y, Z, shading='auto', cmap='jet', zorder=1)
                        plt.colorbar(im, ax=self.ax, label='AE Density')
                    except Exception as e:
                        messagebox.showerror("错误", f"绘制密度图失败:\n{str(e)}")
                        self.draw_empty_hudson()
                        return

            # 绘制框架和标签
            self.draw_hudson_frame()
            self.add_labels()

            # 设置坐标轴
            self.ax.set_xlim([-1.5, 1.5])
            self.ax.set_ylim([-1.1, 1.1])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('u', fontsize=13, fontname='Times New Roman')
            self.ax.set_ylabel('v', fontsize=13, fontname='Times New Roman')
            self.ax.grid(False)
            self.ax.set_facecolor('white')

            self.canvas.draw()

            messagebox.showinfo("成功", f"已成功绘制 {len(u_list)} 个数据点！")

        except Exception as e:
            messagebox.showerror("错误", f"绘图失败:\n{str(e)}")
            self.draw_empty_hudson()


def main():
    root = tk.Tk()
    HudsonDiagramApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
