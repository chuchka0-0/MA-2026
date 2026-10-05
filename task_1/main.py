"""
Компьютерная задача «Спектральный анализ сетевого трафика для обнаружения аномалий»
Вариант 02 (variant_02.csv)

"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "variant_02.csv"
K_MAX = 50          # считаем k = 0..50
K_LOW, K_HIGH = 2, 20   # порядки тригонометрического многочлена

# ---------------------------------------------------------------
# Шаг 1. Загрузка данных и график
# ---------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
idx = df["index"].to_numpy()
T = df["traffic"].to_numpy(dtype=float)
N = len(T)

print(f"Файл: {CSV_PATH}, N = {N}")
print(f"Среднее = {T.mean():.2f}, СКО = {T.std():.2f}, "
      f"min = {T.min():.2f}, max = {T.max():.2f}")

# линейный тренд (для описания графика)
slope, intercept = np.polyfit(idx, T, 1)
print(f"Линейный тренд: наклон = {slope:.4f} пакетов/с на отсчёт")

plt.figure(figsize=(12, 4))
plt.plot(idx, T, color="plum", lw=0.9)
plt.axhline(T.mean(), color="purple", ls="--", lw=0.8, label="среднее")
plt.xlabel("index (время)")
plt.ylabel("traffic (пакетов/с)")
plt.title("Шаг 1. Исходный временной ряд")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("step1_series.png", dpi=150)

# ---------------------------------------------------------------
# Шаг 2. Коэффициенты Фурье «ручками»
# ---------------------------------------------------------------
j = np.arange(N)

a0_half = np.sum(T) / N          # a0/2 — среднее значение
a = np.zeros(K_MAX + 1)
b = np.zeros(K_MAX + 1)
a[0] = 2 * a0_half               # a0

for k in range(1, K_MAX + 1):
    arg = 2 * np.pi * k * j / N
    a[k] = 2.0 / N * np.sum(T * np.cos(arg))
    b[k] = 2.0 / N * np.sum(T * np.sin(arg))

A = np.sqrt(a ** 2 + b ** 2)     # амплитуды гармоник
A[0] = np.nan                    # у k=0 амплитуды нет — есть среднее

print(f"\na0/2 (средний уровень) = {a0_half:.3f}")
print("\n k |      a_k |      b_k |      A_k")
print("---+----------+----------+---------")
for k in range(1, K_MAX + 1):
    print(f"{k:2d} | {a[k]:8.3f} | {b[k]:8.3f} | {A[k]:8.3f}")

# сохраним таблицу в CSV для отчёта
pd.DataFrame({"k": range(1, K_MAX + 1),
              "a_k": a[1:], "b_k": b[1:], "A_k": A[1:]}
             ).to_csv("fourier_coefficients.csv", index=False)

# ---------------------------------------------------------------
# Шаг 3. Спектр (гистограмма A_k) и поиск пиков
# ---------------------------------------------------------------
ks = np.arange(1, K_MAX + 1)
Ak = A[1:]

# робастный порог: медиана + 5 * MAD (устойчив к самим пикам)
med = np.median(Ak)
mad = np.median(np.abs(Ak - med))
threshold = med + 5 * 1.4826 * mad
peaks = ks[Ak > threshold]

print(f"\nМедиана A_k = {med:.2f}, MAD = {mad:.2f}, порог аномалии = {threshold:.2f}")
print(f"Пики выше порога: k = {peaks.tolist()}")

colors = ["hotpink" if k in peaks else "pink" for k in ks]
plt.figure(figsize=(12, 4))
plt.bar(ks, Ak, color=colors)
plt.axhline(threshold, color="purple", ls="--", label=f"порог = {threshold:.1f}")
for k in peaks:
    plt.annotate(f"k={k}\nA={A[k]:.1f}", (k, A[k]),
                 ha="center", va="bottom", fontsize=9)
plt.xlabel("k (число циклов за 500 отсчётов)")
plt.ylabel("A_k")
plt.title("Шаг 3. Амплитудный спектр")
plt.grid(alpha=0.3, axis="y")
plt.legend()
plt.tight_layout()
plt.savefig("step3_spectrum.png", dpi=150)

# ---------------------------------------------------------------
# Шаг 4. Интерпретация аномалии (печатаем числа для выводов)
# ---------------------------------------------------------------
print("\n=== Шаг 4. Аномалии ===")
if len(peaks) == 0:
    print("Аномальных пиков не обнаружено.")
for k in peaks:
    period = N / k
    share = A[k] / np.sum(Ak) * 100
    print(f"k = {k}: A = {A[k]:.2f} (в {A[k] / med:.0f} раз больше медианы фона), "
          f"период = N/k = {period:.1f} отсчётов, доля суммарной амплитуды ≈ {share:.0f}%")
print(f"Фон (медиана) A_k ≈ {med:.2f}; суточного пика на k=1: A_1 = {A[1]:.2f}")

# ---------------------------------------------------------------
# Шаг 5. Тригонометрический многочлен порядка K
# ---------------------------------------------------------------
def trig_poly(K, t):
    """T(t) ≈ a0/2 + sum_{k=1..K} (a_k cos(2πkt/N) + b_k sin(2πkt/N))"""
    y = np.full_like(t, a0_half, dtype=float)
    for k in range(1, K + 1):
        arg = 2 * np.pi * k * t / N
        y += a[k] * np.cos(arg) + b[k] * np.sin(arg)
    return y

t = np.arange(N, dtype=float)
fits = {K: trig_poly(K, t) for K in (K_LOW, K_HIGH)}

print("\n=== Шаг 5. Качество приближения ===")
for K, y in fits.items():
    rmse = np.sqrt(np.mean((T - y) ** 2))
    r2 = 1 - np.sum((T - y) ** 2) / np.sum((T - T.mean()) ** 2)
    print(f"K = {K:2d}: RMSE = {rmse:7.2f}, R^2 = {r2:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, (K, y) in zip(axes, fits.items()):
    ax.plot(t, T, color="plum", lw=1, label="исходный ряд")
    ax.plot(t, y, color="purple", lw=1.8, label=f"многочлен K = {K}")
    ax.set_ylabel("traffic")
    ax.set_title(f"Приближение тригонометрическим многочленом, K = {K}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
axes[-1].set_xlabel("index (время)")
plt.tight_layout()
plt.savefig("step5_approximation.png", dpi=150)

print("\nГрафики сохранены: step1_series.png, step3_spectrum.png, step5_approximation.png")
print("Таблица коэффициентов: fourier_coefficients.csv")
plt.show()