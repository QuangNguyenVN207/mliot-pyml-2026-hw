import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # Sử dụng backend Tkinter để hiển thị cửa sổ
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")
np.random.seed(42) 

try:
    df = sns.load_dataset("titanic")
    print("Đã tải từ seaborn.")
except Exception:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    print("Đã tải từ URL.")

# bỏ các cột rò rỉ/dư thừa, gán lại vào biến df
leaky = ['alive', 'who', 'adult_male', 'class', 'deck', 'embark_town', 'alone']
df = df.drop(columns=leaky)

# chia train/val/test có stratify
X = df.drop(columns=["survived"])
y = df["survived"]

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp,
    test_size=15/85,
    stratify=y_tmp,
    random_state=42
)
print("Tỷ lệ survived từng tập")
print("Train:", y_train.mean().round(3))
print("Val:", y_val.mean().round(3))
print("Test:", y_test.mean().round(3))


num_cols = ["age", "sibsp", "parch", "fare"]
cat_cols = ["sex", "embarked"]
ord_cols = ["pclass"]

# TODO 7: xây pipeline cho biến số và biến phân loại
pipe_so  = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  RobustScaler()),
])
pipe_cat = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    # ignore: bỏ qua giá trị lạ và không báo lỗi(toàn bộ các cột One-Hot tương ứng với hàng dữ liệu đó sẽ được điền số 0.)
    # sparse_output: (Định dạng dữ liệu đầu ra)
    # Nếu True (Mặc định): Quá trình One-Hot sinh ra cực kỳ nhiều số 0 (ví dụ: cột có 100 hạng mục sẽ sinh ra 99 số 0 và chỉ 1 số 1 cho mỗi hàng). Để tiết kiệm bộ nhớ RAM, thư viện sẽ nén nó thành Sparse Matrix (Ma trận thưa) — một dạng cấu trúc chỉ lưu trữ "vị trí" của các số 1.
    # Khi dùng False: Thuật toán sẽ trả về một mảng Numpy 2 chiều thông thường (Dense Array), hiển thị đầy đủ và rõ ràng từng số 0 và 1.
])

preprocess = ColumnTransformer([
    ("num", pipe_so,  num_cols),
    ("cat", pipe_cat, cat_cols),
    ("ord", "passthrough", ord_cols),
])

preprocess.fit(X_train)               # fit CHỈ trên train

X_train_t = preprocess.transform(X_train)
X_val_t = preprocess.transform(X_val)
X_test_t = preprocess.transform(X_test)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(y_true, y_pred):
    # tính loss binary cross-entropy
    epsilon = 1e-15  # để tránh log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # giới hạn giá trị dự đoán
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

y_train_np = y_train.to_numpy()[:, np.newaxis]  # chuyển sang dạng cột
y_val_np = y_val.to_numpy()[:, np.newaxis]  # chuyển sang dạng cột

# 1. Khởi tạo tham số
m_train, n_features = X_train_t.shape
w = np.zeros((n_features, 1))  # trọng số ban đầu
b = 0.0  # bias ban đầu

# Siêu tham số
learning_rate = 0.05
epochs = 2000

train_losses = []
val_losses = []

for epoch in range(epochs):
    # 2. Tính dự đoán
    z_train = np.dot(X_train_t, w) + b
    y_train_pred = sigmoid(z_train)

    z_val = np.dot(X_val_t, w) + b
    y_val_pred = sigmoid(z_val)

    # 3. Tính loss
    train_loss = compute_loss(y_train_np, y_train_pred)
    val_loss = compute_loss(y_val_np, y_val_pred)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # 4. Tính gradient
    dz_train = y_train_pred - y_train_np
    dw = np.dot(X_train_t.T, dz_train) / m_train
    db = np.sum(dz_train) / m_train

    # 5. Cập nhật tham số
    w -= learning_rate * dw
    b -= learning_rate * db

    # --- VALIDATION ---
    # Tính Loss trên tập Val để vẽ biểu đồ (không cập nhật w, b bằng tập Val)
    Z_val = np.dot(X_val_t, w) + b
    A_val = sigmoid(Z_val)
    val_loss = compute_loss(y_val_np, A_val)
    val_losses.append(val_loss)
    
    if epoch % 200 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("Huấn luyện hoàn tất!")

# 3. Vẽ trực quan hóa quá trình Loss cắm đầu xuống
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)

plt.title('Biểu đồ Loss (Code thuần Toán học)')
plt.xlabel('Epochs (Số lần lặp)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# 1. Chuẩn bị định dạng y_test (chuyển thành vector cột để khớp kích thước)
y_test_np = y_test.values.reshape(-1, 1)

# 2. Lan truyền tiến (Forward) trên tập Test bằng w và b hội tụ mới nhất
Z_test = np.dot(X_test_t, w) + b
A_test = sigmoid(Z_test)  # Đây là y_pred ở dạng xác suất (0 đến 1)

# 3. Phân loại bằng ngưỡng (Threshold = 0.5)
# Nếu xác suất >= 0.5 sẽ trả về True (1), ngược lại trả về False (0)
y_pred_labels = (A_test >= 0.5).astype(int)

# 4. Tính toán độ chính xác (Accuracy)
correct_predictions = np.sum(y_pred_labels == y_test_np)
total_samples = y_test_np.shape[0]
accuracy = correct_predictions / total_samples

print("--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ---")
print(f"Số lượng mẫu dự đoán đúng: {correct_predictions} / {total_samples}")
print(f"Độ chính xác (Accuracy)  : {accuracy * 100:.2f}%")



samples = 50 # Chỉ lấy 50 người đầu tiên để đồ thị thoáng mắt
x_axis = np.arange(samples)

plt.figure(figsize=(14, 6))

# 1. Vẽ các điểm Thực tế (y_real) - Chỉ nằm ở mức 0 hoặc 1
plt.scatter(x_axis, y_val_np[:samples], color='black', s=80, label='Thực tế (y_real)', marker='o', zorder=3)

# 2. Vẽ các điểm Dự đoán (y_pred) - Nằm rải rác từ 0 đến 1
plt.scatter(x_axis, A_val[:samples], color='red', s=60, label='Xác suất Dự đoán (y_pred)', marker='x', zorder=3)

# 3. Vẽ đường nối để thấy mô hình kéo xác suất về gần thực tế như thế nào
for i in range(samples):
    plt.plot([i, i], [y_val_np[i][0], A_val[i][0]], color='gray', linestyle='--', alpha=0.5)

# 4. Vẽ đường ranh giới sinh tử (Ngưỡng 0.5)
plt.axhline(0.5, color='green', linestyle='-', linewidth=2, alpha=0.7, label='Ngưỡng phân loại (0.5)')

plt.title('Đối chiếu Thực tế vs Dự đoán (50 Hành khách tập Validation)')
plt.xlabel('Hành khách thứ (Index)')
plt.ylabel('Giá trị / Xác suất')
plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0 (Mất)', '0.25', '0.5 (Ngưỡng)', '0.75', '1 (Sống)'])
plt.legend(loc='center right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

