import pandas as pd
import os
from PIL import Image, ImageDraw, ImageOps
from concurrent.futures import ThreadPoolExecutor
import hashlib

# 創建圖片處理資料夾
image_save_dir = './flipped_img'
if not os.path.exists(image_save_dir):
    print(f"目錄 {image_save_dir} 不存在，已創建 {image_save_dir}")
    os.makedirs(image_save_dir)
else:
    print(f"{image_save_dir} 目錄已存在")
print('開始清整及圖片轉換')


def price(df):
    # 刪除菜價小於30元的資料
    df = df[df.iloc[:, 4] >= 30]

    # 先將價格列轉換為數值，處理特殊情況
    df.iloc[:, 4] = pd.to_numeric(df.iloc[:, 4], errors='coerce')

    # 進行四捨五入，再轉換為int
    df.iloc[:, 4] = df.iloc[:, 4].round()

    # 刪除不是int的整欄
    df = df.dropna(subset=[df.columns[4]], how='any')

    # 將價格列轉換為int，去除小數部分
    df.iloc[:, 4] = df.iloc[:, 4].astype(int, errors='ignore')

    return df


def duplicates(df):
    # 刪除擁有相同餐點id的整欄
    df = df.drop_duplicates(subset=2, keep='first')
    return df


def remove_none(df):
    # 刪除第9列為空的整欄
    df = df.dropna(subset=[8])
    # 刪除第10列沒有店家ID的整欄
    df = df.dropna(subset=[9])
    return df


def loc(df):
    # 使用 str.replace 將地址中的奇怪字符替換為空字符串
    df.iloc[:, 5] = df.iloc[:, 5].str.replace(
        '^\(△\) |\(△\)|\(O\) |\(O\)|\(X\) |\(X\)|\(○\) |\(○\)|\(x\) \(x\)|\(△|\( △\)|  \(M\) |\(M\) |\(#\) |\(#\)', '', regex=True)
    return df


def remove_undefinable_items(df):
    # 要刪除的品項列表
    items_to_remove = ['Soups', 'Noodles', 'Hot Pot',
                       'Fried Chicken', 'Bento', 'Teppanyaki']

    # 使用 str.replace 將指定品項替換為空字符串
    for item in items_to_remove:
        df.iloc[:, 8] = df.iloc[:, 8].str.replace(item, '', regex=False)
    return df


def merge_columns(df):
    # 檢查是否有 NaN 值，如果有，填充為空字符串
    df.iloc[:, 8] = df.iloc[:, 8].fillna('NAN')
    # 根據先決條件進行品項的整合
    conditions = [
        ('Brunch', ['Breakfast', 'Sandwiches & Toast']),
        ('Drinks', ['Drinks', 'Coffee']),
        ('Desserts', ['Desserts', 'Dou Hua', 'Donut', 'Cakes']),
        ('Vegetarian', ['Vegetarian']),
        ('American', ['American', 'Pizza', 'Steak', 'Burgers']),
        ('J&K', ['Japanese', 'Korean']),
        ('International', ['International', 'Southeast Asian', 'Curry']),
        ('TW', ['Taiwanese', 'Lu wei', 'Snacks', 'TW Fried Chicken',
                'Fried Rice', 'Chinese', 'Hong Kong', 'Dumpling', 'Congee', 'Healthy'])
    ]

    for result, conditions_list in conditions:
        df.iloc[:, 8] = df.apply(lambda row: result if any(
            cond in row[8] for cond in conditions_list) else row[8], axis=1)
    return df


def merge_feature_data(df, df_feature_data):
    # 創建一個字典，以 Feature_analys_all.csv 的第一列為鍵，第三列為值
    feature_dict = dict(
        zip(df_feature_data.iloc[:, 0], df_feature_data.iloc[:, 2]))

    # 在原始資料檔案的第11列加入 Feature_analys_all.csv 的相應資料
    df[11] = df[9].map(feature_dict)

    # 沒有對應的值就寫None
    df[11] = df[11].where(df[9].isin(feature_dict.keys()), 'None')
    return df


def move_price_to_right(df):
    # 使用pop方法移除第5列（索引為4）的資料
    price_column = df.pop(4)

    # 將移除的資料插入到DataFrame的最後一列
    df.insert(len(df.columns), "Price", price_column)

    return df


def categorize_price(price):
    if price <= 50:
        return 1
    elif 50 < price <= 100:
        return 2
    elif 100 < price <= 150:
        return 3
    elif 150 < price <= 200:
        return 4
    elif 200 < price <= 300:
        return 5
    else:
        return 6


def price_level(df):
    # 新增價格分類欄位
    df['價格分類'] = df.iloc[:, 10].apply(categorize_price)
    return df


def remove_same_images(df, folder_path):
    # 取得文件夾中的所有圖片名稱
    image_files = set(os.listdir(folder_path))

    # 刪除CSV中不存在於文件夾中的圖片名稱的行
    df = df[df.iloc[:, 0].isin(image_files)]

    # 檢查文件夾中的圖片是否在CSV的A欄位中，如果不在則刪除圖片
    for file in os.listdir(folder_path):
        if file not in df.iloc[:, 0].values:
            os.remove(os.path.join(folder_path, file))
    return df


def process_image(filename):
    # 定義處理圖片的函式
    # 覆蓋的區域範圍
    top1, bottom1 = 690, 760
    left1, right1 = 0, 1000

    top2, bottom2 = 240, 310
    left2, right2 = 0, 1000
    # 打開圖片
    image_path = os.path.join(folder_path, filename)
    img = Image.open(image_path)

    # 在圖片上畫兩個黑色矩形
    draw = ImageDraw.Draw(img)
    draw.rectangle([(left1, top1), (right1, bottom1)], fill="black")
    draw.rectangle([(left2, top2), (right2, bottom2)], fill="black")

    # 調整圖片大小為224x224像素
    img = img.resize((224, 224))

    # 儲存處理後的圖片到新的輸出目錄，使用原始檔案名稱
    output_filename = os.path.join(output_image_path, filename)
    img.save(output_filename)


def process_images(folder_path, output_image_path):
    # 新增處理圖片的函數
    with ThreadPoolExecutor(max_workers=5) as executor:
        image_filenames = [filename for filename in os.listdir(
            folder_path) if filename.endswith(".jpg")]
        executor.map(process_image, image_filenames)


def image_hash(image_path):
    # 計算圖片的哈希值
    image = Image.open(image_path)
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    return image_hash


def remove_duplicate_images(folder_path):
    # 創建一個字典來追蹤已經見過的哈希值
    seen_hashes = set()

    # 遍歷資料夾中的每個圖片
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # 跳過非圖片檔案
        if not os.path.isfile(image_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # 計算圖片的哈希值
        current_hash = image_hash(image_path)

        # 如果哈希值已經見過，刪除這個圖片
        if current_hash in seen_hashes:
            # print(f"刪除重複圖片: {filename}")
            os.remove(image_path)
            # # 移動檔案到 remove_dir 資料夾
            # new_path = os.path.join(remove_dir, os.path.basename(image_path))
            # shutil.move(image_path, new_path)
        else:
            # 將哈希值加入到已經見過的集合中
            seen_hashes.add(current_hash)


if __name__ == "__main__":
    # 初始化 DataFrame
    final_df = pd.DataFrame()

    # 迭代讀取檔案
    for i in range(1, 12):
        # 讀取 CSV 檔案到 DataFrame
        file_path = f'./shop_Data_Cleaning/row_Data/dishes_info_{i}.csv'
        df = pd.read_csv(file_path, header=None, encoding='utf-8-sig')

        # 讀取特徵資料
        feature_file_path = './google_Data_Cleaning/feature_analys/Feature_analys_all.csv'
        df_feature_data = pd.read_csv(
            feature_file_path, header=None, encoding='utf-8-sig')

        # 讀取圖片檔
        folder_path = f'./row_img/food_img_re_{i}'  # 包含圖片的文件夾路徑
        image_files = set(os.listdir(folder_path))

        # 圖片處理的目標資料夾
        output_image_path = "./flipped_img"

        # 進行清理
        df = price(df)
        df = duplicates(df)
        df = loc(df)
        df = remove_none(df)
        df = remove_undefinable_items(df)
        df = merge_columns(df)
        df = merge_feature_data(df, df_feature_data)
        df = remove_same_images(df, folder_path)
        df = move_price_to_right(df)
        df = price_level(df)
        # 使用新增的處理圖片函數
        process_images(folder_path, output_image_path)
        # 將結果追加到 final_df
        final_df = pd.concat([final_df, df], ignore_index=True)
        # 紀錄整理進度
        print(f"已整理完 dishes_info_{i}.csv 的資料")
    # 將清理後的資料寫回 CSV 檔案，並賦予column
    custom_column_names = ['pic_id', 'food_name', 'food_id', 'introduce', 'address',
                           'longitude', 'latitude', 'label', 'shop_id', 'sort', 'price', 'span']
    final_df.to_csv('./merge_data.csv', index=False,
                    header=custom_column_names, encoding='utf-8-sig')

    # 執行刪除重複圖片的函式
    remove_duplicate_images(output_image_path)

    # 讀取CSV文件
    final_df = pd.read_csv(
        './merge_data.csv', encoding='utf-8-sig', header=None, skiprows=1, names=custom_column_names)
    final_df = final_df.iloc[:, :]  # 確保讀取所有需要的欄位
    # 再次使用 remove_same_images 函式
    final_df = remove_same_images(final_df, output_image_path)
    # 初始化新的資料
    new_data = []

    # 遍歷CSV文件中的每一行
    for index, row in final_df.iterrows():
        image_name = row['pic_id']  # 圖片名稱
        d_price = row['price']  # 餐點價格
        image_path = os.path.join(output_image_path, image_name)

        if not os.path.exists(image_path):
            # print(f"文件不存在：跳過{image_path}")
            continue  # 跳過不存在的文件

        # 根據價格條件創建翻轉圖片並新增記錄
        if 300 < d_price <= 500 or 150 <= d_price < 200 or 200 <= d_price <= 300 or d_price > 500:
            img = Image.open(image_path)

            # 左右翻轉
            flipped_lr = ImageOps.mirror(img)
            new_image_name_lr = f"{os.path.splitext(image_name)[0]}_lr.jpg"
            new_image_path_lr = os.path.join(
                output_image_path, new_image_name_lr)
            flipped_lr.save(new_image_path_lr)

            # 新增記錄到新的資料
            new_data.append([new_image_name_lr] + row.tolist()[1:])

            # 如果價格在300以上，進行上下翻轉
            if d_price > 300:
                # 上下翻轉
                flipped_ud = ImageOps.flip(img)
                new_image_name_ud = f"{os.path.splitext(image_name)[0]}_ud.jpg"
                new_image_path_ud = os.path.join(
                    output_image_path, new_image_name_ud)
                flipped_ud.save(new_image_path_ud)

                # 新增上下翻轉的記錄
                new_data.append([new_image_name_ud] + row.tolist()[1:])

    # 將新的資料合併到原始 CSV 文件
    new_df = pd.DataFrame(new_data, columns=custom_column_names)
    df_combined = pd.concat([final_df, new_df], ignore_index=True)

    # 儲存新的 CSV 文件到一個新文件中
    new_csv_file = "merge_data_flipped.csv"
    df_combined.to_csv(new_csv_file, index=False, encoding='utf-8-sig')

    print("已結束清整及反轉，並產生 merge_data.csv 及 merge_data_flipped.csv 兩個檔案")