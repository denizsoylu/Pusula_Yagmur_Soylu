import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

################################################
# 0. Görünüm Ayarları
################################################
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1500)

################################################
# 1. Veri Seti Yükleme
################################################
file_path = r"C:\Users\HP\PycharmProjects\PusulaTalentAcademy\dataset\Talent_Academy_Case_DT_2025.xlsx"
df = pd.read_excel(file_path)

################################################
# 2. Temizleme Fonksiyonları
################################################
def clean_strings(dataframe, cols):
    for col in cols:
        dataframe[col] = (
            dataframe[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\xa0", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"[,;]+", ",", regex=True)
            .str.replace(r"`", "", regex=True)
        )
    return dataframe

def remove_special_characters(text):
    if pd.isnull(text):
        return text
    invisible_pattern = r'[\u0000-\u001F\u007F\u200B-\u200F\u2060-\u206F\uFEFF\u00AD]'
    text = re.sub(invisible_pattern, '', text)
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_commas(text):
    if pd.isnull(text):
        return text
    text = re.sub(r",\s*,+", ",", text)
    return text.strip(", ")

def normalize_text(text):
    if pd.isnull(text):
        return text
    text = text.lower()
    text = text.replace('"', '').replace('“', '').replace('”', '').replace('i̇', 'i')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_list_column(text):
    if pd.isnull(text):
        return text
    text = remove_special_characters(text)
    text = normalize_text(text)
    items = [item.strip() for item in text.split(',') if item.strip() != '']
    items = list(dict.fromkeys(items))  # unique
    return ','.join(items)

def correct_values(dataframe):
    dataframe["KanGrubu"] = dataframe["KanGrubu"].str.replace("0", "O")
    dataframe["Cinsiyet"] = dataframe["Cinsiyet"].fillna("Bilinmiyor")
    dataframe["KronikHastalik"] = dataframe["KronikHastalik"].str.lower().str.replace(", ", ",")
    replace_dict = {"hiportiroidizm": "hipotiroidizm", "volteren": "voltaren"}
    dataframe["KronikHastalik"] = dataframe["KronikHastalik"].replace(replace_dict, regex=True)
    dataframe["Bolum"] = dataframe["Bolum"].str.replace(", ", ",")
    dataframe["Alerji"] = dataframe["Alerji"].str.lower().str.strip()
    dataframe["Tanilar"] = dataframe["Tanilar"].str.replace(r"\xa0", " ", regex=True)
    dataframe["Tanilar"] = dataframe["Tanilar"].str.replace(",,", ",")
    dataframe["Tanilar"] = dataframe["Tanilar"].str.strip()
    return dataframe

################################################
# 3. Kategorik ve Sayısal Değişkenleri Alma
################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   str(dataframe[col].dtypes) not in ["category", "object", "bool"]]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car

################################################
# 4. Temizlik ve Standardizasyon
################################################
df = clean_strings(df, df.select_dtypes(include=["object"]).columns)
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(remove_special_characters)
for col in ["Tanilar", "KronikHastalik", "Bolum", "Alerji"]:
    df[col] = df[col].apply(normalize_text)
df = correct_values(df)

# Sütun isimleri ve tip düzeltmeleri
df.rename(columns={"TedaviSuresi": "TedaviSuresi(Seans)",
                   "UygulamaSuresi": "UygulamaSuresi(dakika)"}, inplace=True)
df["TedaviSuresi(Seans)"] = df["TedaviSuresi(Seans)"].astype(str).str.extract(r"(\d+)").astype(int)
df["UygulamaSuresi(dakika)"] = df["UygulamaSuresi(dakika)"].astype(str).str.extract(r"(\d+)").astype(int)

for col in ["Tanilar", "KronikHastalik", "Bolum", "Alerji"]:
    df[col] = df[col].apply(clean_commas).apply(normalize_list_column).str.lower().str.strip()

# Aynı HastaID’ye ait satırları birleştir
if "HastaID" in df.columns:
    df = df.groupby('HastaID').agg({
        'Tanilar': lambda x: ','.join(x.unique()),
        'KronikHastalik': lambda x: ','.join(x.unique()),
        'Bolum': lambda x: ','.join(x.unique()),
        'Alerji': lambda x: ','.join(x.unique()),
        'KanGrubu': 'first',
        'Cinsiyet': 'first',
        'TedaviSuresi(Seans)': 'mean',
        'UygulamaSuresi(dakika)': 'mean'
    }).reset_index()

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

################################################
# 5. Eksik Değer Doldurma
################################################
# Sayısal için KNN
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

# Kategorik için mod ile doldurma
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

################################################
# 6. Sayısal Standardizasyon
################################################
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

################################################
# 7. Kategorik Encoding (OneHot)
################################################
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop first ile dummy trap önlenir
encoded_cols = pd.DataFrame(ohe.fit_transform(df[cat_cols]),
                            columns=ohe.get_feature_names_out(cat_cols))
df = df.drop(columns=cat_cols)
df = pd.concat([df.reset_index(drop=True), encoded_cols.reset_index(drop=True)], axis=1)

################################################
# 8. EDA / Özet Kontrol
################################################
def check_df(dataframe, head=5):
    print("Shape:", dataframe.shape)
    print(dataframe.head(head))
    print(dataframe.tail(head))
    print("NA values:\n", dataframe.isnull().sum())

check_df(df)

################################################
# 9. Temizlenmiş Veri Setini Kaydetme
################################################
output_path = r"C:\Users\HP\PycharmProjects\PusulaTalentAcademy\dataset\Talent_Academy_Cleaned.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Temizlenmiş veri seti başarıyla kaydedildi: {output_path}")
