import pandas as pd
import joblib
import os

# --- القسم 1: استدعاء المكتبات الضرورية ---
# نستدعي الأدوات التي نحتاجها للتعامل مع البيانات والذكاء الاصطناعي
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

print("⏳ جاري تحميل البيانات وتدريب الموديل على جهازك...")

# --- القسم 2: تحميل البيانات وتنظيفها ---
# 1. تحميل الملف (تأكد أن ملف CSV بجانب هذا الملف)
df = pd.read_csv("credit_risk_dataset.csv")

# 2. تنظيف البيانات: حذف الأعمار غير المنطقية (أكبر من 100)
df = df[df['person_age'] <= 100]

# --- القسم 3: هندسة الميزات (Feature Engineering) ---
# إضافة ميزة "عبء الفائدة" لزيادة ذكاء الموديل
# المعادلة: (قيمة القرض * نسبة الفائدة) / الدخل
df['interest_burden'] = (df['loan_amnt'] * (df['loan_int_rate'] / 100)) / df['person_income']

# معالجة القيم الفارغة الناتجة عن القسمة (إذا كان الدخل 0) واستبدالها بـ 0
df['interest_burden'] = df['interest_burden'].fillna(0)

# تقسيم البيانات إلى مدخلات (X) ومخرجات (y)
X = df.drop('loan_status', axis=1) # كل الأعمدة ما عدا النتيجة
y = df['loan_status']              # النتيجة المطلوبة (0 أو 1)

# --- القسم 4: بناء خط الإنتاج (Pipeline) ---
# تحديد الأعمدة الرقمية والنصية تلقائياً
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# أداة معالجة الأرقام: ملء الفراغات + توحيد المقياس
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# أداة معالجة النصوص: ملء الفراغات + تحويل النصوص لأرقام
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# دمج المحولين في معالج واحد
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- القسم 5: التدريب (Training) ---
# إعداد الموديل النهائي: المعالج + خوارزمية الغابة العشوائية
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

# البدء بالتدريب الفعلي
print("⚙️ جاري تدريب الموديل (قد يستغرق دقيقة)...")
model.fit(X, y)

# --- القسم 6: الحفظ والتحليل (Saving & XAI) ---
# 1. حفظ الموديل لاستخدامه في الموقع
joblib.dump(model, 'credit_risk_model.pkl')

# 2. استخراج أهم الميزات (لفهم سبب اتخاذ القرار)
# نحتاج لاستخراج أسماء الأعمدة الجديدة بعد التحويل (OneHotEncoding)
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
# استخراج الأرقام التي تدل على الأهمية
importances = model.named_steps['classifier'].feature_importances_

# وضعها في جدول للعرض
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print("\n📊 أهم 3 عوامل يعتمد عليها الموديل في قراره:")
print(feature_importance_df.head(3))

print("\n✅ تم الحفظ بنجاح! ملف 'credit_risk_model.pkl' جاهز الآن.")