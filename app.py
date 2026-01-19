from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# تحميل الموديل
current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, 'credit_risk_model.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # سحب البيانات من الواجهة
        data = {
            'person_age': float(request.form['person_age']),
            'person_income': float(request.form['person_income']),
            'person_emp_length': float(request.form['person_emp_length']),
            'loan_amnt': float(request.form['loan_amnt']),
            'loan_int_rate': float(request.form['loan_int_rate']),
            'cb_person_cred_hist_length': float(request.form['cb_person_cred_hist_length']),
            'person_home_ownership': request.form.get('person_home_ownership', 'RENT'),
            'loan_intent': request.form.get('loan_intent', 'PERSONAL'),
            'loan_grade': request.form.get('loan_grade', 'A'),
            'cb_person_default_on_file': request.form.get('cb_person_default_on_file', 'N')
        }

        # الحسابات الذكية
        data['loan_percent_income'] = data['loan_amnt'] / data['person_income']
        data['interest_burden'] = (data['loan_amnt'] * (data['loan_int_rate'] / 100)) / data['person_income']

        # تحويل لجدول DataFrame
        input_df = pd.DataFrame([data])

        # التوقع والنسب
        prediction = model.predict(input_df)[0]
        prob_safe = model.predict_proba(input_df)[0][0] * 100
        
        # استخراج أهم ميزة (XAI)
        # ملاحظة: نستخدم الأسماء التي تظهر للمستخدم
        features_ar = {'loan_int_rate': 'نسبة الفائدة', 'person_income': 'الدخل السنوي', 'loan_amnt': 'مبلغ القرض'}
        top_feat_name = model.named_steps['preprocessor'].get_feature_names_out()[model.named_steps['classifier'].feature_importances_.argmax()]
        friendly_name = features_ar.get(top_feat_name.split('__')[-1], "تاريخ الائتمان")

        res_text = "✅ عميل آمن" if prediction == 0 else "⚠️ خطر تعثر"
        res_color = "#2ecc71" if prediction == 0 else "#e74c3c"

        return render_template('index.html', 
                               prediction_text=res_text, 
                               color=res_color, 
                               prob_safe=round(prob_safe, 1),
                               top_feature=friendly_name)

    except Exception as e:
        return render_template('index.html', prediction_text=f"خطأ: {e}", color="black")

if __name__ == "__main__":
    app.run(debug=True)