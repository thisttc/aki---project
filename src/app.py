import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
st.set_page_config(page_title="AKI Prediction Tool", layout="wide")

@st.cache_resource
def load_model_and_features():
    """
    โหลดโมเดลและรายชื่อ features ที่จำเป็น
    """
    try:
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, 'aki_rf_model.joblib')
        features_path = os.path.join(script_dir, 'feature_names.pkl')

        model = joblib.load(model_path) 
        feature_names = joblib.load(features_path)
        return model, feature_names
    except FileNotFoundError:
        st.error("ไม่พบไฟล์โมเดลที่จำเป็น (aki_rf_model.joblib หรือ feature_names.pkl) กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ src บน GitHub")
        st.stop()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        st.stop()
model, feature_names = load_model_and_features()

# โหลดรูปภาพ
try:
    feature_importance_img = Image.open('feature_importance.png')
    roc_curve_img = Image.open('roc_curve.png')
except FileNotFoundError:
    st.warning("ไม่พบไฟล์รูปภาพประกอบ (feature_importance.png หรือ roc_curve.png)")
    feature_importance_img = None
    roc_curve_img = None

# --- 2. ส่วนหัวของหน้าเว็บ (ย้าย st.set_page_config ไปด้านบนแล้ว) ---
st.title('เครื่องมือทำนายความเสี่ยงภาวะไตวายเฉียบพลัน (AKI)')
st.write("""
อัปโหลดไฟล์ CSV ข้อมูลผู้ป่วยเพื่อรับผลการทำนาย
กรุณาเลือประเภทของไฟล์ที่อัปโหลดให้ถูกต้อง
""")

# --- 3. ส่วนการอัปโหลดไฟล์ ---
uploaded_file = st.file_uploader("อัปโหลดไฟล์ข้อมูลผู้ป่วย (CSV)", type=["csv"])

if uploaded_file is not None:
    # --- 3.1 ให้ผู้ใช้เลือกประเภทไฟล์ (เลือกก่อนกดทำนาย) ---
    file_type = st.radio(
        "1. กรุณาระบุประเภทไฟล์ที่อัปโหลด:",
        ('ไฟล์ข้อมูลดิบ (Raw Dataset ที่มี52คอลัมน์ และ .)',
         'ไฟล์ข้อมูลที่ Clean แล้ว(มี 50 features ที่พร้อมใช้งาน)'),
        key='file_type_radio',
        help="""
        - **Raw Dataset**: ไฟล์ CSV ต้นฉบับ (52 คอลัมน์) แอปจะทำการ Cleansing (แปลง '.', เติม 0) และเลือกเฉพาะ 50 features ที่จำเป็นให้
        - **Cleaned Data**: ไฟล์ CSV ที่มีเฉพาะ 50 features ที่โมเดลต้องการ และต้องไม่มีค่าว่าง (Missing Values)
        """
    )
    
    # --- 3.2 ปุ่มสำหรับเริ่มทำนายผล ---
    if st.button('2. เริ่มทำนายผล (Predict)'):
        
        input_df = None
        original_df = None
        df_for_prediction = None
        
        try:
            # --- 3.3 อ่านและประมวลผลตามประเภทไฟล์ (หลังจากกดปุ่ม) ---
            if 'ไฟล์ข้อมูลดิบ' in file_type:
                # --- ผู้ใช้เลือกไฟล์ดิบ (52 คอลัมน์) ---
                st.info("โหมด: ไฟล์ข้อมูลดิบ - กำลังประมวลผล...")
                # อ่านไฟล์ โดยใช้แถวแรก (index 0) เป็น Header และแปลงค่า '.' เป็น NaN
                input_df = pd.read_csv(uploaded_file, header=0, na_values=['.'])
                original_df = input_df.copy()
                
                # ตรวจสอบว่ามี 52 คอลัมน์ตามที่คาดไว้หรือไม่
                if input_df.shape[1] != 52:
                    st.error(f"ไฟล์ดิบต้องมี 52 คอลัมน์ แต่ไฟล์ที่อัปโหลดมี {input_df.shape[1]} คอลัมน์")
                    st.stop()
                
                # **นี่คือตรรกะใหม่ที่ฉลาดกว่าเดิม (ใช้ตำแหน่ง)**
                # เรารู้ว่า features ที่ต้องใช้คือ คอลัมน์ที่ 2 ถึง 50 และคอลัมน์ที่ 52
                # (ใน index 0-based คือ 1 ถึง 49 และ 51)
                feature_indices = list(range(1, 50)) + [51] # [1, 2, ..., 49, 51]
                
                df_for_prediction = input_df.iloc[:, feature_indices]
                
                # **บังคับเปลี่ยนชื่อคอลัมน์** ให้ตรงกับที่โมเดลถูกฝึกมา
                df_for_prediction.columns = feature_names
                
                # เติมค่าว่าง (เช่น '.' ที่ถูกแปลงมา) ด้วย 0
                df_for_prediction = df_for_prediction.fillna(0)
                
            else:
                # --- ผู้ใช้เลือกไฟล์ที่ Cleansing แล้ว (50 คอลัมน์) ---
                st.info("โหมด: ไฟล์ Cleansing แล้ว - กำลังตรวจสอบ...")
                # อ่านไฟล์ โดย *ไม่* คาดหวัง Header (เพราะอาจมีแต่ข้อมูล)
                input_df = pd.read_csv(uploaded_file, header=None, na_values=['.'])
                original_df = input_df.copy()

                # ตรวจสอบว่ามี 50 คอลัมน์พอดีหรือไม่
                if input_df.shape[1] != 50:
                    st.error(f"ไฟล์ Cleansed ต้องมี 50 คอลัมน์ แต่ไฟล์ที่อัปโหลดมี {input_df.shape[1]} คอลัมน์")
                    st.stop()
                
                # **บังคับเปลี่ยนชื่อคอลัมน์** ให้ตรงกับที่โมเดลถูกฝึกมา
                df_for_prediction = input_df
                df_for_prediction.columns = feature_names
                
                # เติมค่าว่าง (ถ้ามี) ด้วย 0
                if df_for_prediction.isnull().sum().sum() > 0:
                    st.warning("ตรวจพบค่าว่างในไฟล์ 'Cleaned' - จะทำการเติมค่าว่างด้วย 0")
                    df_for_prediction = df_for_prediction.fillna(0)

            # --- 4. แสดงตัวอย่างข้อมูลและทำนายผล ---
            st.subheader("ตัวอย่างข้อมูลที่อัปโหลด (5 แถวแรก)")
            st.dataframe(original_df.head())

            st.subheader("ผลการทำนาย (Prediction Results)")
            with st.spinner('โมเดลกำลังทำนายผล...'):
                # ทำนายผล
                predictions = model.predict(df_for_prediction)
                probabilities = model.predict_proba(df_for_prediction)
                
                risk_map = {
                    0: "Low Risk (No AKI)",
                    1: "Moderate Risk (Stage 1)",
                    2: "High Risk (Stage 2)",
                    3: "Very High Risk (Stage 3)"
                }
                
                # --- สร้างตารางผลลัพธ์แบบง่ายตามที่ผู้ใช้ต้องการ ---
                results_df = pd.DataFrame()
                results_df['Patient ID'] = [f"Patient {i}" for i in range(1, len(predictions) + 1)]

                if 'ไฟล์ข้อมูลดิบ' in file_type:
                     try:
                         # ดึงข้อมูล AKI เดิม (คอลัมน์ที่ 51, index 50)
                         results_df['Original AKI Stage'] = input_df.iloc[:, 50]
                     except Exception as e:
                         st.warning(f"ไม่สามารถดึงข้อมูล 'Original AKI Stage' จากไฟล์ดิบได้: {e}")

                results_df['Predicted AKI Stage'] = predictions
                results_df['Predicted Risk'] = results_df['Predicted AKI Stage'].map(risk_map)

                st.success("ทำนายผลสำเร็จ!")
                st.dataframe(results_df) # แสดงตารางแบบง่าย
                
                
                # --- 5. ปุ่มดาวน์โหลดผลลัพธ์ (สร้างไฟล์ฉบับเต็มสำหรับดาวน์โหลด) ---
                
                # เพิ่มผลลัพธ์กลับเข้าไปในตาราง *Original* เพื่อการดาวน์โหลดที่ครบถ้วน
                original_df['Predicted Stage'] = predictions
                original_df['Predicted Risk'] = original_df['Predicted Stage'].map(risk_map)
                predicted_proba_list = [probabilities[i][pred] for i, pred in enumerate(predictions)]
                original_df['Confidence'] = [f"{p*100:.2f}%" for p in predicted_proba_list]

                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_results = convert_df_to_csv(original_df) # ใช้ original_df ที่มีข้อมูลครบถ้วน
                
                st.download_button(
                    label="ดาวน์โหลดผลลัพธ์ฉบับเต็ม (CSV)",
                    data=csv_results,
                    file_name="aki_predictions_full.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดร้ายแรง: {e}")
            st.error("กรุณาตรวจสอบไฟล์ของคุณอีกครั้ง หรือลองเลือกประเภทไฟล์ให้ถูกต้อง")

# --- 6. ส่วนแสดงผลการประเมินโมเดล (เหมือนเดิม) ---
st.subheader('ประสิทธิภาพและการอธิบายโมเดล')

if feature_importance_img and roc_curve_img:
    with st.expander("แสดงปัจจัยเสี่ยงสำคัญ (Feature Importance)"):
        st.image(feature_importance_img, caption="20 ปัจจัยที่โมเดลใช้ในการตัดสินใจ")

    with st.expander("แสดงความสามารถในการจำแนกโรค (ROC Curve)"):
        st.image(roc_curve_img, caption="ความสามารถของโมเดลในการจำแนก AKI แต่ละ Stage")
else:
    st.info("ไม่พบไฟล์รูปภาพแสดงประสิทธิภาพโมเดล")

