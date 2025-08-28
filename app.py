import streamlit as st
import pickle
import numpy as np
import random as rand
from sklearn.preprocessing import RobustScaler
# --- helper functions ---
def calculate_bmr(weight, height, age, gender):
    if gender.lower() == "male":
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:  # female
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

def calculate_bmi(weight, height):
    return weight / ((height/100) ** 2)

avg_values = {
    "Body_temp":38,
    "HeadCircumference": 56,
    "ShoulderWidth": 45,
    "ChestWidth": 90,
    "Belly": 30,
    "Waist": 80,
    "Hips": 95,
    "ArmLength": 60,
    "ShoulderToWaist": 40,
    "WaistToKnee": 50,
    "LegLength": 90
}

# --- load model ---
with open("New_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔥 Daily Calories Needed Prediction")

# --- user inputs (basic features only) ---
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age (years)", min_value=1, max_value=120, value=25, step=1)
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)
Body_temp = st.number_input("Body Temp C", min_value=38.0, max_value=42.0, step=0.1)
selected_activity = st.text_input("Enter Your Activity Type")
exercise_duration_min = st.number_input("Exercise_Duration_in_Minute", min_value=0.0, max_value=250.0, step=0.1)
activit_level =st.selectbox("Activity level", ["None", "Lightly Active","Moderately Active", "Very Active","Extra Active"])
activity_multiplier = 0
calories_burned = 0

    

# --- extra inputs (with default if left 0) ---
head_circ = st.number_input("Head Circumference (cm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
shoulder_w = st.number_input("Shoulder Width (cm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
chest_w = st.number_input("Chest Width (cm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
belly = st.number_input("Belly (cm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
waist = st.number_input("Waist (cm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
hips = st.number_input("Hips (cm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
arm_len = st.number_input("Arm Length (cm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
shoulder_to_waist = st.number_input("Shoulder to Waist (cm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
waist_to_knee = st.number_input("Waist to Knee (cm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)
leg_len = st.number_input("Leg Length (cm)", min_value=0.0, max_value=2000.0, value=0.0, step=0.1)

# --- derived features ---
bmr = calculate_bmr(weight, height, age, gender)
bmi = calculate_bmi(weight, height)


# --- replace 0 with averages ---
Body_temp = Body_temp if Body_temp > 0 else avg_values["Body_temp"]
head_circ = head_circ if head_circ > 0 else avg_values["HeadCircumference"]
shoulder_w = shoulder_w if shoulder_w > 0 else avg_values["ShoulderWidth"]
chest_w = chest_w if chest_w > 0 else avg_values["ChestWidth"]
belly = belly if belly > 0 else avg_values["Belly"]
waist = waist if waist > 0 else avg_values["Waist"]
hips = hips if hips > 0 else avg_values["Hips"]
arm_len = arm_len if arm_len > 0 else avg_values["ArmLength"]
shoulder_to_waist = shoulder_to_waist if shoulder_to_waist > 0 else avg_values["ShoulderToWaist"]
waist_to_knee = waist_to_knee if waist_to_knee > 0 else avg_values["WaistToKnee"]
leg_len = leg_len if leg_len > 0 else avg_values["LegLength"]

# --- prepare feature vector ---
gender_val = 1 if gender == "Male" else 0

if activit_level == "None":
    activity_multiplier = 1.2
elif activit_level == "Lightly Active":
    activity_multiplier = 1.375
elif activit_level == "Moderately Active":
    activity_multiplier = 1.55
elif activit_level == "Very Active":
    activity_multiplier = 1.725
elif activit_level == "Extra Active":
    activity_multiplier = 1.9

# --- calories burned based on multiplier ---
if activity_multiplier == 1.2:
    calories_burned = rand.randint(0, 50)
elif activity_multiplier == 1.375:
    calories_burned = rand.randint(50, 100)
elif activity_multiplier == 1.55:
    calories_burned = rand.randint(100, 200)
elif activity_multiplier == 1.725:
    calories_burned = rand.randint(200, 300)
elif activity_multiplier == 1.9:
    calories_burned = rand.randint(300, 400)
    


calories_needed = bmr * activity_multiplier
# --- prediction ---
features = np.array([[gender_val, age, height, weight, exercise_duration_min, Body_temp, calories_burned,
                      head_circ, shoulder_w, chest_w, belly, waist, hips, arm_len,
                      shoulder_to_waist, waist_to_knee, leg_len ,  activity_multiplier , bmr , bmi]])

# --- فرضيًا القيم الدنيا والعليا لكل feature حسب بيانات التدريب ---
feature_min = np.array([0, 1, 100, 30, 0, 36, 0, 50, 30, 50, 20, 50, 50, 40, 30, 30, 50, 1.2, 400, 10])
feature_max = np.array([1, 120, 250, 150, 180, 42, 400, 65, 60, 120, 100, 120, 120, 90, 50, 80, 120, 1.9, 2500, 50])

scaler = RobustScaler()
scaler.fit(np.vstack([feature_min, feature_max]))

features_scaled = scaler.transform(features)



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


if st.button("Predict", key="predict_button"):
    prediction = model.predict(features_scaled)
    st.success(f"Predicted Calories Needed Intake: {prediction[0]:.2f} kcal")
    st.info(f"(BMR: {bmr:.2f}, BMI: {bmi:.2f}, Manual Calories Needed: {calories_needed:.2f})")

    # --- Analysis Plots ---
    st.subheader("📊 Analysis & Insights")

    # 1) BMI Category Visualization
    bmi_categories = {
        "Underweight": (0, 18.5),
        "Normal": (18.5, 24.9),
        "Overweight": (25, 29.9),
        "Obese": (30, 100)
    }
# 1) BMI Category Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(bmi_categories.keys(), [18.5, 24.9, 29.9, 35],
               color=['blue','green','orange','red'])
        ax.axhline(y=bmi, color="black", linestyle="--", label=f"Your BMI: {bmi:.1f}")
        ax.set_ylabel("BMI Range")
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        values = [calories_needed, prediction[0], calories_burned]
        labels = ["Manual Calories", "Predicted Calories", "Calories Burned"]
    
        sns.barplot(x=labels, y=values, ax=ax2, palette="viridis")
        ax2.set_ylabel("Calories (kcal)")
        st.pyplot(fig2)


    metrics = ["Head", "Shoulder", "Chest", "Belly", "Waist", "Hips", "Arm", "Leg"]
    user_values = [head_circ, shoulder_w, chest_w, belly, waist, hips, arm_len, leg_len]
    avg_vals = [avg_values["HeadCircumference"], avg_values["ShoulderWidth"], avg_values["ChestWidth"],
                avg_values["Belly"], avg_values["Waist"], avg_values["Hips"],
                avg_values["ArmLength"], avg_values["LegLength"]]

# Put Radar + Pie side by side
    metrics = ["Head", "Shoulder", "Chest", "Belly", "Waist", "Hips", "Arm", "Leg"]
    user_values = [head_circ, shoulder_w, chest_w, belly, waist, hips, arm_len, leg_len]
    avg_vals = [avg_values["HeadCircumference"], avg_values["ShoulderWidth"], avg_values["ChestWidth"],
                avg_values["Belly"], avg_values["Waist"], avg_values["Hips"],
                avg_values["ArmLength"], avg_values["LegLength"]]

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    user_values += user_values[:1]  # close circle
    avg_vals += avg_vals[:1]
    angles += angles[:1]

    metrics = ["Head", "Shoulder", "Chest", "Belly", "Waist", "Hips", "Arm", "Leg"]
    user_values = [head_circ, shoulder_w, chest_w, belly, waist, hips, arm_len, leg_len]
    avg_vals = [avg_values["HeadCircumference"], avg_values["ShoulderWidth"], avg_values["ChestWidth"],
                avg_values["Belly"], avg_values["Waist"], avg_values["Hips"],
                avg_values["ArmLength"], avg_values["LegLength"]]

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    user_values += user_values[:1]  # close circle
    avg_vals += avg_vals[:1]
    angles += angles[:1]

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        ax3.plot(angles, user_values, "o-", label="You")
        ax3.fill(angles, user_values, alpha=0.25)
        ax3.plot(angles, avg_vals, "o-", label="Average")
        ax3.fill(angles, avg_vals, alpha=0.25)
        ax3.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax3.legend()
        st.pyplot(fig3)

    with col4:
        bmr_part = bmr
        activity_part = calories_needed - bmr
        fig4, ax4 = plt.subplots(figsize=(4,4))
        ax4.pie([bmr_part, activity_part], labels=["BMR", "Activity"], autopct="%1.1f%%", startangle=90,
                colors=["#66b3ff", "#99ff99"], wedgeprops={'width':0.4})
        ax4.set_title("Calories Source")
        st.pyplot(fig4)
    



    
    
    
    
