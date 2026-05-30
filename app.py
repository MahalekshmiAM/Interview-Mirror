
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.set_page_config(
    page_title="Interview Mirror",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
    font-family: 'Segoe UI';
}


.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.main-title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(90deg,#38bdf8,#8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-top: 5px;
    margin-bottom: 25px;
}

.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 52px;
    border: none;
    font-weight: 700;
    font-size: 16px;
    color: white;
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(124,58,237,0.35);
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
}

img {
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.08);
}

.landing-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 28px;
    border-radius: 22px;
    backdrop-filter: blur(10px);
}

.section-title {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 15px;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 30px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>Interview Mirror</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Real-Time AI Body Language & Confidence Tracking</div>",
    unsafe_allow_html=True
)

if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.report = False
    st.session_state.eye_hist = []
    st.session_state.pose_hist = []
    st.session_state.stability_hist = []

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("▶ Start Analysis"):
        st.session_state.running = True
        st.session_state.report = False
        st.session_state.eye_hist = []
        st.session_state.pose_hist = []
        st.session_state.stability_hist = []

with c2:
    if st.button("🛑 Stop Session"):
        st.session_state.running = False
        st.session_state.report = True

with c3:
    if st.button("🔄 Reset"):
        st.session_state.running = False
        st.session_state.report = False
        st.session_state.eye_hist = []
        st.session_state.pose_hist = []
        st.session_state.stability_hist = []

if not st.session_state.running and not st.session_state.report:

    st.markdown("<br>", unsafe_allow_html=True)

    left_intro, right_intro = st.columns([1.5, 1])

    with left_intro:

        st.markdown("""
        <div class="landing-box">

        <div class="section-title">
        🚀 AI Powered Interview Intelligence
        </div>

        <p style="font-size:18px;line-height:1.8;color:#cbd5e1;">

        Improve your interview confidence using
        real-time AI body language analysis.

        <br><br>

        This system tracks:
        <br><br>

        👁 Eye Contact<br>
        🧍 Posture<br>
        🎯 Stability<br>
        😐 Confidence Score

        <br><br>

        Get professional interview feedback instantly
        using AI-powered live analysis.

        </p>

        </div>
        """, unsafe_allow_html=True)

    with right_intro:

        st.markdown("""
        <div class="landing-box">

        <div class="section-title">
        💡 Interview Tips
        </div>

        <p style="font-size:17px;line-height:2;color:#cbd5e1;">

        ✔ Sit in proper lighting<br>

        ✔ Keep camera at eye level<br>

        ✔ Maintain stable posture<br>

        ✔ Look at webcam naturally<br>

        ✔ Avoid excessive movement<br>

        ✔ Sit confidently and calmly

        </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        st.metric("⚡ Analysis Speed", "Real-Time")

    with b:
        st.metric("🧠 AI Tracking", "Live")

    with c:
        st.metric("📊 Final Reports", "Enabled")

mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

left, right = st.columns([2, 1])

video_box = left.empty()
panel = right.empty()

def clamp(x):
    return max(0, min(100, x))

if st.session_state.running:

    cap = cv2.VideoCapture(0)

    prev_nose = None

    while st.session_state.running:

        ret, frame = cap.read()

        if not ret:
            st.error("Camera not detected")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        eye_score = 50
        pose_score = 50
        stability_score = 50

        if results.multi_face_landmarks:

            lm = results.multi_face_landmarks[0].landmark

            nose = lm[1]
            left_eye = lm[33]
            right_eye = lm[263]
            left_face = lm[234]
            right_face = lm[454]
            chin = lm[152]
            forehead = lm[10]

            # ================= EYE CONTACT =================
            eye_center = (left_eye.x + right_eye.x) / 2
            face_center = (left_face.x + right_face.x) / 2

            gaze_diff = abs(eye_center - face_center)

            eye_score = 100 - (gaze_diff * 2500)
            eye_score = clamp(eye_score)

            # ================= POSTURE =================
            face_height = abs(forehead.y - chin.y)
            face_width = abs(left_face.x - right_face.x)

            ratio = face_height / (face_width + 1e-6)

            pose_score = 100 - abs((ratio - 1.15) * 120)
            pose_score = clamp(pose_score)

            # ================= STABILITY =================
            if prev_nose is not None:

                movement = (
                    abs(nose.x - prev_nose.x) +
                    abs(nose.y - prev_nose.y)
                )

                stability_score = 100 - (movement * 3000)
                stability_score = clamp(stability_score)

            prev_nose = nose

        # ================= SAVE =================
        st.session_state.eye_hist.append(eye_score)
        st.session_state.pose_hist.append(pose_score)
        st.session_state.stability_hist.append(stability_score)

        if len(st.session_state.eye_hist) > 50:
            st.session_state.eye_hist.pop(0)
            st.session_state.pose_hist.pop(0)
            st.session_state.stability_hist.pop(0)

        # ================= FINAL VALUES =================
        eye = int(np.mean(st.session_state.eye_hist))
        pose = int(np.mean(st.session_state.pose_hist))
        stability = int(np.mean(st.session_state.stability_hist))

        confidence = int(
            (eye * 0.45) +
            (pose * 0.35) +
            (stability * 0.20)
        )

        # ================= VIDEO =================
        video_box.image(frame, channels="BGR")

        # ================= ANALYSIS PANEL =================
        with panel.container():

            st.markdown("## 📊 Live Analysis")

            st.metric("👁 Eye Contact", f"{eye}%")
            st.metric("🧍 Posture", f"{pose}%")
            st.metric("🎯 Stability", f"{stability}%")
            st.metric("😐 Confidence", f"{confidence}%")

            st.markdown("---")

            st.subheader("🧠 AI Feedback")

            if eye < 60:
                st.warning("Eye contact is weak — look at camera consistently")
            elif eye < 80:
                st.info("Eye contact is good but improve consistency")
            else:
                st.success("Strong eye contact")

            if pose < 60:
                st.warning("Posture is not upright — adjust sitting position")
            elif pose < 80:
                st.info("Posture is acceptable")
            else:
                st.success("Excellent posture")

            if stability < 60:
                st.warning("Too much movement detected")
            elif stability < 80:
                st.info("Minor movement detected")
            else:
                st.success("Very stable presence")

        time.sleep(0.05)

    cap.release()

if st.session_state.report:

    eye = int(np.mean(st.session_state.eye_hist)) if st.session_state.eye_hist else 0
    pose = int(np.mean(st.session_state.pose_hist)) if st.session_state.pose_hist else 0
    stability = int(np.mean(st.session_state.stability_hist)) if st.session_state.stability_hist else 0

    confidence = int(
        (eye * 0.45) +
        (pose * 0.35) +
        (stability * 0.20)
    )

    st.markdown("---")

    st.subheader("📊 Final Interview Report")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👁 Eye Contact", f"{eye}%")
    c2.metric("🧍 Posture", f"{pose}%")
    c3.metric("🎯 Stability", f"{stability}%")
    c4.metric("😐 Confidence", f"{confidence}%")

    st.markdown("---")

    st.subheader("🧠 Final Feedback")

    if confidence >= 80:
        st.success("""
🔥 Excellent Interview Readiness

You demonstrated strong confidence, professional body language, and a positive on-screen presence throughout the session.

Key strengths observed:
• Consistent engagement and attention.
• Professional posture and composure.
• Good control of movements and distractions.
• Confident overall interview appearance.

Recommendation:
Continue practicing answer delivery, communication clarity, and technical preparation to maximize your interview performance.
""")
    elif confidence >= 60:
        st.warning("""
⚠ Good Foundation With Room For Improvement

Your overall performance indicates a reasonable level of interview readiness, but a few improvements could significantly strengthen your professional presence.

Areas to focus on:
• Maintain more consistent eye contact.
• Improve posture during longer responses.
• Reduce unnecessary movements.
• Project confidence through steady body language.

Recommendation:
Practice mock interviews regularly and focus on maintaining engagement from the beginning to the end of the conversation.
""")
    else:
        st.error("""
❌ Interview Readiness Needs Improvement

Your current body language patterns suggest that additional practice may be beneficial before attending important interviews.

Key improvement areas:
• Eye contact consistency.
• Upright and confident posture.
• Stable and controlled movements.
• Stronger overall professional presence.

Recommendation:
Spend time practicing in front of a webcam, review your recordings, and focus on developing confident non-verbal communication habits alongside your interview answers.
""")

st.markdown(
    "<div class='footer'>AI Powered Interview Intelligence System</div>",
    unsafe_allow_html=True
)
