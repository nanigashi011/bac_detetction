'''

import os
import cv2
import base64
import json
import io
import time
import tempfile
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Use the older OpenAI package interface
import openai

# Set page configuration
st.set_page_config(
    page_title="Waste Collection Analysis Platform",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1976d2, #26a69a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1976d2;
    }
    
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .metric-card {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1976d2;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0 0;
        gap: 1;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1976d2;
        color: white;
    }
    
    div.block-container {
        padding-top: 2rem;
    }
    
    .step-card {
        border-left: 3px solid #1976d2;
        padding-left: 20px;
        margin-bottom: 20px;
    }
    
    .step-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1976d2;
        margin-bottom: 5px;
    }
    
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Functions for video processing
def extract_frames(video_file, fps_extract=1, max_images=None):
    """Extract frames from a video file at specified FPS rate"""
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_file.read())
        temp_path = temp_file.name
    
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        os.unlink(temp_path)
        raise RuntimeError(f"Unable to open video")
    
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(vid_fps / fps_extract) if vid_fps > fps_extract else 1
    
    images = []
    count = 0
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = min(int(100 * frame_idx / total_frames), 100)
            progress_bar.progress(progress)
            status_text.text(f"Extracting frames: {progress}%")
            
            if frame_idx % step == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                images.append(pil_img)
                count += 1
                
                if max_images and count >= max_images:
                    break
            
            frame_idx += 1
    finally:
        cap.release()
        os.unlink(temp_path)  # Clean up the temp file
        
    status_text.text(f"Extracted {len(images)} frames")
    progress_bar.empty()
    
    return images

def pil_to_base64(im):
    """Convert PIL Image to base64 string for API transmission"""
    buffered = io.BytesIO()
    im.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def build_prompt():
    """Build the system prompt for the OpenAI API"""
    return (
        "You are a computer vision analyst specialized in waste collection scenes. "
        "You will receive several frames extracted from a short video (20 seconds). "
        "Analyze them in temporal order and output a structured JSON with:\n"
        "  total_bacs, small_bacs, large_bacs, plastic_bacs, metal_bacs, empty_bacs, full_bacs, broken_bacs, "
        "emptying_events, simultaneous_emptying, refill_events, notes.\n"
        "Respond strictly in JSON only — no explanations."
    )

def analyze_video_frames(frames, api_key):
    """Analyze video frames using OpenAI API"""
    if not frames:
        st.error("No frames to analyze.")
        return None
    
    # Convert images to base64
    images_b64 = [pil_to_base64(frame) for frame in frames]
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Preparing to analyze frames...")
    progress_bar.progress(10)
    
    # Build content for API request using old-style API
    system_prompt = build_prompt()
    
    # Format the content for the API request
    progress_bar.progress(30)
    status_text.text("Sending frames to AI for analysis...")
    
    try:
        # For older versions of OpenAI
        openai.api_key = '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'
        
        # Create a list of messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze the following video frames as instructed."}
            ]}
        ]
        
        # Add each image to the user message content
        for img_b64 in images_b64:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": img_b64}
            })
        
        # Call OpenAI API with the older style API
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",  # Use vision model for older API
            messages=messages,
            max_tokens=800
        )
        
        # Get content from response
        result = response["choices"][0]["message"]["content"]
        
        progress_bar.progress(90)
        status_text.text("Processing AI response...")
        
        # Parse result
        try:
            result_json = json.loads(result)
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            time.sleep(0.5)  # Give user time to see the "complete" message
            status_text.empty()
            progress_bar.empty()
            return result_json
        except json.JSONDecodeError:
            st.error("Error: The AI response was not valid JSON.")
            st.code(result, language="text")
            progress_bar.empty()
            status_text.empty()
            return None
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error during analysis: {str(e)}")
        return None

# Session state initialization
if 'frames' not in st.session_state:
    st.session_state.frames = []
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'api_key_saved' not in st.session_state:
    st.session_state.api_key_saved = False

# Main App Layout
def main():
    # App Header
    st.markdown('<h1 class="main-title">Waste Collection Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('#### AI-Powered Video Analysis for Waste Management Optimization')
    
    # Create tabs for different sections
    tabs = st.tabs(["📋 Project Overview", "📤 Upload & Analyze", "📊 Results Dashboard"])
    
    # Project Overview Tab
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-title">Waste Collection Intelligence Platform</h2>', unsafe_allow_html=True)
            st.markdown("""
            Our platform transforms waste collection footage into actionable insights using advanced AI technology.
            
            This tool helps waste management companies optimize their operations by providing detailed analytics on:
            - Container counts and types
            - Container fullness status
            - Emptying events and efficiency
            - Equipment condition monitoring
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-title">How It Works</h2>', unsafe_allow_html=True)
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.markdown("""
                <div class="step-card">
                    <div class="step-title">1. Upload Video</div>
                    <p>Upload MP4 footage of waste collection activities</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col_b:
                st.markdown("""
                <div class="step-card">
                    <div class="step-title">2. Frame Extraction</div>
                    <p>System extracts key frames at specified intervals</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col_c:
                st.markdown("""
                <div class="step-card">
                    <div class="step-title">3. AI Analysis</div>
                    <p>GPT-4 Vision analyzes frames to identify waste containers</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col_d:
                st.markdown("""
                <div class="step-card">
                    <div class="step-title">4. View Insights</div>
                    <p>Interactive dashboard presents key metrics and findings</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-title">Key Benefits</h2>', unsafe_allow_html=True)
            st.markdown("""
            - **Operational Efficiency**: Reduce costs by 15-30% through optimized collection routes and resource allocation
            
            - **Data-Driven Decisions**: Make informed decisions based on quantitative analysis rather than guesswork
            
            - **Equipment Management**: Track container status and maintenance needs automatically
            
            - **Environmental Impact**: Reduce carbon footprint through optimized routes and improved recycling rates
            
            - **Stakeholder Reporting**: Generate comprehensive reports with visual analytics
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload & Analyze Tab
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-title">Upload Waste Collection Video</h2>', unsafe_allow_html=True)
        
        # API Key Input in sidebar
        with st.sidebar:
            st.markdown("### OpenAI API Key")
            
            # Check if API key is already saved
            if st.session_state.api_key_saved:
                st.success("API Key saved ✓")
                if st.button("Change API Key"):
                    st.session_state.api_key_saved = False
                    st.experimental_user()
            else:
                api_key = st.text_input(
                    "Enter your OpenAI API Key", 
                    type="password",
                    help="Your API key is required for image analysis and is not stored permanently."
                )
                if st.button("Save API Key"):
                    if api_key and api_key.startswith("sk-"):
                        st.session_state.api_key = api_key
                        st.session_state.api_key_saved = True
                        st.success("API Key saved successfully!")
                        st.experimental_user()
                    else:
                        st.error("Please enter a valid OpenAI API key starting with 'sk-'")
        
        # Video upload settings
        col1, col2 = st.columns(2)
        
        with col1:
            fps_extract = st.selectbox(
                "Frames Per Second (FPS)", 
                options=[0.5, 1, 2, 5],
                index=1,
                format_func=lambda x: f"{x} (1 frame per {1/x:.0f} second)" if x < 1 else f"{x} (frames per second)",
                help="Higher FPS provides more detail but uses more API credits"
            )
            
        with col2:
            max_frames = st.selectbox(
                "Maximum Frames to Extract",
                options=[5, 10, 15, 20],
                index=1,
                help="More frames may provide better analysis but will use more API credits"
            )
        
        # Video upload
        uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4"])
        
        if uploaded_file is not None:
            # Display video preview
            st.video(uploaded_file)
            
            # Process button
            if st.button("Process Video", type="primary", use_container_width=True):
                if not st.session_state.api_key_saved:
                    st.error("Please save your OpenAI API key first.")
                else:
                    with st.spinner("Extracting frames from video..."):
                        # Extract frames
                        try:
                            frames = extract_frames(uploaded_file, fps_extract, max_frames)
                            st.session_state.frames = frames
                            
                            if frames:
                                # Show preview of extracted frames
                                st.markdown("<h3 style='margin-top: 20px;'>Extracted Frames</h3>", unsafe_allow_html=True)
                                cols = st.columns(min(5, len(frames)))
                                for i, (col, frame) in enumerate(zip(cols, frames[:5])):
                                    with col:
                                        st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
                                
                                # If more than 5 frames, show a note
                                if len(frames) > 5:
                                    st.info(f"Showing 5 of {len(frames)} frames. All frames will be used for analysis.")
                                
                                # Analyze frames
                                with st.spinner("Analyzing frames with AI..."):
                                    analysis_result = analyze_video_frames(frames, st.session_state.api_key)
                                    if analysis_result:
                                        st.session_state.analysis_result = analysis_result
                                        st.success("Analysis complete! View results in the Results Dashboard tab.")
                        except Exception as e:
                            st.error(f"Error processing video: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Results Dashboard Tab
    with tabs[2]:
        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            frames = st.session_state.frames
            
            # Display metrics in cards
            st.markdown('<h2 class="sub-title">Analysis Results</h2>', unsafe_allow_html=True)
            
            # Display metric cards in grid layout
            metrics = [
                {"key": "total_bacs", "label": "Total Containers", "icon": "🗑️"},
                {"key": "small_bacs", "label": "Small Containers", "icon": "🧺"},
                {"key": "large_bacs", "label": "Large Containers", "icon": "🗑️"},
                {"key": "empty_bacs", "label": "Empty Containers", "icon": "⚪"},
                {"key": "full_bacs", "label": "Full Containers", "icon": "⚫"},
                {"key": "emptying_events", "label": "Emptying Events", "icon": "🔄"}
            ]
            
            # Create columns for metrics
            cols = st.columns(3)
            for i, metric in enumerate(metrics):
                with cols[i % 3]:
                    value = result.get(metric["key"], 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 2rem;">{metric["icon"]}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-label">{metric["label"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Results tabs for different visualizations
            result_tabs = st.tabs(["📊 Dashboard", "🖼️ Frames Gallery", "📝 JSON Data"])
            
            # Dashboard tab with visualizations
            with result_tabs[0]:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create data for the pie chart
                    if all(k in result for k in ["small_bacs", "large_bacs"]):
                        container_types = {
                            "Small Containers": result.get("small_bacs", 0),
                            "Large Containers": result.get("large_bacs", 0)
                        }
                        
                        if "plastic_bacs" in result and "metal_bacs" in result:
                            container_materials = {
                                "Plastic Containers": result.get("plastic_bacs", 0),
                                "Metal Containers": result.get("metal_bacs", 0)
                            }
                            
                            # Create a figure with subplots
                            fig = go.Figure()
                            
                            # Add traces for container types
                            labels = list(container_types.keys())
                            values = list(container_types.values())
                            
                            fig.add_trace(go.Pie(
                                labels=labels,
                                values=values,
                                name="Container Types",
                                domain={"x": [0, 0.45]},
                                title="Container Types",
                                marker=dict(colors=['rgba(25, 118, 210, 0.8)', 'rgba(38, 166, 154, 0.8)']),
                            ))
                            
                            # Add traces for container materials
                            labels = list(container_materials.keys())
                            values = list(container_materials.values())
                            
                            fig.add_trace(go.Pie(
                                labels=labels,
                                values=values,
                                name="Container Materials",
                                domain={"x": [0.55, 1]},
                                title="Container Materials",
                                marker=dict(colors=['rgba(255, 143, 0, 0.8)', 'rgba(124, 77, 255, 0.8)']),
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                title_text="Container Distribution",
                                height=350,
                                margin=dict(t=50, b=20, l=20, r=20),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Create another chart for container fullness
                    if all(k in result for k in ["empty_bacs", "full_bacs"]):
                        container_status = {
                            "Empty Containers": result.get("empty_bacs", 0),
                            "Full Containers": result.get("full_bacs", 0)
                        }
                        
                        if "broken_bacs" in result:
                            container_status["Broken Containers"] = result.get("broken_bacs", 0)
                        
                        df = pd.DataFrame({
                            "Status": list(container_status.keys()),
                            "Count": list(container_status.values())
                        })
                        
                        fig = px.bar(
                            df, 
                            x="Status", 
                            y="Count", 
                            color="Status",
                            color_discrete_map={
                                "Empty Containers": "rgba(38, 166, 154, 0.8)",
                                "Full Containers": "rgba(25, 118, 210, 0.8)",
                                "Broken Containers": "rgba(244, 67, 54, 0.8)"
                            },
                            title="Container Status"
                        )
                        
                        fig.update_layout(
                            height=350,
                            margin=dict(t=50, b=20, l=20, r=20),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Event Analysis")
                    
                    event_metrics = [
                        {"key": "emptying_events", "label": "Emptying Events", "icon": "🔄"},
                        {"key": "simultaneous_emptying", "label": "Simultaneous Emptying", "icon": "⚡"},
                        {"key": "refill_events", "label": "Refill Events", "icon": "🔋"}
                    ]
                    
                    for metric in event_metrics:
                        value = result.get(metric["key"], 0)
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 15px;">
                            <div style="font-size: 1.5rem; margin-right: 10px;">{metric["icon"]}</div>
                            <div>
                                <div style="font-size: 1.4rem; font-weight: 600; color: #1976d2;">{value}</div>
                                <div style="color: #6c757d; font-size: 0.9rem;">{metric["label"]}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Analysis Notes")
                    
                    if "notes" in result and result["notes"]:
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; font-style: italic;">
                            {result["notes"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="color: #6c757d; font-style: italic;">
                            No additional notes from the analysis.
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Download buttons
                st.markdown("### Export Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    json_data = json.dumps(result, indent=2)
                    st.download_button(
                        label="Download JSON Data",
                        data=json_data,
                        file_name="waste_collection_analysis.json",
                        mime="application/json",
                    )
                
                with col2:
                    # Convert to CSV for Excel compatibility
                    csv_data = io.StringIO()
                    for key, value in result.items():
                        if key != "notes":  # Handle notes separately
                            csv_data.write(f"{key},{value}\n")
                    
                    st.download_button(
                        label="Download CSV (Excel)",
                        data=csv_data.getvalue(),
                        file_name="waste_collection_analysis.csv",
                        mime="text/csv",
                    )
            
            # Frames Gallery tab
            with result_tabs[1]:
                if frames:
                    # Create a grid of images
                    cols_per_row = 4
                    for i in range(0, len(frames), cols_per_row):
                        row_frames = frames[i:i+cols_per_row]
                        cols = st.columns(cols_per_row)
                        
                        for j, frame in enumerate(row_frames):
                            with cols[j]:
                                st.image(frame, caption=f"Frame {i+j+1}", use_column_width=True)
                else:
                    st.info("No frames available.")
            
            # JSON Data tab
            with result_tabs[2]:
                st.json(result)
                
        else:
            st.info("No analysis results yet. Please upload a video in the 'Upload & Analyze' tab.")

# Run the main function
if __name__ == "__main__":
    main()

'''




    #### best quality code

import os
import cv2
import base64
import json
import io
import time
import tempfile
from PIL import Image
import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Waste Collection Video Analysis",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match the blue design in the screenshot
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    /* Streamlit container styling */
    .css-18e3th9 {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #2c3e50;
        font-size: 2.3rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    /* Selector label styling */
    .stSelectbox label, .stSlider label {
        color: #2c3e50;
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: #f1f3f5;
        border-radius: 5px;
        padding: 15px;
        border: 1px dashed #ccc;
    }
    
    /* Help icon styling */
    .stSelectbox [data-testid="stToolbar"] {
        color: #6c757d;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer {
        visibility: hidden;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #0069d9;
    }
    
    /* Video styling */
    [data-testid="stVideo"] {
        border-radius: 4px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Simple app layout matching the screenshot
st.title("Upload Waste Collection Video")

# API Key Input in sidebar
with st.sidebar:
    st.header("API Configuration")
    
    # Check if API key is already saved
    if 'api_key_saved' not in st.session_state:
        st.session_state.api_key_saved = False
        
    if st.session_state.api_key_saved:
        st.success("✓ OpenAI API Key saved")
        if st.button("Change API Key"):
            st.session_state.api_key_saved = False
            st.rerun()
    else:
        st.info("Enter your OpenAI API Key")
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Your API key is needed for image analysis"
        )
        if st.button("Save Key"):
            if api_key and api_key.startswith("sk-"):
                st.session_state.api_key = api_key
                st.session_state.api_key_saved = True
                st.success("API Key saved successfully!")
                st.rerun()
            else:
                st.error("Please enter a valid OpenAI API Key")

# Two columns for parameters
col1, col2 = st.columns(2)

with col1:
    fps_extract = st.selectbox(
        "Frames Per Second (FPS)",
        options=[0.5, 1, 2, 5],
        index=1,
        format_func=lambda x: f"{x} (frames per second)" if x >= 1 else f"{x} (1 frame/{1/x:.0f}s)",
        help="Select how many frames to extract per second of video"
    )

with col2:
    max_frames = st.selectbox(
        "Maximum Frames to Extract",
        options=[5, 10, 15, 20],
        index=1,
        help="Maximum number of frames to extract for analysis"
    )

# Video upload section
uploaded_file = st.file_uploader(
    "Upload MP4 Video",
    type=["mp4"],
    help="Limit 200MB per file • MP4, MPEG4"
)

# Functions for video processing
def extract_frames(video_file, fps_extract=1, max_images=None):
    """Extract frames from a video file at specified FPS rate"""
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_file.read())
        temp_path = temp_file.name
    
    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        os.unlink(temp_path)
        raise RuntimeError(f"Could not open the video")
    
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(vid_fps / fps_extract) if vid_fps > fps_extract else 1
    
    images = []
    count = 0
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = min(int(100 * frame_idx / total_frames), 100)
            progress_bar.progress(progress)
            status_text.text(f"Extracting frames: {progress}%")
            
            if frame_idx % step == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                images.append(pil_img)
                count += 1
                
                if max_images and count >= max_images:
                    break
            
            frame_idx += 1
    finally:
        cap.release()
        os.unlink(temp_path)  # Clean up the temp file
        
    status_text.text(f"{len(images)} frames successfully extracted")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    return images

def pil_to_base64(im):
    """Convert PIL Image to base64 string for API transmission"""
    buffered = io.BytesIO()
    im.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def analyze_video_frames(frames, api_key):
    """Analyze video frames using OpenAI API"""
    if not frames:
        st.error("No frames to analyze.")
        return None
    
    # Convert images to base64
    images_b64 = [pil_to_base64(frame) for frame in frames]
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Preparing analysis...")
    progress_bar.progress(10)
    
    # Build system prompt for API request
    system_prompt = (
        "You are a computer vision analyst specialized in waste collection scenes. "
        "You will receive several frames extracted from a short video (20 seconds). "
        "Analyze them in temporal order and output a structured JSON with:\n"
        "  total_bacs, small_bacs, large_bacs, plastic_bacs, metal_bacs, empty_bacs, full_bacs, broken_bacs, "
        "emptying_events, simultaneous_emptying, refill_events, notes.\n"
        "Respond strictly in JSON only — no explanations."
    )
    
    progress_bar.progress(30)
    status_text.text("Sending images to AI for analysis...")
    
    try:
        # Using the OpenAI API (v1.0+ syntax)
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Create a list of messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze the following video frames as instructed."}
            ]}
        ]
        
        # Add each image to the user message content
        for img_b64 in images_b64:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": img_b64}
            })
        
        # Call OpenAI API with GPT-4 Vision capabilities
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=800
        )
        
        # Get content from response
        result = response.choices[0].message.content
        
        progress_bar.progress(90)
        status_text.text("Processing analysis results...")
        
        # Parse result
        try:
            # Check if the response contains markdown code block indicators
            if '```' in result:
                # Try to extract JSON from within code blocks
                import re
                json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', result)
                if json_blocks:
                    # Use the first extracted JSON block
                    cleaned_result = json_blocks[0].strip()
                else:
                    # If no blocks found with regex, try simple splitting
                    parts = result.split('```')
                    if len(parts) >= 3:  # Should have at least 3 parts if proper markdown
                        # The middle part (index 1) should contain the JSON
                        cleaned_result = parts[1]
                        # Remove the "json" language identifier if it exists
                        if cleaned_result.startswith('json'):
                            cleaned_result = cleaned_result[4:].strip()
                    else:
                        cleaned_result = result
            else:
                cleaned_result = result
            
            # Show the cleaned result for debugging
            st.text("Cleaned Response :")
            st.code(cleaned_result, language="json")
            
            # Try to parse the JSON
            result_json = json.loads(cleaned_result)
            progress_bar.progress(100)
            status_text.text("Analysis completed!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # If we successfully parsed the JSON, remove the debug output
            st.empty()  # Clear the "Raw API Response" text
            st.empty()  # Clear the raw response code block
            st.empty()  # Clear the "Cleaned Response for Parsing" text
            st.empty()  # Clear the cleaned response code block
            
            return result_json
            
        except json.JSONDecodeError as e:
            # If normal parsing fails, try a more aggressive approach to salvage the JSON
            try:
                # Find anything that looks like JSON between curly braces
                import re
                json_pattern = re.compile(r'\{[\s\S]*\}')
                match = json_pattern.search(result)
                if match:
                    potential_json = match.group(0)
                    # Try to parse this extracted JSON
                    fallback_result = json.loads(potential_json)
                    st.warning("Used fallback JSON extraction method. Results may not be complete.")
                    progress_bar.empty()
                    status_text.empty()
                    return fallback_result
            except:
                # If all approaches fail, give up
                pass
                
            st.error(f"Error: AI response is not valid JSON. Details: {str(e)}")
            progress_bar.empty()
            status_text.empty()
            
            # As a last resort, create a simple JSON from what we can see
            try:
                # Extract key-value pairs using regex
                import re
                pairs = re.findall(r'"([^"]+)":\s*([^,\n]+)', result)
                if pairs:
                    manual_json = {}
                    for key, value in pairs:
                        try:
                            # Try to convert to appropriate type
                            if value.strip().isdigit():
                                manual_json[key] = int(value.strip())
                            elif value.strip() in ["true", "false"]:
                                manual_json[key] = value.strip() == "true"
                            elif value.strip().startswith('"') and value.strip().endswith('"'):
                                manual_json[key] = value.strip().strip('"')
                            else:
                                manual_json[key] = value.strip()
                        except:
                            manual_json[key] = value.strip()
                    
                    if manual_json:
                        st.warning("Created a partial JSON from the response. Some data may be missing or incorrect.")
                        return manual_json
            except:
                pass
            
            return None
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error during analysis: {str(e)}")
        return None
# Session state initialization
if 'frames' not in st.session_state:
    st.session_state.frames = []
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Display video if uploaded
if uploaded_file is not None:
    # Show video preview
    st.video(uploaded_file)
    
    # Add analyze button
    if st.button("Analyze Video", type="primary"):
        if not st.session_state.api_key_saved:
            st.error("Please save your OpenAI API Key in the sidebar first.")
        else:
            with st.spinner("Processing video..."):
                # Extract frames
                frames = extract_frames(uploaded_file, fps_extract, max_frames)
                st.session_state.frames = frames
                
                if frames:
                    # Show preview of extracted frames
                    st.subheader("Extracted Frames")
                    
                    # Display frames in a grid
                    cols = st.columns(min(5, len(frames)))
                    for i, (col, frame) in enumerate(zip(cols, frames[:5])):
                        with col:
                            st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
                    
                    # If more than 5 frames, show a note
                    if len(frames) > 5:
                        st.info(f"Showing 5 frames out of {len(frames)}. All frames will be used for analysis.")
                    
                    # Use the OpenAI API key
                    analysis_result = analyze_video_frames(frames, st.session_state.api_key)
                    
                    if analysis_result:
                        st.session_state.analysis_result = analysis_result
                        
                        # Display results in tabs
                        tabs = st.tabs(["Summary", "Detailed Data", "JSON Data"])
                        
                        with tabs[0]:
                            # Summary metrics in a grid
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Containers", analysis_result.get("total_bacs", 0))
                                st.metric("Small Containers", analysis_result.get("small_bacs", 0))
                            with col2:
                                st.metric("Large Containers", analysis_result.get("large_bacs", 0))
                                st.metric("Empty Containers", analysis_result.get("empty_bacs", 0))
                            with col3:
                                st.metric("Full Containers", analysis_result.get("full_bacs", 0))
                                st.metric("Emptying Events", analysis_result.get("emptying_events", 0))
                            
                            # Notes
                            if "notes" in analysis_result and analysis_result["notes"]:
                                st.subheader("Analysis Notes")
                                st.info(analysis_result["notes"])
                        
                        with tabs[1]:
                            # Create a DataFrame for better viewing
                            data = {}
                            for key, value in analysis_result.items():
                                if key != "notes":  # Handle notes separately
                                    data[key] = [value]
                            
                            if data:
                                df = pd.DataFrame(data)
                                st.dataframe(df.T.rename(columns={0: "Value"}), use_container_width=True)
                        
                        with tabs[2]:
                            # Raw JSON
                            st.json(analysis_result)
                        
                        # Download buttons
                        st.subheader("Export Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            json_data = json.dumps(analysis_result, indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=json_data,
                                file_name="waste_collection_analysis.json",
                                mime="application/json",
                            )
                        
                        with col2:
                            # Convert to CSV
                            csv_data = io.StringIO()
                            for key, value in analysis_result.items():
                                if key != "notes":  # Handle notes separately
                                    csv_data.write(f"{key},{value}\n")
                            
                            st.download_button(
                                label="Export to CSV",
                                data=csv_data.getvalue(),
                                file_name="waste_collection_analysis.csv",
                                mime="text/csv",
                            )



