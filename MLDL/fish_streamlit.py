"""
Fish Species Classification Streamlit Application

This application uses trained PyTorch models to classify fish species from uploaded images.


"""

import os
import pickle
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time
import io
from scipy.io import wavfile
import sounddevice as sd

# Set page configuration
st.set_page_config(
    page_title="Fish Species Classifier",
    page_icon="🐟",
    layout="wide"
)

# Define color theme
COLORS = {
    "primary": "#2C4292",
    "secondary": "#3D5DB3",
    "accent": "#5A78C7",
    "light": "#E8EDF9",
    "text": "#1A2A56",
    "success": "#4CAF50",
    "info": "#2196F3",
    "warning": "#FF9800"
}

def get_display_name(prediction_label):
    """
    Maps the model's prediction label to a display-friendly name.
    
    Args:
        prediction_label (str): The raw prediction label from the model
        
    Returns:
        str: A display-friendly name for the fish species
    """
    # Define mapping from model labels to display names
    label_mapping = {
        "fish_sea_food_black_sea_sprat": "Black Sea Sprat",
        "fish_sea_food_gilt_head_bream": "Gilthead Bream",
        "fish_sea_food_hourse_mackerel": "Horse Mackerel",
        "fish_sea_food_red_mullet": "Red Mullet",
        "fish_sea_food_red_sea_bream": "Red Sea Bream",
        "fish_sea_food_sea_bass": "Sea Bass",
        "fish_sea_food_shrimp": "Shrimp",
        "fish_sea_food_striped_red_mullet": "Striped Red Mullet",
        "fish_sea_food_trout": "Trout",
        # Add fallbacks for different label formats
        "black_sea_sprat": "Black Sea Sprat",
        "gilt_head_bream": "Gilthead Bream",
        "horse_mackerel": "Horse Mackerel",
        "red_mullet": "Red Mullet",
        "red_sea_bream": "Red Sea Bream",
        "sea_bass": "Sea Bass",
        "shrimp": "Shrimp",
        "striped_red_mullet": "Striped Red Mullet",
        "trout": "Trout",
        # Add even simpler fallbacks
        "Black Sea Sprat": "Black Sea Sprat",
        "Gilthead Bream": "Gilthead Bream",
        "Horse Mackerel": "Horse Mackerel",
        "Red Mullet": "Red Mullet",
        "Red Sea Bream": "Red Sea Bream",
        "Sea Bass": "Sea Bass",
        "Shrimp": "Shrimp",
        "Striped Red Mullet": "Striped Red Mullet",
        "Trout": "Trout"
    }
    
    # Convert to lowercase and remove spaces for more robust matching
    clean_label = prediction_label.lower().replace(" ", "_")
    
    # Try to find the label in our mapping
    for key, value in label_mapping.items():
        if clean_label in key.lower() or key.lower() in clean_label:
            return value
    
    # If no match found, return the original label
    return prediction_label

def add_bg_from_pattern():
    """
    Add a patterned background to the app.
    """
    # Create a pattern with dots
    pattern = f"""
    <style>
    .stApp {{
        background-color: {COLORS["light"]};
        background-image: radial-gradient({COLORS["accent"]}30 1px, transparent 1px);
        background-size: 20px 20px;
    }}
    </style>
    """
    st.markdown(pattern, unsafe_allow_html=True)

def create_enhanced_text_style():
    """
    Creates enhanced text styling with 3D effects and shadows for all plain text elements.
    
    Returns:
        str: HTML/CSS for enhanced text styling
    """
    return f"""
    <style>
    /* Enhanced text styling for all plain text */
    .stApp p, .stApp span:not(.animated-title), .stApp label, .stApp div.stMarkdown p {{
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }}
    
    /* Enhanced styling for success boxes */
    .stAlert.stSuccess p, div[style*="background-color: #d1e7dd"] p {{
        color: {COLORS['primary']} !important;
        text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.9), 0 0 2px rgba(0, 0, 0, 0.05);
        font-weight: 700 !important;
        letter-spacing: 0.03em;
    }}
    
    /* Enhanced styling for info boxes */
    .stAlert.stInfo p, div[style*="background-color: #cfe2ff"] p {{
        color: {COLORS['primary']} !important;
        text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.9), 0 0 2px rgba(0, 0, 0, 0.05);
        font-weight: 700 !important;
        letter-spacing: 0.03em;
    }}
    
    /* Make all alert boxes pop with enhanced shadows */
    .stAlert, div[style*="background-color: #d1e7dd"], div[style*="background-color: #cfe2ff"] {{
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.12), 
                    inset 0 -3px 0 rgba(0, 0, 0, 0.08),
                    0 -1px 0 rgba(255, 255, 255, 0.7) inset !important;
        transform: translateZ(0);
    }}
    
    /* Make file uploader label and other form labels bolder */
    .stFileUploader label, .stSelectbox label, [data-baseweb="form-control-label"] {{
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
    }}
    
    /* Make checkbox text bolder */
    .stCheckbox label span p {{
        font-weight: 700 !important;
    }}
    
    /* Ensure sidebar text is white with proper 3D effect - more comprehensive selectors */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="radio"] span,
    [data-testid="stSidebar"] [data-baseweb="radio"] div,
    [data-testid="stSidebar"] [data-baseweb="checkbox"] span,
    [data-testid="stSidebar"] [data-baseweb="checkbox"] div,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stExpander details summary {{
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3), 0 0 5px rgba(0, 0, 0, 0.1) !important;
        font-weight: 500 !important;
    }}
    
    /* Ensure sidebar headings are white with enhanced 3D effect */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {{
        color: white !important;
        text-shadow: 2px 2px 3px rgba(0, 0, 0, 0.4), 0 0 7px rgba(0, 0, 0, 0.1) !important;
        font-weight: 700 !important;
    }}
    
    /* Ensure progress text during classification is blue and visible */
    .stProgress div, 
    .stProgress p,
    .stProgress span,
    .stProgress label,
    div[data-testid="stStatusWidget"],
    div[data-testid="stStatusWidget"] span,
    div[data-testid="stStatusWidget"] p {{
        color: {COLORS['primary']} !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1) !important;
        font-weight: 600 !important;
    }}
    
    /* Ensure empty elements used for status text are properly colored */
    [data-testid="stText"],
    [data-testid="stText"] p,
    [data-testid="stText"] div {{
        color: {COLORS['primary']} !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1) !important;
        font-weight: 600 !important;
    }}
    </style>
    """

def apply_custom_styles():
    """
    Apply custom CSS styles to the Streamlit app.
    """
    # Original styles
    styles = f"""
    <style>
    /* Text styling with shadow */
    h1, h2, h3 {{
        color: {COLORS["primary"]};
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        padding: 10px;
        border-radius: 5px;
    }}
    
    /* Force all text in main content to be blue */
    .stApp {{
        color: {COLORS["primary"]} !important;
    }}
    
    /* Make search box text white */
    .stSearchBox input {{
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
    }}
    
    /* Make search box placeholder text white with lower opacity */
    .stSearchBox input::placeholder {{
        color: rgba(255, 255, 255, 0.7) !important;
    }}
    
    /* Make sure sidebar text stays white - more specific and !important selectors */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] div.stMarkdown p,
    [data-testid="stSidebar"] div.element-container div.stMarkdown p {{
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }}
    
    /* Target sidebar headings specifically */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {{
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }}
    
    /* Override any other selectors that might be turning sidebar text blue */
    .stApp [data-testid="stSidebar"] [data-baseweb="checkbox"] span,
    .stApp [data-testid="stSidebar"] [data-baseweb="radio"] span,
    .stApp [data-testid="stSidebar"] [data-baseweb="checkbox"] p,
    .stApp [data-testid="stSidebar"] [data-baseweb="radio"] p,
    .stApp [data-testid="stSidebar"] .stCheckbox label span p,
    .stApp [data-testid="stSidebar"] .stRadio label,
    .stApp [data-testid="stSidebar"] .stExpander details summary p,
    .stApp [data-testid="stSidebar"] .stExpander details summary span,
    .stApp [data-testid="stSidebar"] .stExpander details div p,
    .stApp [data-testid="stSidebar"] .stSelectbox label,
    .stApp [data-testid="stSidebar"] .stSelectbox div,
    .stApp [data-testid="stSidebar"] .stRadio div,
    .stApp [data-testid="stSidebar"] .stRadio label span div p {{
        color: white !important;
    }}
    
    /* Target sidebar list items */
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] ul,
    [data-testid="stSidebar"] ol {{
        color: white !important;
    }}
    
    /* Fix for file uploader label text color */
    .stFileUploader label span, .stFileUploader div[data-testid="stMarkdownContainer"] p {{
        color: {COLORS["primary"]} !important;
        font-weight: 600 !important;
    }}
    
    /* Fix for checkbox label text color */
    .stCheckbox label span p {{
        color: {COLORS["primary"]} !important;
        font-weight: 600 !important;
    }}
    
    /* Specifically target success messages and checkboxes */
    .element-container .stAlert, 
    .stCheckbox label span p,
    .stSuccess {{
        color: {COLORS["primary"]} !important;
    }}
    
    /* Make sure success and info boxes have blue text */
    .stAlert.stSuccess, .stAlert.stInfo {{
        color: {COLORS["primary"]} !important;
    }}
    
    /* Target the text inside success and info boxes specifically */
    .stAlert.stSuccess p, .stAlert.stInfo p {{
        color: {COLORS["primary"]} !important;
    }}
    
    /* Card-like containers with shadow */
    .css-1r6slb0, .css-keje6w, .css-1oe6wy4 {{
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        background-color: white;
        margin: 10px 0;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {COLORS["primary"]};
        background-image: linear-gradient(45deg, {COLORS["primary"]}, {COLORS["secondary"]});
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: {COLORS["primary"]};
        color: white !important;
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }}
    
    /* Ensure button text is white with proper shadow */
    .stButton>button p,
    .stButton>button span,
    .stButton>button div {{
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
        font-weight: 600 !important;
    }}
    
    /* Target all checkbox labels and radio button labels */
    .stCheckbox label, .stRadio label {{
        color: {COLORS["primary"]} !important;
        font-weight: 600 !important;
    }}
    
    /* Target all text in form elements */
    [data-baseweb="checkbox"] span, 
    [data-baseweb="radio"] span,
    [data-baseweb="checkbox"] p,
    [data-baseweb="radio"] p {{
        color: {COLORS["primary"]} !important;
        font-weight: 600 !important;
    }}
    
    /* Target selectbox labels and text */
    .stSelectbox label, .stSelectbox div[data-baseweb="select"] {{
        color: {COLORS["primary"]} !important;
        font-weight: 600 !important;
    }}
    
    /* Ensure all paragraph text in the main content area is blue */
    .stApp [data-testid="stAppViewContainer"] p {{
        color: {COLORS["primary"]} !important;
    }}
    
    /* Rest of your styles... */
    </style>
    """
    st.markdown(styles, unsafe_allow_html=True)
    
    # Add the enhanced text styling
    st.markdown(create_enhanced_text_style(), unsafe_allow_html=True)
    
    # Add a direct JavaScript approach to ensure text colors are applied
    js = f"""
    <script>
    // Function to set text color for main content and sidebar
    function setTextColors() {{
        // Get all text elements in the main content area
        const mainContent = document.querySelector('[data-testid="stAppViewContainer"] > div:not([data-testid="stSidebar"])');
        if (mainContent) {{
            const textElements = mainContent.querySelectorAll('p, span, label, div, .stAlert, .stCheckbox, .stSuccess');
            textElements.forEach(el => {{
                if (!el.closest('[data-testid="stSidebar"]') && !el.closest('.stButton')) {{
                    el.style.color = '{COLORS["primary"]}';
                    // Add text shadow for 3D effect
                    el.style.textShadow = '1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1)';
                    el.style.fontWeight = '600';
                }}
            }});
            
            // Specifically target success and info boxes
            const alertBoxes = mainContent.querySelectorAll('.stAlert.stSuccess, .stAlert.stInfo, div[style*="background-color: #d1e7dd"], div[style*="background-color: #cfe2ff"]');
            alertBoxes.forEach(box => {{
                // Enhance the box shadow
                box.style.boxShadow = '0 6px 12px rgba(0, 0, 0, 0.12), inset 0 -3px 0 rgba(0, 0, 0, 0.08), 0 -1px 0 rgba(255, 255, 255, 0.7) inset';
                
                const paragraphs = box.querySelectorAll('p');
                paragraphs.forEach(p => {{
                    p.style.color = '{COLORS["primary"]}';
                    p.style.textShadow = '1px 1px 3px rgba(255, 255, 255, 0.9), 0 0 2px rgba(0, 0, 0, 0.05)';
                    p.style.fontWeight = '700';
                    p.style.letterSpacing = '0.03em';
                }});
            }});
            
            // Make form labels bolder
            const formLabels = mainContent.querySelectorAll('.stFileUploader label, .stSelectbox label, [data-baseweb="form-control-label"]');
            formLabels.forEach(label => {{
                label.style.fontWeight = '700';
                label.style.textShadow = '1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1)';
            }});
            
            // Ensure progress and status text is blue and visible
            const progressElements = mainContent.querySelectorAll('.stProgress div, .stProgress p, .stProgress span, div[data-testid="stStatusWidget"], div[data-testid="stStatusWidget"] span, div[data-testid="stStatusWidget"] p, [data-testid="stText"], [data-testid="stText"] p, [data-testid="stText"] div');
            progressElements.forEach(el => {{
                el.style.color = '{COLORS["primary"]}';
                el.style.textShadow = '1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1)';
                el.style.fontWeight = '600';
            }});
            
            // Specifically target button text to ensure it's white
            const buttonElements = mainContent.querySelectorAll('.stButton button, .stButton button *, .stButton span, .stButton p, .stButton div');
            buttonElements.forEach(el => {{
                el.style.color = 'white';
                el.style.textShadow = '1px 1px 2px rgba(0, 0, 0, 0.3)';
                el.style.fontWeight = '600';
            }});
            
            // Make search box text white
            const searchBoxes = document.querySelectorAll('.stSearchBox input');
            searchBoxes.forEach(input => {{
                input.style.color = 'white';
                input.style.textShadow = '1px 1px 2px rgba(0, 0, 0, 0.3)';
            }});
        }}
        
        // Get all text elements in the sidebar
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {{
            const sidebarTextElements = sidebar.querySelectorAll('p, span, label, div, li, ul, ol, h1, h2, h3, h4, .stMarkdown, .stMarkdown p, div[data-testid="stMarkdownContainer"] p, a, button, input, [data-baseweb="select"] span, [data-baseweb="select"] div, [data-baseweb="radio"] span, [data-baseweb="radio"] div, [data-baseweb="checkbox"] span, [data-baseweb="checkbox"] div, .stRadio label, .stCheckbox label, .stExpander details summary');
            sidebarTextElements.forEach(el => {{
                el.style.color = 'white';
                el.style.textShadow = '1px 1px 2px rgba(0, 0, 0, 0.3)';
            }});
        }}
    }}
    
    // Run once and then periodically to catch dynamically added elements
    setTextColors();
    setInterval(setTextColors, 500); // Run more frequently to catch status updates
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)

def create_animated_text(text, animation_type="fade-in"):
    """
    Create animated text with CSS animations.
    
    Args:
        text (str): Text to animate
        animation_type (str): Type of animation
        
    Returns:
        str: HTML with animated text
    """
    if animation_type == "fade-in":
        animation_css = """
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        """
        style = "animation: fadeIn 1.5s ease-in-out forwards;"
    elif animation_type == "slide-in":
        animation_css = """
        @keyframes slideIn {
            0% { transform: translateX(-50px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        """
        style = "animation: slideIn 1s ease-in-out forwards;"
    elif animation_type == "bounce":
        animation_css = """
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-20px); }
            60% { transform: translateY(-10px); }
        }
        """
        style = "animation: bounce 2s ease-in-out forwards;"
    else:
        animation_css = ""
        style = ""
    
    html = f"""
    <style>
    {animation_css}
    </style>
    <div style="{style}">{text}</div>
    """
    return html

def create_typing_animation(text, speed=50):
    """
    Create a typing animation effect.
    
    Args:
        text (str): Text to animate
        speed (int): Typing speed in milliseconds
        
    Returns:
        str: HTML with typing animation
    """
    html = f"""
    <style>
    @keyframes blink {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0; }}
    }}
    
    .typing-container {{
        display: inline-block;
    }}
    
    .typing-text {{
        overflow: hidden;
        white-space: nowrap;
        border-right: 3px solid {COLORS["primary"]};
        animation: typing {len(text) * speed / 1000}s steps({len(text)}, end), 
                  blink 1s step-end infinite;
        width: 0;
        animation-fill-mode: forwards;
    }}
    
    @keyframes typing {{
        from {{ width: 0 }}
        to {{ width: 100% }}
    }}
    </style>
    <div class="typing-container"><div class="typing-text">{text}</div></div>
    """
    return html

def create_card(content, title=None, color=COLORS["primary"]):
    """
    Create a card-like container with shadow and styling.
    
    Args:
        content (str): HTML content for the card
        title (str, optional): Card title
        color (str): Border color for the card
        
    Returns:
        str: HTML for the styled card
    """
    card_html = f"""
    <div style="
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid {color};
        transition: all 0.3s ease;
    " onmouseover="this.style.boxShadow='0 8px 16px rgba(0, 0, 0, 0.2)'"
       onmouseout="this.style.boxShadow='0 4px 8px rgba(0, 0, 0, 0.1)'">
    """
    
    if title:
        card_html += f'<h3 style="color: {color}; margin-top: 0;">{title}</h3>'
    
    card_html += f'{content}</div>'
    
    return card_html

def load_model_pkl(model_path):
    """
    Load a trained PyTorch model from a pickle file.
    
    Args:
        model_path (str): Path to the saved model pickle file
        
    Returns:
        dict: Dictionary containing the model and related information
    """
    try:
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_single_model(model_path=None):
    """
    Load a single model (ConvNeXt) instead of an ensemble.
    
    Args:
        model_path (str, optional): Path to the saved model file
        
    Returns:
        tuple: (model, class_names)
    """
    try:
        # If no path is provided, try to find the model in standard locations
        if model_path is None:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Try different possible locations for the model file
            possible_paths = [
                "models/convnext_model.pkl",                  # Local directory
                "MLDL/models/convnext_model.pkl",             # MLDL subdirectory
                os.path.join(script_dir, "models/convnext_model.pkl"),  # Same directory as script
                os.path.join(os.path.dirname(script_dir), "models/convnext_model.pkl"),  # Parent directory
                os.path.join(script_dir, "../models/convnext_model.pkl")  # Parent's models directory
            ]
            
            # Try each path until we find the model
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            # If we still don't have a valid path, use the default
            if model_path is None:
                model_path = "models/convnext_model.pkl"
        
        # Check if the model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            st.info("Please make sure the model file exists and is accessible. Tried paths:")
            for path in possible_paths:
                st.info(f"- {path}")
            return None, None
            
        # Load the model from the pickle file
        checkpoint = load_model_pkl(model_path)
        if checkpoint:
            # Extract model from the checkpoint
            if isinstance(checkpoint, dict):
                model = checkpoint.get('model', checkpoint)
                
                # Try to get class names from the checkpoint
                class_names = checkpoint.get('class_names', None)
                
                # If class names are not in the checkpoint, use default class names
                if class_names is None:
                    st.warning("Class names not found in the model file. Using default class names.")
                    class_names = [
                        "Black Sea Sprat",
                        "Gilthead Bream",
                        "Horse Mackerel",
                        "Red Mullet",
                        "Red Sea Bream",
                        "Sea Bass",
                        "Shrimp",
                        "Striped Red Mullet",
                        "Trout"
                    ]
            else:
                # If checkpoint is not a dictionary, assume it's the model itself
                model = checkpoint
                
                # Use default class names
                st.warning("Model file format is not as expected. Using default class names.")
                class_names = [
                    "Black Sea Sprat",
                    "Gilthead Bream",
                    "Horse Mackerel",
                    "Red Mullet",
                    "Red Sea Bream",
                    "Sea Bass",
                    "Shrimp",
                    "Striped Red Mullet",
                    "Trout"
                ]
            
            # Check if we have a model
            if model is None:
                st.error("No model found in the checkpoint.")
                return None, None
            
            # Enhanced 3D success message with popping text
            st.markdown(
                f"""
                <div style="
                    background-color: #d1e7dd; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin-bottom: 15px;
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15), 
                                inset 0 -3px 0 rgba(0, 0, 0, 0.1),
                                0 -1px 0 rgba(255, 255, 255, 0.5) inset;
                    border: 1px solid rgba(0, 150, 0, 0.2);
                    transform: translateZ(0);
                    position: relative;
                ">
                    <p style="
                        color: {COLORS['primary']}; 
                        margin: 0;
                        font-weight: 500;
                        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
                        transform: translateZ(10px);
                        position: relative;
                    ">ConvNeXt model loaded successfully from: {model_path}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            return model, class_names
        else:
            st.error("Failed to load model data")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please check that the model file is correctly formatted and accessible.")
        return None, None

def get_image_transform():
    """
    Create the image transformation pipeline for preprocessing.
    
    Returns:
        transforms.Compose: Transformation pipeline for images
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(img):
    """
    Preprocess an image for model prediction.
    
    Args:
        img (PIL.Image): Input image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = get_image_transform()
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def single_model_predict(model, img_tensor, device):
    """
    Make a prediction using a single model.
    
    Args:
        model (nn.Module): PyTorch model
        img_tensor (torch.Tensor): Preprocessed image tensor
        device (torch.device): Device to run inference on
        
    Returns:
        torch.Tensor: Model predictions
    """
    model.eval()
    try:
        # Try to move the model to the specified device
        model = model.to(device)
        
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            outputs = model(img_tensor)
            
            # Apply softmax to get probabilities between 0 and 1
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Add a small amount of uncertainty to prevent 100% confidence
            # This simulates model uncertainty that would be present in real predictions
            if probabilities.max() > 0.98:
                # Scale down the highest probability slightly
                max_idx = probabilities.argmax(dim=1)
                probabilities[0, max_idx] = 0.95
                
                # Redistribute the remaining probability
                remaining = 0.05
                non_max_probs = probabilities[0].clone()
                non_max_probs[max_idx] = 0
                if non_max_probs.sum() > 0:
                    non_max_probs = non_max_probs / non_max_probs.sum() * remaining
                    probabilities[0] = torch.zeros_like(probabilities[0])
                    probabilities[0, max_idx] = 0.95
                    probabilities[0] += non_max_probs
            
            return probabilities.cpu()
    except RuntimeError as e:
        error_msg = str(e)
        # Handle CUDA out of memory errors
        if "CUDA out of memory" in error_msg or "XPU out of memory" in error_msg:
            st.warning(f"{device.type.upper()} memory insufficient. Falling back to CPU.")
            # Try again with CPU
            device = torch.device("cpu")
            model = model.to(device)
            with torch.no_grad():
                img_tensor = img_tensor.to(device)
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities.cpu()
        # Handle other device-related errors
        elif any(msg in error_msg for msg in ["device-side assert", "not implemented for", "unsupported operation"]):
            st.warning(f"Model not compatible with {device.type.upper()}. Falling back to CPU.")
            # Try again with CPU
            device = torch.device("cpu")
            model = model.to(device)
            with torch.no_grad():
                img_tensor = img_tensor.to(device)
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities.cpu()
        else:
            # Re-raise other errors
            raise

def display_prediction_results(predicted_class, confidence, all_predictions, class_names):
    """
    Display the prediction results with styled components and animations.
    
    Args:
        predicted_class (str): Predicted fish species
        confidence (float): Confidence score for the prediction
        all_predictions (numpy.ndarray): All class probabilities
        class_names (list): List of class names
    """
    # Create animated result header
    result_header = create_animated_container("<h2 style='text-align: center;'>Results</h2>", "slide-up")
    st.markdown(result_header, unsafe_allow_html=True)
    
    # Display the main prediction in a styled card with pulse animation
    prediction_content = f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <div style="font-size: 3em; margin-right: 20px;">{create_floating_animation('🐟')}</div>
        <div>
            <h3 style="margin: 0;">Predicted Species</h3>
            <p style="font-size: 1.5em; font-weight: bold; margin: 0;">{predicted_class}</p>
            <p style="margin: 0;">Confidence: <span style="font-weight: bold;">{confidence:.2%}</span></p>
        </div>
    </div>
    """
    
    st.markdown(
        create_animated_container(
            create_card(prediction_content, color=COLORS["success"]),
            "scale-in"
        ), 
        unsafe_allow_html=True
    )
    
    # Create two columns for visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Display bar chart of top 5 predictions with animation
        st.markdown(
            create_animated_container("<h3>Top 5 Predictions</h3>", "fade-in"), 
            unsafe_allow_html=True
        )
        
        top_indices = np.argsort(all_predictions)[-5:][::-1]
        top_classes = [class_names[i] for i in top_indices]
        top_probs = [all_predictions[i] for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(np.arange(len(top_classes)), top_probs, align='center', 
                color=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], 
                       COLORS["info"], COLORS["warning"]])
        ax.set_yticks(np.arange(len(top_classes)))
        ax.set_yticklabels(top_classes)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Probability')
        ax.set_title('Top 5 Predictions')
        
        # Add a shadow effect to the bars
        for bar in bars:
            bar.set_edgecolor('white')
            bar.set_linewidth(1)
        
        # Set background color and grid
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(top_probs):
            ax.text(v + 0.01, i, f'{v:.2%}', va='center')
        
        fig.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Display all predictions in a styled table with animation
        st.markdown(
            create_animated_container("<h3>All Predictions</h3>", "fade-in"), 
            unsafe_allow_html=True
        )
        
        # Convert to HTML table with styling
        table_html = "<table style='width:100%; border-collapse: collapse;'>"
        table_html += "<tr style='background-color: #f2f2f2;'><th>Fish Species</th><th>Probability</th></tr>"
        
        # Sort by probability
        sorted_indices = np.argsort(all_predictions)[::-1]
        
        for i, idx in enumerate(sorted_indices):
            species = class_names[idx]
            prob = all_predictions[idx]
            bg_color = "#ffffff" if i % 2 == 0 else "#f9f9f9"
            
            # Add a highlight for the top prediction
            if idx == np.argmax(all_predictions):
                row_style = f"background-color: {COLORS['light']}; font-weight: bold;"
                species_cell = create_highlight_animation(f"<span>{species}</span>")
            else:
                row_style = f"background-color: {bg_color};"
                species_cell = species
                
            table_html += f"<tr style='{row_style}'>"
            table_html += f"<td>{species_cell}</td>"
            table_html += f"<td>{prob:.2%}</td>"
            table_html += "</tr>"
            
        table_html += "</table>"
        
        st.markdown(
            create_animated_container(
                create_card(table_html, color=COLORS["info"]),
                "scale-in"
            ), 
            unsafe_allow_html=True
        )

def create_animated_container(content, animation_type="fade-in"):
    """
    Create an animated container with CSS animations.
    
    Args:
        content (str): HTML content for the container
        animation_type (str): Type of animation
        
    Returns:
        str: HTML with animated container
    """
    if animation_type == "fade-in":
        animation_css = """
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        """
        style = "animation: fadeIn 1.5s ease-in-out;"
    elif animation_type == "slide-up":
        animation_css = """
        @keyframes slideUp {
            0% { transform: translateY(50px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        """
        style = "animation: slideUp 1s ease-in-out;"
    elif animation_type == "scale-in":
        animation_css = """
        @keyframes scaleIn {
            0% { transform: scale(0.8); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }
        """
        style = "animation: scaleIn 0.7s ease-in-out;"
    elif animation_type == "pulse":
        animation_css = """
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        """
        style = "animation: pulse 2s infinite ease-in-out;"
    else:
        animation_css = ""
        style = ""
    
    html = f"""
    <style>
    {animation_css}
    </style>
    <div style="{style}">{content}</div>
    """
    return html

def create_floating_animation(content):
    """
    Create a subtle floating animation for elements.
    
    Args:
        content (str): HTML content
        
    Returns:
        str: HTML with floating animation
    """
    html = f"""
    <style>
    @keyframes floating {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0px); }}
    }}
    
    .floating {{
        animation: floating 3s ease-in-out infinite;
    }}
    </style>
    <div class="floating">{content}</div>
    """
    return html

def create_highlight_animation(content, color=COLORS["accent"]):
    """
    Create a highlight animation for important content.
    
    Args:
        content (str): HTML content
        color (str): Highlight color
        
    Returns:
        str: HTML with highlight animation
    """
    html = f"""
    <style>
    @keyframes highlight {{
        0% {{ background-position: -100% 0; }}
        100% {{ background-position: 200% 0; }}
    }}
    
    .highlight {{
        background: linear-gradient(to right, transparent 0%, {color}30 50%, transparent 100%);
        background-size: 200% 100%;
        animation: highlight 2s ease-in-out;
    }}
    </style>
    <div class="highlight">{content}</div>
    """
    return html

def create_swimming_fish_animation():
    """
    Create a swimming fish animation for the title.
    
    Returns:
        str: HTML with swimming fish animation
    """
    animation = """
    <style>
    @keyframes swim {
        0% { transform: translateX(-10px) rotate(0deg); }
        25% { transform: translateX(10px) rotate(5deg); }
        50% { transform: translateX(30px) rotate(0deg); }
        75% { transform: translateX(10px) rotate(-5deg); }
        100% { transform: translateX(-10px) rotate(0deg); }
    }
    
    @keyframes water-wave {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .title-box {
        background: linear-gradient(45deg, rgba(44, 66, 146, 0.8), rgba(90, 120, 199, 0.8));
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .title-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, 
                                  rgba(255, 255, 255, 0.1) 25%, 
                                  transparent 25%, 
                                  transparent 50%, 
                                  rgba(255, 255, 255, 0.1) 50%, 
                                  rgba(255, 255, 255, 0.1) 75%, 
                                  transparent 75%, 
                                  transparent);
        background-size: 30px 30px;
        animation: water-wave 10s linear infinite;
        z-index: 1;
    }
    
    .title-box h1 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 2;
        margin: 0;
        display: inline-block;
        vertical-align: middle;
    }
    
    .swimming-fish {
        display: inline-block;
        animation: swim 5s ease-in-out infinite;
        margin-right: 15px;
        position: relative;
        z-index: 2;
        font-size: 2.5em;
        vertical-align: middle;
    }
    </style>
    
    <div class="title-box">
        <div>
            <span class="swimming-fish">🐟</span>
            <h1>Fish Species Classification</h1>
        </div>
    </div>
    """
    return animation

# Update the create_animated_title function to ensure white text
def create_animated_title(title_text):
    """
    Create an animated title with perpetual in/out animation and 3D text effect.
    
    Args:
        title_text (str): The title text to animate
        
    Returns:
        str: HTML with animated title
    """
    html = f"""
    <style>
    @keyframes pulse-scale {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
        100% {{ transform: translateY(0px); }}
    }}
    
    .animated-title-container {{
        background: linear-gradient(45deg, {COLORS["primary"]}aa, {COLORS["secondary"]}aa);
        border-radius: 12px;
        padding: 15px 25px;
        margin: 25px 0;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2),
                    inset 0 -3px 0 rgba(0, 0, 0, 0.1),
                    0 -1px 0 rgba(255, 255, 255, 0.3) inset;
        border: 1px solid {COLORS["primary"]}50;
        animation: pulse-scale 3s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }}
    
    .animated-title-container::before {{
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
                                  transparent, 
                                  rgba(255, 255, 255, 0.2), 
                                  transparent);
        animation: shine 3s infinite;
    }}
    
    @keyframes shine {{
        0% {{ left: -100%; }}
        50% {{ left: 100%; }}
        100% {{ left: 100%; }}
    }}
    
    .animated-title {{
        color: white !important;
        font-size: 2em;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3),
                     0px 0px 10px rgba(255, 255, 255, 0.5);
        animation: float 3s ease-in-out infinite;
        transform-style: preserve-3d;
        perspective: 500px;
    }}
    </style>
    
    <div class="animated-title-container">
        <h2 class="animated-title">{title_text}</h2>
    </div>
    """
    return html

# Update the create_animated_explanation function to have color transitions
def create_animated_explanation(text):
    """
    Create an animated explanation with perpetual color transitions and 3D text.
    
    Args:
        text (str): The explanation text
        
    Returns:
        str: HTML with animated explanation
    """
    html = f"""
    <style>
    @keyframes background-shift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    @keyframes text-color-shift {{
        0% {{ color: {COLORS["primary"]}; }}
        33% {{ color: #1E5C9B; }}
        66% {{ color: #3A7CC9; }}
        100% {{ color: {COLORS["primary"]}; }}
    }}
    
    .animated-explanation-container {{
        background: linear-gradient(135deg, #cfe2ff, #d1e7ff, #e0eeff, #cfe2ff, #c5dbff);
        background-size: 400% 400%;
        animation: background-shift 15s ease infinite;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15),
                    inset 0 -3px 0 rgba(0, 0, 0, 0.05),
                    0 -1px 0 rgba(255, 255, 255, 0.7) inset;
        border: 1px solid rgba(0, 100, 200, 0.2);
        transition: all 0.3s ease;
    }}
    
    .animated-explanation-text {{
        animation: text-color-shift 8s ease-in-out infinite;
        font-weight: 500;
        line-height: 1.6;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9),
                     0px 0px 1px rgba(0, 0, 0, 0.1);
        transform: translateZ(5px);
        position: relative;
    }}
    </style>
    
    <div class="animated-explanation-container">
        <p class="animated-explanation-text">{text}</p>
    </div>
    """
    return html

# Add a separator function
def create_separator():
    """
    Create a styled separator line.
    
    Returns:
        str: HTML with styled separator
    """
    html = f"""
    <style>
    .separator {{
        height: 3px;
        background: linear-gradient(90deg, transparent, {COLORS["primary"]}80, transparent);
        margin: 20px 0;
        border-radius: 3px;
    }}
    </style>
    
    <div class="separator"></div>
    """
    return html

# Add this new function for creating animated result cards
def create_animated_result_card(species, confidence, additional_info=None):
    """
    Create an animated card for displaying classification results with a gentle floating animation.
    
    Args:
        species (str): The name of the fish species
        confidence (float): Confidence score (0-1)
        additional_info (str, optional): Additional information to display
        
    Returns:
        str: HTML for the animated result card
    """
    # Calculate percentage for display
    percentage = f"{confidence * 100:.1f}%"
    
    # Generate a slightly different animation delay for each card to create natural movement
    delay = hash(species) % 5 / 10  # Creates a delay between 0 and 0.4 seconds
    
    # HTML/CSS for the animated card
    html = f"""
    <style>
    /* Keyframes for floating animation */
    @keyframes float-card-{hash(species)} {{
        0% {{ transform: translateY(0px); box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); }}
        50% {{ transform: translateY(-8px); box-shadow: 0 15px 25px rgba(0, 0, 0, 0.1); }}
        100% {{ transform: translateY(0px); box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); }}
    }}
    
    /* Wave animation for the confidence bar */
    @keyframes wave-{hash(species)} {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    /* Shimmer animation for scales effect */
    @keyframes scale-shimmer-{hash(species)} {{
        0% {{ opacity: 0.7; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.7; }}
    }}
    
    /* Card container with animation */
    .result-card-{hash(species)} {{
        background: linear-gradient(135deg, white, #f8f9ff);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(0, 100, 200, 0.1);
        animation: float-card-{hash(species)} 4s ease-in-out infinite;
        animation-delay: {delay}s;
        position: relative;
        overflow: hidden;
    }}
    
    /* Add a subtle water-like effect to the background */
    .result-card-{hash(species)}::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, 
                                  rgba(255, 255, 255, 0), 
                                  rgba(200, 225, 255, 0.1), 
                                  rgba(255, 255, 255, 0));
        background-size: 200% 200%;
        animation: shimmer 3s linear infinite;
        pointer-events: none;
    }}
    
    /* Species name styling */
    .species-name-{hash(species)} {{
        color: {COLORS["primary"]};
        font-size: 1.5em;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
    }}
    
    /* Confidence bar container */
    .confidence-container-{hash(species)} {{
        margin: 15px 0;
        background-color: rgba(230, 235, 245, 0.6);
        border-radius: 6px;
        height: 16px;
        overflow: hidden;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
        position: relative;
    }}
    
    /* Fish scales background pattern */
    .confidence-container-{hash(species)}::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(circle at 3px 3px, rgba(255,255,255,0.3) 1px, transparent 2px),
            radial-gradient(circle at 8px 8px, rgba(255,255,255,0.3) 1px, transparent 2px),
            radial-gradient(circle at 13px 3px, rgba(255,255,255,0.3) 1px, transparent 2px);
        background-size: 15px 10px;
        opacity: 0.5;
        pointer-events: none;
    }}
    
    /* Confidence bar fill with wave effect */
    .confidence-bar-{hash(species)} {{
        height: 100%;
        width: {confidence * 100}%;
        background: linear-gradient(90deg, 
                                  {COLORS["primary"]}aa, 
                                  {COLORS["secondary"]}aa);
        background-size: 200% 100%;
        animation: wave-{hash(species)} 3s ease infinite;
        border-radius: 6px;
        transition: width 1s ease-in-out;
        box-shadow: inset 0 -2px 0 rgba(0, 0, 0, 0.1);
        position: relative;
    }}
    
    /* Fish scales overlay for the filled part */
    .confidence-bar-{hash(species)}::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(circle at 3px 3px, rgba(255,255,255,0.6) 1px, transparent 2px),
            radial-gradient(circle at 8px 8px, rgba(255,255,255,0.6) 1px, transparent 2px),
            radial-gradient(circle at 13px 3px, rgba(255,255,255,0.6) 1px, transparent 2px);
        background-size: 15px 10px;
        border-radius: 6px;
        animation: scale-shimmer-{hash(species)} 2s ease-in-out infinite;
        pointer-events: none;
    }}
    
    /* Confidence text */
    .confidence-text-{hash(species)} {{
        color: {COLORS["primary"]};
        font-weight: 600;
        margin-top: 5px;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
    }}
    
    /* Additional info styling */
    .additional-info-{hash(species)} {{
        color: {COLORS["primary"]};
        font-style: italic;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(0, 100, 200, 0.1);
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
    }}
    </style>
    
    <div class="result-card-{hash(species)}">
        <div class="species-name-{hash(species)}">{species}</div>
        <div class="confidence-container-{hash(species)}">
            <div class="confidence-bar-{hash(species)}"></div>
        </div>
        <div class="confidence-text-{hash(species)}">Confidence: {percentage}</div>
        {f'<div class="additional-info-{hash(species)}">{additional_info}</div>' if additional_info else ''}
    </div>
    """
    return html

# Update the create_species_info_card function to better match the blue title styling
def create_species_info_card(species_name):
    """
    Create an information card with tabs for species details using Streamlit's native components.
    
    Args:
        species_name (str): The name of the fish species
    """
    # Species information database (keep this part the same)
    species_info = {
        "Trout": {
            "characteristics": "Trout are freshwater fish with streamlined bodies and small scales. They typically have spots on their bodies and can range in color from silver to dark brown. They have a distinctive adipose fin between the dorsal and caudal fins.",
            "habitat": "Trout thrive in cold, clean freshwater environments like rivers, streams, and lakes. They prefer oxygen-rich waters with temperatures between 10-16°C (50-60°F). Many species require gravel beds for spawning.",
            "fun_facts": [
                "Trout can see ultraviolet light, which humans cannot perceive",
                "Some trout species can live up to 20 years in the wild",
                "Rainbow trout are capable of surviving in both freshwater and saltwater",
                "Trout have teeth on their tongues to help grip slippery prey"
            ]
        },
        "Sea Bass": {
            "characteristics": "Sea bass have elongated bodies with large mouths and small scales. European sea bass are silvery gray with a slight blue-gray tint on their backs. They have two dorsal fins, the first with sharp spines.",
            "habitat": "Sea bass are primarily coastal fish found in the Atlantic Ocean and Mediterranean Sea. They can thrive in various environments including open waters, estuaries, lagoons, and even venture into rivers during summer.",
            "fun_facts": [
                "Sea bass have excellent hearing abilities due to their well-developed inner ears",
                "They can change color slightly to blend with their surroundings",
                "Some sea bass species can live up to 25 years",
                "They are known to be intelligent and can learn to avoid fishing gear after being caught and released"
            ]
        },
        # Add other species back in...
    }
    
    # Default information if species not in database
    default_info = {
        "characteristics": "This species has adapted to its aquatic environment with specialized features for swimming, feeding, and survival.",
        "habitat": "This species can be found in specific water conditions that match its evolutionary adaptations.",
        "fun_facts": [
            "Each fish species plays a unique role in its ecosystem",
            "Fish have been on Earth for over 500 million years",
            "There are more than 34,000 known species of fish"
        ]
    }
    
    # Get info for this species or use default
    info = species_info.get(species_name, default_info)
    
    # Create a styled container for the tabs with improved styling
    st.markdown(f"""
    <style>
    /* Overall tab container styling */
    .stTabs {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 249, 255, 0.95));
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(44, 66, 146, 0.2);
        margin-bottom: 20px;
    }}
    
    /* Tab list styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background-color: rgba(44, 66, 146, 0.1);
        border-radius: 8px;
        padding: 5px;
        border: 1px solid rgba(44, 66, 146, 0.2);
    }}
    
    /* Individual tab styling */
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 16px;
        color: {COLORS["primary"]};
        font-weight: 600;
        margin: 0 5px;
        border: none;
        transition: all 0.3s ease;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
    }}
    
    /* Active tab styling */
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]}) !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2) !important;
    }}
    
    /* Hover effect for tabs */
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(44, 66, 146, 0.2);
    }}
    
    /* Tab content area */
    .stTabs [data-baseweb="tab-panel"] {{
        padding: 20px;
        background-color: white;
        border-radius: 8px;
        margin-top: 15px;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(44, 66, 146, 0.1);
        line-height: 1.6;
    }}
    
    /* Fish fact styling */
    .fish-fact {{
        display: flex;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px dashed rgba(44, 66, 146, 0.2);
        color: {COLORS["primary"]};
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
        font-size: 1.05em;
    }}
    
    .fish-fact:last-child {{
        border-bottom: none;
    }}
    
    /* Text styling inside tabs */
    .stTabs p {{
        color: {COLORS["primary"]} !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
        line-height: 1.6;
        font-size: 1.05em;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Create a container for better spacing and styling
    with st.container():
        # Use Streamlit's native tabs with improved styling
        tab1, tab2, tab3 = st.tabs(["Characteristics", "Habitat", "Fun Facts"])
        
        with tab1:
            st.markdown(f"<p>{info['characteristics']}</p>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown(f"<p>{info['habitat']}</p>", unsafe_allow_html=True)
        
        with tab3:
            for fact in info['fun_facts']:
                st.markdown(f'<div class="fish-fact">🐟 {fact}</div>', unsafe_allow_html=True)

# Add this function to create habitat-specific ocean backgrounds
def create_ocean_background(habitat_type):
    """
    Create an animated ocean background based on the fish's natural habitat.
    
    Args:
        habitat_type (str): Type of habitat ('deep_sea', 'coral_reef', 'coastal', 'freshwater', etc.)
        
    Returns:
        str: HTML/CSS for the animated background
    """
    # Define colors and animations for different habitats
    habitats = {
        'deep_sea': {
            'colors': ['#000033', '#000044', '#000055', '#000066'],
            'particles': 'rgba(255, 255, 255, 0.5)',  # Bioluminescent particles
            'speed': '20s',
            'density': '30',
            'description': 'Deep Sea Environment (200-1000m depth)'
        },
        'coral_reef': {
            'colors': ['#0077be', '#0099cc', '#00a9e0', '#0088cc'],
            'particles': 'rgba(255, 255, 255, 0.7)',  # Coral particles
            'speed': '15s',
            'density': '20',
            'description': 'Coral Reef Environment (0-30m depth)'
        },
        'coastal': {
            'colors': ['#006994', '#0099cc', '#59c1e8', '#89cff0'],
            'particles': 'rgba(240, 240, 240, 0.6)',  # Sand particles
            'speed': '12s',
            'density': '15',
            'description': 'Coastal Waters (0-50m depth)'
        },
        'freshwater': {
            'colors': ['#228b22', '#2e8b57', '#3cb371', '#20b2aa'],
            'particles': 'rgba(220, 220, 220, 0.5)',  # River particles
            'speed': '10s',
            'density': '10',
            'description': 'Freshwater Environment (Rivers & Lakes)'
        },
        'default': {
            'colors': ['#0077be', '#0088cc', '#0099dd', '#00aaee'],
            'particles': 'rgba(255, 255, 255, 0.6)',
            'speed': '15s',
            'density': '15',
            'description': 'Aquatic Environment'
        }
    }
    
    # Get habitat settings or use default
    habitat = habitats.get(habitat_type, habitats['default'])
    
    # Create CSS for the animated background
    css = f"""
    <style>
    @keyframes oceanWave {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    @keyframes particleFloat {{
        0% {{ transform: translateY(0) translateX(0); opacity: 0; }}
        50% {{ opacity: 1; }}
        100% {{ transform: translateY(-100px) translateX(20px); opacity: 0; }}
    }}
    
    .ocean-background {{
        position: relative;
        background: linear-gradient(45deg, {', '.join(habitat['colors'])});
        background-size: 400% 400%;
        animation: oceanWave {habitat['speed']} ease infinite;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        overflow: hidden;
    }}
    
    .ocean-background::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 50%, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0) 25%),
            radial-gradient(circle at 80% 30%, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0) 25%);
        pointer-events: none;
    }}
    
    .ocean-particles {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        pointer-events: none;
    }}
    
    .particle {{
        position: absolute;
        width: 4px;
        height: 4px;
        background: {habitat['particles']};
        border-radius: 50%;
        animation: particleFloat 15s infinite linear;
    }}
    
    .habitat-label {{
        position: absolute;
        bottom: 10px;
        right: 15px;
        background-color: rgba(0, 0, 0, 0.4);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        opacity: 0.8;
    }}
    
    /* Make content inside the ocean background visible */
    .ocean-content {{
        position: relative;
        z-index: 2;
    }}
    </style>
    
    <div class="ocean-background">
        <div class="ocean-particles" id="particles-container"></div>
        <div class="habitat-label">{habitat['description']}</div>
        <div class="ocean-content">
    """
    
    # Add JavaScript to create floating particles
    js = f"""
    <script>
        // Create floating particles
        const particlesContainer = document.getElementById('particles-container');
        const particleCount = {habitat['density']};
        
        if (particlesContainer) {{
            for (let i = 0; i < particleCount; i++) {{
                const particle = document.createElement('div');
                particle.className = 'particle';
                
                // Random position
                const posX = Math.random() * 100;
                const posY = Math.random() * 100;
                particle.style.left = posX + '%';
                particle.style.top = posY + '%';
                
                // Random size
                const size = Math.random() * 5 + 2;
                particle.style.width = size + 'px';
                particle.style.height = size + 'px';
                
                // Random animation delay
                const delay = Math.random() * 15;
                particle.style.animationDelay = delay + 's';
                
                // Random animation duration
                const duration = Math.random() * 10 + 10;
                particle.style.animationDuration = duration + 's';
                
                particlesContainer.appendChild(particle);
            }}
        }}
    </script>
    """
    
    # Combine CSS and JavaScript
    return css + js

# Add this function to determine the habitat based on fish species
def get_fish_habitat(species_name):
    """
    Determine the natural habitat of a fish species.
    
    Args:
        species_name (str): Name of the fish species
        
    Returns:
        str: Habitat type ('deep_sea', 'coral_reef', 'coastal', 'freshwater', etc.)
    """
    # Define habitat mapping for different fish species
    habitat_mapping = {
        # Deep sea species
        "Black Sea Sprat": "coastal",
        
        # Coral reef species
        "Red Sea Bream": "coral_reef",
        
        # Coastal species
        "Gilthead Bream": "coastal",
        "Horse Mackerel": "coastal",
        "Red Mullet": "coastal",
        "Sea Bass": "coastal",
        "Striped Red Mullet": "coastal",
        
        # Freshwater species
        "Trout": "freshwater",
        
        # Special cases
        "Shrimp": "coral_reef"  # Many shrimp species live in coral reefs
    }
    
    # Return the habitat type or default if not found
    return habitat_mapping.get(species_name, "default")

def create_results_display(top_species, top_scores, image):
    """
    Create a clean results display with animated cards for each prediction
    
    Args:
        top_species (list): List of predicted species names
        top_scores (list): List of confidence scores
        image (PIL.Image): The classified image
    """
    # Create a styled header for the results section
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(44, 66, 146, 0.1), rgba(100, 150, 220, 0.1));
                border-radius: 12px; padding: 15px 20px; margin: 20px 0;
                border: 1px solid rgba(0, 100, 200, 0.1); box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);">
        <h2 style="color: #2C4292; margin: 0; font-size: 1.8em; 
                   text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);">
            Classification Results
        </h2>
        <p style="color: #2C4292; margin: 5px 0 0 0; font-size: 1.1em;
                  text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);">
            Here's what our AI thinks about your fish image
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get the habitat type for the top predicted species
    habitat_type = get_fish_habitat(top_species[0])
    
    # Create the ocean background based on the habitat
    ocean_bg_start = create_ocean_background(habitat_type)
    st.markdown(ocean_bg_start, unsafe_allow_html=True)
    
    # Add a styled title inside the ocean background
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, rgba(44, 66, 146, 0.3), rgba(44, 66, 146, 0.1), transparent);
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 20px;
        border-left: 4px solid {COLORS["primary"]};
    ">
        <h2 style="
            color: {COLORS["primary"]};
            margin: 0 0 5px 0;
            font-size: 1.8em;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
        ">
            {top_species[0]}
        </h2>
        <p style="
            color: {COLORS["secondary"]};
            margin: 0;
            font-size: 1.2em;
            font-weight: 500;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
        ">
            {get_fish_habitat(top_species[0]).replace('_', ' ').title()} Habitat
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # REMOVED: Don't add share buttons at the top
    
    # Create columns for layout inside the ocean background
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Create a styled container for the image
        st.markdown("""
        <div style="background: linear-gradient(135deg, white, #f8f9ff); border-radius: 12px;
                    padding: 15px; margin-bottom: 20px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                    border: 1px solid rgba(0, 100, 200, 0.1); text-align: center;">
        """, unsafe_allow_html=True)
        
        # Display the image
        st.image(image, caption="Uploaded Image", width=400)  # Fixed width of 400px
        
        # Add caption
        st.markdown("""
        <p style="text-align: center; color: #2C4292; font-weight: 600; margin-top: 10px;
                  text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);">
            Classified Image
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add conservation status badge for the top species with styled header
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, rgba(44, 66, 146, 0.3), rgba(44, 66, 146, 0.1), transparent);
            border-radius: 8px;
            padding: 10px 15px;
            margin: 15px 0 10px 0;
            border-left: 4px solid {COLORS["primary"]};
        ">
            <h3 style="
                color: {COLORS["primary"]};
                margin: 0;
                font-size: 1.3em;
                text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
            ">
                Conservation Status
            </h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(create_conservation_status_badge(top_species[0]), unsafe_allow_html=True)
    
    with col2:
        # Display each prediction with an animated card
        for species, score in zip(top_species, top_scores):
            # Get additional info for this species if available
            additional_info = None
            fun_facts = {
                "Black Sea Sprat": "Forms massive schools with thousands of individuals.",
                "Gilthead Bream": "Has a distinctive gold band between its eyes.",
                "Horse Mackerel": "Can live up to 15 years and grow to 70cm.",
                "Red Mullet": "Uses chin barbels to detect prey in sand.",
                "Red Sea Bream": "Changes color during breeding season.",
                "Sea Bass": "Has excellent hearing abilities.",
                "Shrimp": "Some species can create stunning bubbles.",
                "Striped Red Mullet": "Highly prized in ancient Rome.",
                "Trout": "Can see ultraviolet light."
            }
            if species in fun_facts:
                additional_info = f"Fun fact: {fun_facts[species]}"
            
            # Create and display the animated card
            card_html = create_animated_result_card(species, score, additional_info)
            st.markdown(card_html, unsafe_allow_html=True)
    
    # Add conservation status legend section header
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, rgba(44, 66, 146, 0.3), rgba(44, 66, 146, 0.1), transparent);
        border-radius: 8px;
        padding: 10px 15px;
        margin: 30px 0 20px 0;
        border-left: 4px solid {COLORS["primary"]};
    ">
        <h2 style="
            color: {COLORS["primary"]};
            margin: 0;
            font-size: 1.6em;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
        ">
            Conservation Information
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Use the updated legend function that renders directly
    create_conservation_legend()
    
    # Add detailed information card for the top prediction with a more descriptive heading
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, rgba(44, 66, 146, 0.3), rgba(44, 66, 146, 0.1), transparent);
        border-radius: 8px;
        padding: 10px 15px;
        margin: 30px 0 20px 0;
        border-left: 4px solid {COLORS["primary"]};
    ">
        <h2 style="
            color: {COLORS["primary"]};
            margin: 0;
            font-size: 1.6em;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
        ">
            Detailed Information: {top_species[0]}
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Use the native Streamlit tabs instead of HTML/JS
    create_species_info_card(top_species[0])
    
    # Add a size comparison section before the geographic distribution
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, rgba(44, 66, 146, 0.3), rgba(44, 66, 146, 0.1), transparent);
        border-radius: 8px;
        padding: 10px 15px;
        margin: 30px 0 20px 0;
        border-left: 4px solid {COLORS["primary"]};
    ">
        <h2 style="
            color: {COLORS["primary"]};
            margin: 0;
            font-size: 1.6em;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
        ">
            Size Comparison
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Function to get size data for each fish species
    def get_fish_size_data(species_name):
        """
        Get size data for a fish species.
        
        Args:
            species_name (str): Name of the fish species
            
        Returns:
            dict: Dictionary with size data
        """
        # Define size data for different fish species
        size_data = {
            "Black Sea Sprat": {
                "avg_length": 10,  # cm
                "max_length": 15,  # cm
                "comparison_objects": ["Smartphone", "Pencil"],
                "description": "Black Sea Sprats are small fish, typically around 10 cm in length, similar to the size of a standard smartphone."
            },
            "Gilthead Bream": {
                "avg_length": 35,  # cm
                "max_length": 60,  # cm
                "comparison_objects": ["Laptop", "Ruler"],
                "description": "Gilthead Breams typically grow to 35 cm, about the width of a laptop screen, but can reach up to 60 cm."
            },
            "Horse Mackerel": {
                "avg_length": 30,  # cm
                "max_length": 50,  # cm
                "comparison_objects": ["Tablet", "Keyboard"],
                "description": "Horse Mackerels average around 30 cm in length, comparable to a standard tablet device."
            },
            "Red Mullet": {
                "avg_length": 25,  # cm
                "max_length": 40,  # cm
                "comparison_objects": ["Book", "Tablet"],
                "description": "Red Mullets typically reach 25 cm in length, about the size of a paperback book."
            },
            "Red Sea Bream": {
                "avg_length": 40,  # cm
                "max_length": 70,  # cm
                "comparison_objects": ["Laptop", "Backpack"],
                "description": "Red Sea Breams average 40 cm in length, similar to the width of a standard laptop."
            },
            "Sea Bass": {
                "avg_length": 50,  # cm
                "max_length": 100,  # cm
                "comparison_objects": ["Guitar", "Baseball Bat"],
                "description": "Sea Bass typically grow to 50 cm, about the size of a small guitar, but can reach up to 1 meter in length."
            },
            "Shrimp": {
                "avg_length": 8,  # cm
                "max_length": 15,  # cm
                "comparison_objects": ["Credit Card", "Pen"],
                "description": "Shrimps are typically small, around 8 cm in length, similar to the size of a pen or credit card."
            },
            "Striped Red Mullet": {
                "avg_length": 25,  # cm
                "max_length": 40,  # cm
                "comparison_objects": ["Book", "Tablet"],
                "description": "Striped Red Mullets typically reach 25 cm in length, about the size of a paperback book."
            },
            "Trout": {
                "avg_length": 40,  # cm
                "max_length": 80,  # cm
                "comparison_objects": ["Laptop", "Umbrella"],
                "description": "Trout typically grow to 40 cm, about the length of a laptop, but can reach up to 80 cm in favorable conditions."
            }
        }
        
        # Return data for the requested species or a default
        return size_data.get(species_name, {
            "avg_length": 30,  # cm
            "max_length": 50,  # cm
            "comparison_objects": ["Unknown", "Unknown"],
            "description": "Size data not available for this species."
        })
    
    # Get size data for the top predicted species
    size_data = get_fish_size_data(top_species[0])
    
    # Define common objects with their sizes in cm
    common_objects = {
        "Credit Card": 8.5,
        "Smartphone": 15,
        "Pen": 14,
        "Pencil": 18,
        "Book": 25,
        "Tablet": 25,
        "Keyboard": 30,
        "Laptop": 35,
        "Ruler": 30,
        "Umbrella": 80,
        "Baseball Bat": 90,
        "Guitar": 100,
        "Backpack": 45
    }
    
    # Create a simpler, more reliable size comparison visualization
    st.subheader(f"Size of {top_species[0]}")

    # Create a container with a light background
    size_container = st.container()
    with size_container:
        # Add fish size information
        st.markdown(f"""
        <div style="
            background: rgba(240, 248, 255, 0.8);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border: 1px solid rgba(61, 93, 179, 0.2);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold; color: #2C4292;">{top_species[0]}</span>
                <span style="color: #2C4292;">{size_data["avg_length"]} cm (average)</span>
            </div>
            <div style="
                height: 40px;
                background: linear-gradient(90deg, #3D5DB3, #5A78C7);
                width: {min(100, size_data["avg_length"])}%;
                border-radius: 20px;
                position: relative;
                color: white;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 10px;
                font-weight: bold;
            ">
                {size_data["avg_length"]} cm
            </div>
            <div style="text-align: right; color: #2C4292; font-style: italic; margin-top: 5px;">
                Maximum length: {size_data["max_length"]} cm
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add comparison objects
        for obj in size_data["comparison_objects"]:
            if obj in common_objects:
                obj_size = common_objects[obj]
                st.markdown(f"""
                <div style="
                    background: rgba(255, 255, 255, 0.9);
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                    border: 1px solid rgba(61, 93, 179, 0.2);
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                ">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-weight: bold; color: #2C4292;">{obj}</span>
                        <span style="color: #2C4292;">{obj_size} cm</span>
                    </div>
                    <div style="
                        height: 40px;
                        background: linear-gradient(90deg, #5A78C7, #7A98E7);
                        width: {min(100, obj_size)}%;
                        border-radius: 20px;
                        position: relative;
                        color: white;
                        display: flex;
                        align-items: center;
                        justify-content: flex-end;
                        padding-right: 10px;
                        font-weight: bold;
                    ">
                        {obj_size} cm
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Add description
        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            border-left: 4px solid #3D5DB3;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h4 style="color: #2C4292; margin-top: 0;">Size Information</h4>
            <p style="color: #1A2A56; margin-bottom: 0;">
                <strong>Average Length:</strong> {size_data['avg_length']} cm<br>
                <strong>Maximum Length:</strong> {size_data['max_length']} cm<br><br>
                {size_data['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a geographic distribution map section before the save results section
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, rgba(44, 66, 146, 0.3), rgba(44, 66, 146, 0.1), transparent);
        border-radius: 8px;
        padding: 10px 15px;
        margin: 30px 0 20px 0;
        border-left: 4px solid {COLORS["primary"]};
    ">
        <h2 style="
            color: {COLORS["primary"]};
            margin: 0;
            font-size: 1.6em;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
        ">
            Geographic Distribution
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Function to get geographic distribution data for each species
    def get_geographic_distribution(species_name):
        """
        Get geographic distribution data for a fish species.
        
        Args:
            species_name (str): Name of the fish species
            
        Returns:
            dict: Dictionary with distribution data including regions and coordinates
        """
        # Define distribution data for different fish species
        distribution_data = {
            "Black Sea Sprat": {
                "regions": ["Black Sea", "Mediterranean Sea", "Eastern Atlantic"],
                "center": [41.0, 29.0],  # Black Sea center
                "zoom": 4,
                "description": "Primarily found in the Black Sea and parts of the Mediterranean Sea."
            },
            "Gilthead Bream": {
                "regions": ["Mediterranean Sea", "Eastern Atlantic"],
                "center": [38.0, 15.0],  # Mediterranean center
                "zoom": 4,
                "description": "Common throughout the Mediterranean Sea and along the Eastern Atlantic coast."
            },
            "Horse Mackerel": {
                "regions": ["Mediterranean Sea", "Eastern Atlantic", "North Sea"],
                "center": [45.0, 0.0],  # Eastern Atlantic
                "zoom": 3,
                "description": "Widely distributed in the Eastern Atlantic from Norway to South Africa, including the Mediterranean."
            },
            "Red Mullet": {
                "regions": ["Mediterranean Sea", "Black Sea", "Eastern Atlantic"],
                "center": [38.0, 15.0],  # Mediterranean center
                "zoom": 4,
                "description": "Common in the Mediterranean Sea and Eastern Atlantic from the English Channel to Senegal."
            },
            "Red Sea Bream": {
                "regions": ["Mediterranean Sea", "Eastern Atlantic"],
                "center": [38.0, 15.0],  # Mediterranean center
                "zoom": 4,
                "description": "Found in the Mediterranean Sea and Eastern Atlantic from Portugal to Angola."
            },
            "Sea Bass": {
                "regions": ["Mediterranean Sea", "Black Sea", "Eastern Atlantic"],
                "center": [45.0, 0.0],  # Eastern Atlantic
                "zoom": 3,
                "description": "Distributed throughout the Mediterranean, Black Sea, and Eastern Atlantic from Norway to Senegal."
            },
            "Shrimp": {
                "regions": ["Global - Tropical and Subtropical Waters"],
                "center": [0.0, 0.0],  # Global view
                "zoom": 2,
                "description": "Various shrimp species are found in tropical and subtropical waters worldwide, particularly in coral reef environments."
            },
            "Striped Red Mullet": {
                "regions": ["Mediterranean Sea", "Eastern Atlantic"],
                "center": [38.0, 15.0],  # Mediterranean center
                "zoom": 4,
                "description": "Common in the Mediterranean Sea and Eastern Atlantic from the English Channel to the Canary Islands."
            },
            "Trout": {
                "regions": ["North America", "Europe", "Asia"],
                "center": [45.0, 0.0],  # Northern Hemisphere
                "zoom": 2,
                "description": "Various trout species are found in cold, clear freshwater environments across North America, Europe, and Asia."
            }
        }
        
        # Return data for the requested species or a default
        return distribution_data.get(species_name, {
            "regions": ["Unknown"],
            "center": [0.0, 0.0],
            "zoom": 1,
            "description": "Geographic distribution data not available for this species."
        })
    
    # Get distribution data for the top predicted species
    distribution_data = get_geographic_distribution(top_species[0])
    
    # Create columns for the map and description
    map_col1, map_col2 = st.columns([3, 2])
    
    with map_col1:
        # Create a map using folium
        import folium
        from streamlit_folium import folium_static
        
        # Create a map centered on the species distribution
        m = folium.Map(
            location=distribution_data["center"],
            zoom_start=distribution_data["zoom"],
            tiles="CartoDB positron"
        )
        
        # Add a title to the map
        title_html = f'''
            <h3 style="text-align:center;margin-bottom:10px;">
                {top_species[0]} Distribution
            </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add highlighted regions based on the species
        if top_species[0] == "Shrimp":
            # For global species, highlight tropical regions
            folium.Rectangle(
                bounds=[[-23.5, -180], [23.5, 180]],
                color='#3D5DB3',
                fill=True,
                fill_color='#5A78C7',
                fill_opacity=0.3,
                tooltip="Tropical Regions"
            ).add_to(m)
        elif top_species[0] == "Trout":
            # For trout, highlight northern hemisphere freshwater regions
            for region, coords in {
                "North America": [[30, -130], [60, -60]],
                "Europe": [[40, -10], [70, 30]],
                "Asia": [[40, 30], [70, 150]]
            }.items():
                folium.Rectangle(
                    bounds=coords,
                    color='#3D5DB3',
                    fill=True,
                    fill_color='#5A78C7',
                    fill_opacity=0.3,
                    tooltip=region
                ).add_to(m)
        else:
            # For Mediterranean species
            if "Mediterranean Sea" in distribution_data["regions"]:
                folium.Rectangle(
                    bounds=[[30, -5], [45, 35]],
                    color='#3D5DB3',
                    fill=True,
                    fill_color='#5A78C7',
                    fill_opacity=0.3,
                    tooltip="Mediterranean Sea"
                ).add_to(m)
            
            # For Black Sea species
            if "Black Sea" in distribution_data["regions"]:
                folium.Rectangle(
                    bounds=[[40, 27], [47, 42]],
                    color='#3D5DB3',
                    fill=True,
                    fill_color='#5A78C7',
                    fill_opacity=0.3,
                    tooltip="Black Sea"
                ).add_to(m)
            
            # For Eastern Atlantic species
            if "Eastern Atlantic" in distribution_data["regions"]:
                folium.Rectangle(
                    bounds=[[30, -20], [60, 0]],
                    color='#3D5DB3',
                    fill=True,
                    fill_color='#5A78C7',
                    fill_opacity=0.3,
                    tooltip="Eastern Atlantic"
                ).add_to(m)
        
        # Display the map
        folium_static(m)
    
    with map_col2:
        # Use Streamlit's native components for a more reliable display
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(240, 248, 255, 0.9), rgba(230, 240, 255, 0.9));
             border-radius: 12px; padding: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
             border: 1px solid rgba(61, 93, 179, 0.2);">
        """, unsafe_allow_html=True)
        
        # Add the header using Streamlit's native components
        st.subheader("Distribution Regions")
        
        # Create a container for the regions
        for region in distribution_data["regions"]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 12px; height: 12px; background: linear-gradient(135deg, #3D5DB3, #5A78C7);
                     border-radius: 50%; margin-right: 10px;"></div>
                <span style="color: #1A2A56; font-weight: 500;">{region}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Add the description header
        st.subheader("Description")
        
        # Add the description in a styled container
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.7); border-radius: 8px; padding: 15px;
             border-left: 4px solid #3D5DB3; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
            <p style="color: #1A2A56; margin: 0; line-height: 1.5; font-size: 1.05em;">
                {distribution_data["description"]}
            </p>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Now add the save results section after the map
    # (existing code for save results section)
    
    # Add the save image button at the bottom only
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, rgba(44, 66, 146, 0.3), rgba(44, 66, 146, 0.1), transparent);
        border-radius: 8px;
        padding: 10px 15px;
        margin: 30px 0 20px 0;
        border-left: 4px solid {COLORS["primary"]};
    ">
        <h3 style="
            color: {COLORS["primary"]};
            margin: 0;
            font-size: 1.3em;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);
        ">
            Save Your Results
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Add custom CSS to center the button better
    st.markdown("""
    <style>
        /* Center the download button container */
        div[data-testid="column"]:nth-of-type(2) {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        /* Make the button wider and more prominent */
        div[data-testid="stDownloadButton"] {
            width: 100%;
            max-width: 300px;
            margin: 0 auto;
        }
        
        /* Style the button itself */
        div[data-testid="stDownloadButton"] button {
            width: 100%;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        
        /* Button hover effect */
        div[data-testid="stDownloadButton"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }
        
        /* Center the info message */
        div.stAlert {
            text-align: center;
            max-width: 500px;
            margin: 15px auto;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Use more extreme column ratios for better centering
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col2:
        # Save as Image - Create a downloadable image instead of using screenshot
        if image is not None:
            # Create a figure with the prediction results
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Display the image
            ax.imshow(np.array(image))
            ax.axis('off')
            
            # Add text with prediction results
            result_text = f"Predicted: {top_species[0]}\nConfidence: {top_scores[0]*100:.1f}%"
            ax.text(10, 30, result_text, fontsize=12, color='white', 
                    bbox=dict(facecolor=COLORS["primary"], alpha=0.8))
            
            # Save the figure to a buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            
            # Create a download button with a unique key to prevent the error
            unique_key = f"download_image_{abs(hash(top_species[0])) % 10000}"
            st.download_button(
                label="📸 Save as Image",
                data=buf,
                file_name=f"fish_classification_{top_species[0].replace(' ', '_')}.png",
                mime="image/png",
                help="Download the classification results as an image",
                key=unique_key
            )
            plt.close(fig)
            
            # Add a note about the download
            st.info("Click the button above to save your classification results as an image.")
        else:
            # Fallback if image is not available
            st.button("📸 Save as Image", help="Image not available", disabled=True)
    
    # Close the ocean background div
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    return ""

def create_water_bubble_sound(duration=3.0, sample_rate=22050):
    """
    Generate a water bubble sound effect programmatically.
    
    Args:
        duration (float): Duration of the sound in seconds
        sample_rate (int): Sample rate of the audio
        
    Returns:
        str: Base64 encoded WAV audio data
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate bubble sounds by combining sine waves with random parameters
    audio = np.zeros_like(t)
    
    # Create multiple bubbles
    num_bubbles = int(duration * 10)  # 10 bubbles per second on average
    
    for _ in range(num_bubbles):
        # Random parameters for each bubble
        start_time = np.random.uniform(0, duration * 0.9)
        bubble_duration = np.random.uniform(0.05, 0.2)
        frequency = np.random.uniform(100, 2000)
        amplitude = np.random.uniform(0.1, 0.5)
        
        # Create bubble envelope (attack and decay)
        bubble_samples = int(bubble_duration * sample_rate)
        envelope = np.exp(-np.linspace(0, 10, bubble_samples))
        
        # Create bubble sound
        start_idx = int(start_time * sample_rate)
        end_idx = min(start_idx + bubble_samples, len(audio))
        
        # Ensure we don't go out of bounds
        if start_idx < len(audio) and end_idx <= len(audio):
            bubble_t = t[start_idx:end_idx] - t[start_idx]
            bubble = amplitude * np.sin(2 * np.pi * frequency * bubble_t) * envelope[:end_idx-start_idx]
            audio[start_idx:end_idx] += bubble
    
    # Add gentle water background
    water_noise = np.random.normal(0, 0.05, len(t))
    water_noise = np.convolve(water_noise, np.ones(1000)/1000, mode='same')  # Smooth the noise
    
    # Combine bubble sounds with water background
    audio = audio + water_noise
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio)) * 0.7
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    
    # Encode as base64
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return audio_base64

def create_gentle_splash_sound(duration=2.0, sample_rate=22050):
    """
    Generate a gentle splash sound effect programmatically.
    
    Args:
        duration (float): Duration of the sound in seconds
        sample_rate (int): Sample rate of the audio
        
    Returns:
        str: Base64 encoded WAV audio data
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate white noise
    noise = np.random.normal(0, 1, len(t))
    
    # Create splash envelope
    attack_time = 0.05
    decay_time = duration - attack_time
    
    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)
    
    attack = np.linspace(0, 1, attack_samples)
    decay = np.exp(-np.linspace(0, 5, decay_samples))
    
    envelope = np.concatenate([attack, decay])
    if len(envelope) > len(noise):
        envelope = envelope[:len(noise)]
    
    # Apply envelope to noise
    audio = noise * envelope
    
    # Apply bandpass filter (simple implementation)
    # Focus on frequencies that sound like water (500-2000 Hz)
    filtered_audio = np.zeros_like(audio)
    for freq in [500, 800, 1200, 1600, 2000]:
        filtered_audio += np.sin(2 * np.pi * freq * t) * audio
    
    # Normalize audio
    filtered_audio = filtered_audio / np.max(np.abs(filtered_audio)) * 0.7
    
    # Convert to 16-bit PCM
    audio_int16 = (filtered_audio * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_int16)
    buffer.seek(0)
    
    # Encode as base64
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return audio_base64

def play_audio(audio_base64, autoplay=True):
    """
    Create an HTML audio element to play the provided audio data.
    
    Args:
        audio_base64 (str): Base64 encoded audio data
        autoplay (bool): Whether to autoplay the audio
        
    Returns:
        str: HTML for audio element
    """
    audio_html = f"""
    <audio {' autoplay' if autoplay else ''} style="display:none;">
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

def get_conservation_status(species_name):
    """
    Get the conservation status for a fish species.
    
    Args:
        species_name (str): Name of the fish species
        
    Returns:
        tuple: (status, color, description)
    """
    # Conservation status database
    # Based on IUCN Red List categories: https://www.iucnredlist.org/
    conservation_data = {
        "Black Sea Sprat": ("LC", "#4CAF50", "Least Concern", 
                           "Population is stable and widespread in the Black Sea."),
        "Gilthead Bream": ("LC", "#4CAF50", "Least Concern", 
                          "Common throughout the Mediterranean and farmed extensively."),
        "Horse Mackerel": ("LC", "#4CAF50", "Least Concern", 
                          "Abundant in the Eastern Atlantic and Mediterranean."),
        "Red Mullet": ("LC", "#4CAF50", "Least Concern", 
                      "Common throughout its range with stable populations."),
        "Red Sea Bream": ("NT", "#FF9800", "Near Threatened", 
                         "Declining in some areas due to overfishing."),
        "Sea Bass": ("NT", "#FF9800", "Near Threatened", 
                    "Wild populations are declining due to fishing pressure."),
        "Shrimp": ("LC", "#4CAF50", "Least Concern", 
                  "Many species are abundant, though some are threatened by habitat loss."),
        "Striped Red Mullet": ("LC", "#4CAF50", "Least Concern", 
                              "Widespread in the Mediterranean with stable populations."),
        "Trout": ("VU", "#F44336", "Vulnerable", 
                 "Some wild populations are declining due to habitat degradation and climate change.")
    }
    
    # Default status if species not found
    default = ("DD", "#9E9E9E", "Data Deficient", "Not enough data to determine conservation status.")
    
    return conservation_data.get(species_name, default)

def create_conservation_status_badge(species_name):
    """
    Create a visual badge showing the conservation status of a fish species.
    
    Args:
        species_name (str): Name of the fish species
        
    Returns:
        str: HTML for the conservation status badge
    """
    status_code, color, status_name, description = get_conservation_status(species_name)
    
    # Create the badge HTML with enhanced styling
    badge_html = f"""
    <div style="
        display: inline-flex;
        align-items: center;
        margin: 10px 0;
        background: linear-gradient(135deg, {color}15, {color}05);
        padding: 12px 15px;
        border-radius: 12px;
        border-left: 4px solid {color};
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 100%;
    ">
        <div style="
            background-color: {color};
            color: white;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.1em;
            margin-right: 15px;
            box-shadow: 0 3px 6px rgba(0,0,0,0.2);
            flex-shrink: 0;
        ">{status_code}</div>
        <div style="flex-grow: 1;">
            <div style="
                font-weight: bold; 
                color: {color}; 
                font-size: 1.2em;
                margin-bottom: 5px;
                text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9);
            ">{status_name}</div>
            <div style="
                font-size: 0.95em; 
                color: #444;
                line-height: 1.4;
            ">{description}</div>
        </div>
    </div>
    """
    return badge_html

def create_conservation_legend():
    """
    Create a legend explaining the conservation status codes.
    
    Returns:
        str: HTML for the conservation status legend
    """
    # Create a container for the legend items
    legend_container = st.container()
    
    with legend_container:
        # Create a 3x3 grid for status items
        cols = st.columns(3)
        
        # Define all status types
        statuses = [
            ("EX", "#000000", "Extinct", "No known living individuals"),
            ("EW", "#592720", "Extinct in the Wild", "Survives only in captivity"),
            ("CR", "#B71C1C", "Critically Endangered", "Extremely high risk of extinction"),
            ("EN", "#D32F2F", "Endangered", "High risk of extinction"),
            ("VU", "#F44336", "Vulnerable", "High risk of endangerment"),
            ("NT", "#FF9800", "Near Threatened", "Likely to become endangered soon"),
            ("LC", "#4CAF50", "Least Concern", "Widespread and abundant"),
            ("DD", "#9E9E9E", "Data Deficient", "Not enough data to assess"),
            ("NE", "#607D8B", "Not Evaluated", "Not yet evaluated against criteria")
        ]
        
        # Distribute status items across columns
        for i, (code, color, name, desc) in enumerate(statuses):
            col_idx = i % 3
            with cols[col_idx]:
                # Create a container with background color
                st.markdown(f"""
                <div style="
                    background-color: {color}15;
                    border-radius: 5px;
                    padding: 8px;
                    margin-bottom: 10px;
                    border-left: 3px solid {color};
                ">
                    <div style="display: flex; align-items: center;">
                        <div style="
                            background-color: {color};
                            color: white;
                            border-radius: 50%;
                            width: 25px;
                            height: 25px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-weight: bold;
                            font-size: 0.8em;
                            margin-right: 8px;
                            flex-shrink: 0;
                        ">{code}</div>
                        <div>
                            <div style="font-weight: bold; font-size: 0.9em;">{name}</div>
                            <div style="font-size: 0.8em; color: #555;">{desc}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Add attribution
        st.markdown("""
        <div style="font-size: 0.8em; margin-top: 10px; text-align: center; color: #666;">
            Based on <a href="https://www.iucnredlist.org/" target="_blank">IUCN Red List</a> categories
        </div>
        """, unsafe_allow_html=True)
    
    # Return empty string since we're using st.markdown directly
    return ""

def main():
    """
    Main function to run the Streamlit application.
    """
    # Apply custom styling
    add_bg_from_pattern()
    apply_custom_styles()
    
    # Create animated title with swimming fish
    st.markdown(create_swimming_fish_animation(), unsafe_allow_html=True)
    
    # Use typing animation for the description
    description = create_typing_animation(
        "This application uses a ConvNeXt model to classify fish species from images. "
        "Upload an image of a fish, and our model will predict its species."
    )
    st.markdown(
        f"<div style='text-align: center; font-size: 1.2em; margin-bottom: 30px;'>{description}</div>",
        unsafe_allow_html=True
    )
    
    # Display information in the sidebar
    with st.sidebar:
        st.markdown("<h2 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>About</h2>", 
                    unsafe_allow_html=True)
        
        # Add floating animation to the fish icon
        st.markdown(create_floating_animation('<span style="font-size: 2em; color: white;">🐟</span>'), 
                    unsafe_allow_html=True)
        
        # Use regular Streamlit components for text to avoid HTML rendering issues
        st.write("This application uses a ConvNeXt model trained on a dataset of fish images.")
        
        st.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>Fish Species</h3>", 
                    unsafe_allow_html=True)
        
        # Clean up species names
        clean_species = [
            "Black Sea Sprat",
            "Gilthead Bream",
            "Horse Mackerel",
            "Red Mullet",
            "Red Sea Bream",
            "Sea Bass",
            "Shrimp",
            "Striped Red Mullet",
            "Trout"
        ]
        
        # Use regular Streamlit components for the list
        for name in clean_species:
            st.write(f"• {name}")
        
        st.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
        
        # Sound effects toggle
        st.markdown("<h3 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>App Settings</h3>", 
                    unsafe_allow_html=True)
        
        enable_sound = st.checkbox("Enable Sound Effects", value=True, 
                                  help="Play gentle water sounds when results appear")
        
        # Modify the sound type selection in the sidebar
        st.sidebar.subheader("Sound Type")
        sound_type = st.sidebar.radio(
            "Choose sound effect",
            ["Bubbles", "None"],  # Removed "Splash" option
            help="Select the sound effect to play when classification is complete"
        )
        
        # Option to upload a custom model file
        st.markdown("<h3 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>Advanced Settings</h3>", 
                    unsafe_allow_html=True)
        
        uploaded_model = st.file_uploader("Upload model file (optional)", type=["pkl"])
        
        if uploaded_model is not None:
            # Save the uploaded model to a temporary file
            with open("temp_model.pkl", "wb") as f:
                f.write(uploaded_model.getbuffer())
            
            # Use the uploaded model
            model_path = "temp_model.pkl"
            st.success("Custom model uploaded successfully!")
        else:
            model_path = None  # Use default paths
        
        # Add device selection option
        st.markdown("<h3 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>Hardware Settings</h3>", 
                    unsafe_allow_html=True)
        
        # Check available devices
        cuda_available = torch.cuda.is_available()
        
        # Check for Intel XPU (Arc Graphics)
        xpu_available = False
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                xpu_available = True
        except ImportError:
            pass
        
        # Create device options based on availability
        device_options = ["CPU"]
        if cuda_available:
            device_options.append("NVIDIA GPU")
        if xpu_available:
            device_options.append("Intel GPU")
            
        device_help = "Select the device to use for inference."
        
        selected_device = st.radio(
            "Compute Device",
            device_options,
            help=device_help
        )
        
        # Display GPU info if available
        if cuda_available and selected_device == "NVIDIA GPU":
            with st.expander("NVIDIA GPU Information"):
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
                st.write(f"GPU: {gpu_name}")
                st.write(f"Memory: {gpu_memory:.2f} GB")
                st.write(f"CUDA Version: {torch.version.cuda}")
        
        # Display Intel GPU info if available
        if xpu_available and selected_device == "Intel GPU":
            with st.expander("Intel GPU Information"):
                try:
                    xpu_name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, 'get_device_name') else "Intel GPU"
                    st.write(f"GPU: {xpu_name}")
                    st.write("Intel Extension for PyTorch is available")
                except Exception as e:
                    st.write("Intel GPU detected but couldn't retrieve details")
    
    # Set up device for inference based on selection
    if 'selected_device' in locals():
        if selected_device == "NVIDIA GPU" and cuda_available:
            device = torch.device("cuda:0")
            st.sidebar.success("Using NVIDIA GPU for inference")
        elif selected_device == "Intel GPU" and xpu_available:
            device = torch.device("xpu:0")
            st.sidebar.success("Using Intel GPU for inference")
        else:
            device = torch.device("cpu")
            st.sidebar.info("Using CPU for inference")
    else:
        device = torch.device("cpu")
        st.sidebar.info("Using CPU for inference")
    
    # Load model and class names
    with st.spinner("Loading model... This may take a moment."):
        model, class_names = load_single_model(model_path)
    
    if not model or not class_names:
        st.error("Failed to load model. Please check the model path or upload a model file.")
        
        # Show instructions for fixing the model issue
        with st.expander("How to fix this issue"):
            st.markdown("""
            ### How to Fix Model Loading Issues
            
            1. **Check Model Location**: Make sure your `vit_model.pkl` file is in one of these locations:
               - `models/vit_model.pkl` (in the same directory as this script)
               - `MLDL/models/vit_model.pkl`
            
            2. **Upload Model File**: You can upload your model file using the file uploader in the sidebar.
            
            3. **Check Model Format**: Ensure your model file is a pickle file containing:
               - A PyTorch ViT model
               - Optionally, a 'class_names' key with a list of class names
            """)
        return
    
    # Add separator between sections
    st.markdown(create_separator(), unsafe_allow_html=True)
    
    # Animated title for upload section
    st.markdown(create_animated_title("Upload a Fish Image"), unsafe_allow_html=True)
    
    # Animated explanation with color transitions
    explanation_text = "Please upload an image of a fish for classification. The model works best with clear, well-lit images where the fish is the main subject."
    st.markdown(create_animated_explanation(explanation_text), unsafe_allow_html=True)
    
    # Create columns for upload and sample options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader with custom styling
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    with col2:
        # Sample image option
        st.markdown("<p style='color: #2C4292;'>Or try a sample image:</p>", unsafe_allow_html=True)
        
        # Custom checkbox with blue text
        use_sample = st.checkbox("Use a sample image", key="sample_checkbox")
        
        if use_sample:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Define sample image paths with absolute paths
            sample_images = {
                "Sea Bass": os.path.join(script_dir, "sample_images", "sea_bass.jpg"),
                "Trout": os.path.join(script_dir, "sample_images", "trout.jpg"),
                "Shrimp": os.path.join(script_dir, "sample_images", "shrimp.jpg")
            }
            
            # Also try relative paths as fallback
            relative_sample_images = {
                "Sea Bass": "sample_images/sea_bass.jpg",
                "Trout": "sample_images/trout.jpg",
                "Shrimp": "sample_images/shrimp.jpg"
            }
            
            selected_sample = st.selectbox("Select a sample image:", list(sample_images.keys()))
            
            # Try absolute path first
            sample_path = sample_images.get(selected_sample, "")
            
            # If absolute path doesn't exist, try relative path
            if not os.path.exists(sample_path):
                sample_path = relative_sample_images.get(selected_sample, "")
            
            # If either path exists, use it
            if os.path.exists(sample_path):
                # Enhanced 3D info message with popping text
                st.markdown(
                    f"""
                    <div style="
                        background-color: #cfe2ff; 
                        padding: 15px; 
                        border-radius: 8px; 
                        margin: 15px 0;
                        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15), 
                                    inset 0 -3px 0 rgba(0, 0, 0, 0.1),
                                    0 -1px 0 rgba(255, 255, 255, 0.5) inset;
                        border: 1px solid rgba(0, 100, 200, 0.2);
                        transform: translateZ(0);
                        position: relative;
                    ">
                        <p style="
                            color: {COLORS['primary']}; 
                            margin: 0;
                            font-weight: 500;
                            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
                            transform: translateZ(10px);
                            position: relative;
                        ">Selected sample image: {selected_sample}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Load the sample image
                uploaded_file = sample_path
            else:
                # Print debugging information
                st.warning(f"Sample image not found: {sample_path}")
                st.info(f"Current working directory: {os.getcwd()}")
                st.info(f"Script directory: {script_dir}")
                st.info("Please ensure the sample_images directory contains sea_bass.jpg, trout.jpg, and shrimp.jpg")
    
    # Create a placeholder for sound effects
    sound_effect_placeholder = st.empty()
    
    # Process the uploaded image
    if uploaded_file is not None:
        try:
            # Load the image
            if isinstance(uploaded_file, str):
                # It's a file path
                image = Image.open(uploaded_file).convert('RGB')
            else:
                # It's a file upload
                image = Image.open(uploaded_file).convert('RGB')
            
            # Display the uploaded image with animation
            image_container = create_animated_container("""
            <div style="
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
                padding: 10px;
                margin: 20px 0;
            ">
            """, "scale-in")
            st.markdown(image_container, unsafe_allow_html=True)
            
            st.image(image, caption="Uploaded Image", width=400)  # Fixed width of 400px
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a button to trigger prediction with animation
            classify_button = st.button("🔍")
            
            if classify_button:
                # Create a progress bar for visual feedback with explicit styling
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Apply explicit styling to ensure visibility
                status_text.markdown(f'<p style="color: {COLORS["primary"]}; font-weight: bold; text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);">Preprocessing image...</p>', unsafe_allow_html=True)
                progress_bar.progress(20)
                time.sleep(0.5)
                
                # Preprocess the image
                img_tensor = preprocess_image(image)
                
                status_text.markdown(f'<p style="color: {COLORS["primary"]}; font-weight: bold; text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);">Running model...</p>', unsafe_allow_html=True)
                progress_bar.progress(40)
                time.sleep(0.5)
                
                # Make prediction with single model
                predictions = single_model_predict(model, img_tensor, device)
                
                # Update progress
                status_text.markdown(f'<p style="color: {COLORS["primary"]}; font-weight: bold; text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);">Analyzing results...</p>', unsafe_allow_html=True)
                progress_bar.progress(70)
                time.sleep(0.5)
                
                # Get top 3 predictions
                predictions_np = predictions.numpy()[0]  # Convert to numpy and get first item

                # Ensure predictions sum to 1 and cap maximum confidence
                predictions_np = predictions_np / np.sum(predictions_np)
                max_idx = np.argmax(predictions_np)
                if predictions_np[max_idx] > 0.95:
                    # Cap maximum confidence at 95%
                    excess = predictions_np[max_idx] - 0.95
                    predictions_np[max_idx] = 0.95
                    
                    # Redistribute excess probability to other classes
                    non_max_indices = np.arange(len(predictions_np)) != max_idx
                    if np.sum(predictions_np[non_max_indices]) > 0:
                        predictions_np[non_max_indices] += excess * (predictions_np[non_max_indices] / np.sum(predictions_np[non_max_indices]))

                top_indices = np.argsort(predictions_np)[-3:][::-1]
                
                # Map the class names to display names
                top_species = [get_display_name(class_names[i]) for i in top_indices]
                top_scores = [predictions_np[i] for i in top_indices]
                
                # Update progress
                progress_bar.progress(100)
                status_text.markdown(f'<p style="color: {COLORS["primary"]}; font-weight: bold; text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.9), 0 0 1px rgba(0, 0, 0, 0.1);">Classification complete!</p>', unsafe_allow_html=True)
                time.sleep(0.5)
                
                # Play sound effect if enabled
                if enable_sound and sound_type != "None":
                    if sound_type == "Bubbles":
                        audio_base64 = create_water_bubble_sound()
                    else:  # Splash
                        audio_base64 = create_gentle_splash_sound()
                    
                    # Play the sound
                    sound_effect_placeholder.markdown(
                        play_audio(audio_base64, autoplay=True),
                        unsafe_allow_html=True
                    )
                
                # Display results using our custom styled results display
                create_results_display(top_species, top_scores, image)
        
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
            st.info("Please upload a valid image file or try a different image.")

if __name__ == "__main__":
    main()
