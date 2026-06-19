import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import StringVar, DoubleVar
from ml_model import CVDRiskModel

# Initialize model
model = CVDRiskModel()
model.load_model()

# Dummy model names for demonstration
MODEL_NAMES = [
    "K-Nearest Neighbours",
    "Support Vector Machine",
    "Logistic Regression",
    "Random Forest Classifier"
]

# Dummy prediction function (replace with your real logic)
def get_predictions(user_data):
    # Example: Replace with your actual model predictions
    # Return a dict: {model_name: True/False}
    # True = Heart disease detected, False = No heart disease
    # For demonstration, randomize or use your model's output
    return {
        "K-Nearest Neighbours": False,
        "Support Vector Machine": False,
        "Logistic Regression": True,
        "Random Forest Classifier": True
    }

def show_results(results):
    for idx, model in enumerate(MODEL_NAMES):
        card = cards[idx]
        has_disease = results[model]
        card.configure(bootstyle="danger" if has_disease else "success")
        card_title_vars[idx].set(model)
        card_result_vars[idx].set("Heart disease detected." if has_disease else "No heart disease detected.")

def on_assess():
    # Collect user input here (expand as needed)
    user_data = {}  # TODO: Collect from input fields
    results = get_predictions(user_data)
    show_results(results)

# --- GUI Setup ---
app = tb.Window(themename="cyborg")  # or another theme
app.title("Heart Disease Prediction")
app.geometry("800x600")
app.configure(bg="#1abc9c")  # Turquoise background

# Header
tb.Label(app, text="Heart Disease Prediction", font=("Arial", 24, "bold"), bootstyle="info", background="#1abc9c").pack(pady=(30, 0))
tb.Label(app, text="Predicted Results", font=("Arial", 14), bootstyle="info", background="#1abc9c").pack(pady=(0, 20))

# Card grid frame
card_frame = tb.Frame(app, bootstyle="secondary", padding=20)
card_frame.pack(expand=True, fill=BOTH, padx=40, pady=20)

cards = []
card_title_vars = []
card_result_vars = []

for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        card_title = StringVar()
        card_result = StringVar()
        card = tb.Frame(card_frame, bootstyle="success", width=250, height=120, padding=15)
        card.grid(row=i, column=j, padx=30, pady=30, sticky="nsew")
        tb.Label(card, textvariable=card_title, font=("Arial", 13, "bold"), bootstyle="inverse").pack(anchor="w")
        tb.Label(card, textvariable=card_result, font=("Arial", 11), bootstyle="inverse").pack(anchor="w", pady=(10,0))
        cards.append(card)
        card_title_vars.append(card_title)
        card_result_vars.append(card_result)
        # Set initial values
        card_title.set(MODEL_NAMES[idx])
        card_result.set("No heart disease detected.")

# Make grid cells expand
for i in range(2):
    card_frame.grid_rowconfigure(i, weight=1)
    card_frame.grid_columnconfigure(i, weight=1)

# Assess button
tb.Button(app, text="Assess Risk", bootstyle="primary", command=on_assess).pack(pady=20)

app.mainloop()